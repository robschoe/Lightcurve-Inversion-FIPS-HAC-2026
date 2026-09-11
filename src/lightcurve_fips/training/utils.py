import trimesh
import numpy as np
from pathlib import Path
import torch
from skimage import measure
import re
import torch.nn.functional as F

R_max = 6

def parse_radius_from_stl(stl_path):
    """Simple function to extract radius written in the name of a file"""
    m = re.search(r"radius([0-9]+(?:\.[0-9]+)?)", stl_path.name)

    if m is None:
        raise ValueError(f"Could not parse radius from {stl_path.name}")

    return float(m.group(1))

def positional_encoding(points, num_freqs=6):
    """Expand points by sin and cos of their points times given frequencies"""
    enc = [points]

    for i in range(num_freqs):
        freq = 2.0 ** i
        enc.append(torch.sin(freq * torch.pi * points))
        enc.append(torch.cos(freq * torch.pi * points))

    return torch.cat(enc, dim=-1)

def reconstruct_sdf(model, lc, radius, grid_extent=1.1, res=128, chunk=200000, device=None):
    """Evaluate a trained model on a 3D grid. The given Lightcurves and radius define the model."""
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    try:
        #Convert the radius to a float tensor.
        radius = torch.as_tensor(radius, dtype=torch.float32, device=device)

        #Make sure the radius is only a single number.
        if radius.dim() == 0:
            radius = radius.unsqueeze(0)

        radius = radius.reshape(-1)

        #Only one Lightcurve input at a time.
        if lc.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, but got {lc.shape[0]}."
            )

        if radius.numel() != 1:
            raise ValueError(
                f"Expected exactly one radius, but got shape {tuple(radius.shape)}."
            )

        lc = lc.to(device)

        #Create a 3D grid that is evenly spaced in the given extent with the given resolution.
        coords = torch.linspace(-grid_extent, grid_extent, res, device=device, dtype=torch.float32,)

        X, Y, Z = torch.meshgrid(
            coords, coords, coords,
            indexing="ij"
        )

        grid_points = torch.stack([X, Y, Z], dim=-1)
        grid_points = grid_points.reshape(-1, 3)

        sdf_values = []

        with torch.inference_mode():
            #Query each point in the grid and collect the models output.
            for start in range(0, len(grid_points), chunk):
                points = grid_points[start:start + chunk].unsqueeze(0)

                sdf = model(lc, points, radius)

                sdf_values.append(sdf.reshape(-1).float().cpu())

        sdf_grid = torch.cat(sdf_values, dim=0)
        sdf_grid = sdf_grid.reshape(res, res, res).numpy()

        return sdf_grid
    finally:
        model.train(was_training)

def sdf_to_stl(sdf, out_path, radius, grid_extent=1.1, padding_voxels=2, keep_largest_component=True, try_fill_holes=True, outside_positive=None):
    """Extract the STL file described by a set of sdf point-value sets."""
    if torch.is_tensor(sdf):
        sdf = sdf.detach().cpu().numpy()

    sdf = np.asarray(sdf, dtype=np.float32)

    if torch.is_tensor(radius):
        if radius.numel() != 1:
            raise ValueError(f"Expected one scalar radius, got shape {tuple(radius.shape)}.")
        radius = radius.detach().cpu().item()

    radius = float(radius)

    #Marching cubes expects a 3D scalar field.
    if sdf.ndim != 3:
        raise ValueError(f"Expected SDF shape [res, res, res], got {sdf.shape}")

    if not np.isfinite(sdf).all():
        raise ValueError("SDF contains NaN or Inf values.")

    if not (sdf.shape[0] == sdf.shape[1] == sdf.shape[2]):
        raise ValueError(f"Expected a cubic SDF grid, got {sdf.shape}.")

    res = sdf.shape[0]

    if res < 2:
        raise ValueError("SDF resolution must be at least 2.")

    if padding_voxels < 0:
        raise ValueError("padding_voxels must be non-negative.")

    if not (float(sdf.min()) <= 0.0 <= float(sdf.max())):
        print("Warning: SDF does not cross zero. "
            f"min={sdf.min():.6f}, max={sdf.max():.6f}")
        return None

    #Define boundary values to determine if the outside region is positive or negative. The standard is positive.
    boundary_values = np.concatenate([
        sdf[0, :, :].ravel(),
        sdf[-1, :, :].ravel(),
        sdf[:, 0, :].ravel(),
        sdf[:, -1, :].ravel(),
        sdf[:, :, 0].ravel(),
        sdf[:, :, -1].ravel(),
    ])

    if outside_positive is None:
        outside_positive = np.median(boundary_values) >= 0.0

    # Use a sufficiently large constant SDF value outside the original grid.
    max_abs_sdf = float(np.max(np.abs(sdf)))
    outside_value = max(2.0 * max_abs_sdf, 1.0)

    if not outside_positive:
        outside_value = -outside_value

    sdf_padded = np.pad(
        sdf,
        pad_width=padding_voxels,
        mode="constant",
        constant_values=outside_value
    )

    if not (float(sdf_padded.min()) <= 0.0 <= float(sdf_padded.max())):
        print("Warning: padded SDF does not cross zero.")
        return None

    #Determine the granularity for marching cubes.
    voxel_size = (2.0 * grid_extent) / (res - 1)

    try:
        #Use marching cubes to extract surface of the sdf sets.
        verts, faces, normals, values = measure.marching_cubes(
            sdf_padded,
            level=0.0,
            spacing=(
                voxel_size,
                voxel_size,
                voxel_size,
            ),
        )
    except ValueError as exc:
        print(f"Marching Cubes failed: {exc}")
        return None

    grid_origin = -grid_extent - padding_voxels * voxel_size
    verts += grid_origin

    verts[:, 0] *= radius
    verts[:, 1] *= radius

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=True
    )

    #Remove unnecessary triangles.
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) == 0:
        print("Warning: Mesh contains no valid faces after cleanup.")
        return None

    if keep_largest_component:
        components = mesh.split(only_watertight=False)

        if len(components) > 1:
            mesh = max(
                components,
                key=lambda component: component.area
            )

    if try_fill_holes and not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    trimesh.repair.fix_normals(
        mesh,
        multibody=True
    )

    #Create new directory if necessary and save the mesh.
    out_path = Path(out_path)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mesh.export(out_path)

    return mesh

def stl_to_sdf_grid(stl_path, radius, resolution=128, tau=0.1, grid_extent=1.1,):
    """Convert a given STL file to a SDF (signed distance field) grid"""
    radius = float(radius)

    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}.")

    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}.")

    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}.")

    #Load the STL file.
    mesh = trimesh.load(stl_path)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("The loaded scene does not contain any geometry.")

        #Merge all geometries in the scene into one mesh.
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    mesh = mesh.copy()

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("The input mesh contains no valid vertices or faces.")

    #Normalize z to [-1,1]
    z_min = mesh.vertices[:, 2].min()
    z_max = mesh.vertices[:, 2].max()
    z_range = z_max - z_min

    if z_range <= 0.0:
        raise ValueError("Cannot normalize z coordinates because the mesh has zero height.")

    mesh.vertices[:, 2] = 2.0 * (mesh.vertices[:, 2] - z_min) / z_range - 1.0

    #Normalize x and y by the given radius.
    mesh.vertices[:, 0] /= radius
    mesh.vertices[:, 1] /= radius

    #Sample in [-1,1]^3
    coords = np.linspace(-grid_extent, grid_extent, resolution, dtype=np.float32)

    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    points = np.stack(
        [X.ravel(), Y.ravel(), Z.ravel()],
        axis=1
    )

    #Compute a signed distance value for each given point.
    sdf = signed_distance_chunked(mesh, points)

    sdf = np.clip(sdf, -tau, tau)
    sdf = sdf / tau

    sdf = sdf.reshape(resolution, resolution, resolution).astype(np.float32)

    return sdf

def get_original_stl(folder):
    """Simple function to extract stl files"""
    stl_files = sorted(
        path
        for path in folder.glob("asteroid*.stl")
        if "_reconstructed" not in path.stem
        and "_repaired" not in path.stem
        and "_temporary" not in path.stem
    )

    return stl_files[0] if stl_files else None

def load_as_mesh(path):
    """Loads stl as trimesh"""
    loaded = trimesh.load(path, force="mesh")

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]

        if not meshes:
            raise ValueError("Scene does not contain Trimesh.")

        return trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"Unexpected Type: {type(loaded)}"
        )

    return loaded