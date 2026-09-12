import numpy as np
import trimesh
from pathlib import Path
from numpy import random
from opensimplex import OpenSimplex
import os

def add_noise_and_craters(mesh, noise_amp=0.02, noise_scale=2.0, n_craters=30, crater_max_radius=0.25, crater_max_depth=0.2):
    """Add surface noise and random craters to a mesh."""

    #Change copy to keep original unchanged.
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals.copy()

    gen = OpenSimplex(random.randint(0,2**31-1))

    r = np.linalg.norm(verts, axis=1)
    dirs = verts / (r[:, None] + 1e-12)

    def noise_val(p, scale):
        """Evaluate 3D noise at one scaled direction."""
        return gen.noise3(p[0]*scale, p[1]*scale, p[2]*scale)

    noises = np.array([noise_val(d, noise_scale) for d in dirs], dtype=np.float32)
    verts += (normals * (noise_amp * noises)[:, None])

    crater_centers = []
    crater_radii = []
    crater_depths = []
    for _ in range(n_craters):
        #Sample a uniformly distributed direction on the unit sphere.
        theta = random.random() * 2.0 * np.pi
        u = random.uniform(-1.0, 1.0)

        direction = np.array([
            np.cos(theta) * np.sqrt(1.0 - u * u),
            np.sin(theta) * np.sqrt(1.0 - u * u),
            u,
        ], dtype=np.float32)    

        #Store random crater position, angular radius, and relative depth.
        crater_centers.append(direction)
        crater_radii.append(
            random.uniform(0.03, crater_max_radius)
        )
        crater_depths.append(
            random.uniform(0.02, crater_max_depth)
        )
        
    crater_centers = np.array(crater_centers)

    #Calculate angular distances between vertices and crater centers.
    dots = np.dot(dirs, crater_centers.T)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)

    for j in range(len(crater_centers)):
        ang = angles[:, j]

        mask = ang < crater_radii[j]

        if not np.any(mask):
            continue

        frac = 1.0 - (ang[mask] / crater_radii[j])
        falloff = 0.5 * (1 - np.cos(frac * np.pi))
        depth_vals = crater_depths[j] * falloff

        #Move crater vertices inward.
        verts[mask] -= (dirs[mask] * (depth_vals[:, None] * r[mask, None]))     #push vertices inside

    #Apply one optional Laplacian smoothing step.
    try:
        from scipy import sparse
        
        faces = mesh.faces
        n = len(verts)

        I = np.hstack([faces[:,0], faces[:,1], faces[:,2]])
        J = np.hstack([faces[:,1], faces[:,2], faces[:,0]])

        data = np.ones(len(I))
        A = sparse.coo_matrix((data, (I, J)), shape=(n, n)).tocsr()
        
        A = A + A.T

        deg = np.array(A.sum(axis=1)).reshape(-1)

        L = sparse.eye(n) - A.multiply(1.0 / (deg+1e-12)[:,None])
        
        verts = verts - 0.1 * (L.dot(verts))

    except Exception:
        pass

    #Create modified mesh using original face shapes.
    new_mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=True)
    return new_mesh

def sample_points_in_cylinder(n_points, R):
    """Sample random points inside a cylinder with radius R and height 2."""

    rng = np.random.default_rng()

    #Distribute points uniformly over the circle area.
    u = rng.random(n_points)
    r = R * np.sqrt(u)

    #Get random polar angles in the xy-plane.
    theta = rng.random(n_points) * 2 * np.pi

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    #Sample a z value between -1 and 1
    z = rng.uniform(-1.0, 1.0, size=n_points)

    #Combine coordinates to 3D points.
    pts = np.column_stack([x, y, z])

    #Add a random point on each cylinder disk.
    ru = np.sqrt(rng.random(2)) * R
    th = rng.random(2) * 2 * np.pi

    xx = ru * np.cos(th)
    yy = ru * np.sin(th)
    zz = np.array([1,-1])

    edges = np.column_stack([xx, yy, zz])

    pts = np.vstack([pts, edges])
    return pts

def points_to_convex_stl(points, out_path):
    """Create and export the convex hull of a point cloud."""
    pcloud = trimesh.points.PointCloud(points)
    hull = pcloud.convex_hull  # returns a trimesh.Trimesh
    hull.export(out_path)
    return hull