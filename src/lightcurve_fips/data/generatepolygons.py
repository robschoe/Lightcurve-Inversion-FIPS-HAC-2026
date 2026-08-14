import numpy as np
import trimesh
from pathlib import Path
from numpy import random
from opensimplex import OpenSimplex
import os

def add_noise_and_craters(mesh, noise_amp=0.02, noise_scale=2.0,
                          n_craters=30, crater_max_radius=0.25, crater_max_depth=0.2):
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals.copy()

    gen = OpenSimplex(random.randint(0,2**31-1))

    r = np.linalg.norm(verts, axis=1)
    dirs = verts / (r[:, None] + 1e-12)

    def noise_val(p, scale):
        return gen.noise3(p[0]*scale, p[1]*scale, p[2]*scale)

    noises = np.array([noise_val(d, noise_scale) for d in dirs], dtype=np.float32)
    verts += (normals * (noise_amp * noises)[:, None])

    crater_centers = []
    crater_radii = []
    crater_depths = []
    for i in range(n_craters):
        theta = random.random() * 2*np.pi                           #generate random angle 
        u = random.uniform(-1,1)                                    #generate random z coordinate
        dirc = np.array([np.cos(theta)*np.sqrt(1-u*u),              #random normalized direction to center of crater
                         np.sin(theta)*np.sqrt(1-u*u), u],
                         dtype=np.float32)      
        crater_centers.append(dirc)                                     #save crater center
        crater_radii.append(random.uniform(0.03, crater_max_radius))    #random crater radius
        crater_depths.append(random.uniform(0.02, crater_max_depth))    #random crater depth
    crater_centers = np.array(crater_centers)

    dots = np.dot(dirs, crater_centers.T)   #cos of angle between Vertex and crater center
    dots = np.clip(dots, -1.0, 1.0)         #arccos only defined on (-1,1)
    angles = np.arccos(dots)                #calculate angular distance

    for j in range(len(crater_centers)):
        ang = angles[:, j]
        mask = ang < crater_radii[j]        #get vertices inside of crater
        if not np.any(mask):
            continue
        frac = 1.0 - (ang[mask] / crater_radii[j])  #closeness to center, 1 in center, 0 on edge
        falloff = 0.5 * (1 - np.cos(frac * np.pi))  #smoothing of crater
        depth_vals = crater_depths[j] * falloff     #calculate relative depth of vertices
        verts[mask] -= (dirs[mask] * (depth_vals[:, None] * r[mask, None]))     #push vertices inside

    # optional smoothing to remove artifacts (Laplacian smoothing)
    # here a single laplacian step:
    try:
        from scipy import sparse
        # build adjacency from faces
        faces = mesh.faces
        n = len(verts)
        I = np.hstack([faces[:,0], faces[:,1], faces[:,2]])
        J = np.hstack([faces[:,1], faces[:,2], faces[:,0]])
        data = np.ones(len(I))
        A = sparse.coo_matrix((data, (I, J)), shape=(n, n)).tocsr()
        # symmetric
        A = A + A.T
        deg = np.array(A.sum(axis=1)).reshape(-1)
        L = sparse.eye(n) - A.multiply(1.0 / (deg+1e-12)[:,None])
        # one smoothing step
        verts = verts - 0.1 * (L.dot(verts))
    except Exception:
        pass

    new_mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=True)
    return new_mesh

def sample_points_in_cylinder(n_points, R):
    rng = np.random.default_rng()

    u = rng.random(n_points)
    r = R * np.sqrt(u)                          #random distance to (0,0,0)
    theta = rng.random(n_points) * 2 * np.pi    #random angle in cylinder

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.uniform(-1.0, 1.0, size=n_points)   #z between -1 and 1

    pts = np.column_stack([x, y, z])            #combine all generated coordinates

    ru = np.sqrt(rng.random(2)) * R             #create Points on z=1 and z=-1
    th = rng.random(2) * 2 * np.pi
    xx = ru * np.cos(th)
    yy = ru * np.sin(th)
    zz = np.array([1,-1])
    edges = np.column_stack([xx, yy, zz])

    pts = np.vstack([pts, edges])
    return pts

def points_to_convex_stl(points, out_path):
    pcloud = trimesh.points.PointCloud(points)
    hull = pcloud.convex_hull  # returns a trimesh.Trimesh
    hull.export(out_path)
    return hull