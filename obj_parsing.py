from pathlib import Path
import trimesh
import numpy as np
import os
import shutil


root = Path("C:/Users/rober/Python Datengenerierung/damit-20260705T000301Z")

files = list(root.rglob("obj.txt"))

g=17

g1=1000*(g-1)
g2=1000*g

i=0
files=files[g1:g2]


def replace_vertices_in_objtxt(input_path, output_path, vertices_new):
    out_lines = []
    vertex_index = 0

    with open(input_path, "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) == 4 and parts[0] == "v":
                x, y, z = vertices_new[vertex_index]

                new_line = f"v {x:.6f} {y:.6f} {z:.6f}\n"
                out_lines.append(new_line)

                vertex_index += 1
            else:
                out_lines.append(line)

    if vertex_index != len(vertices_new):
        raise ValueError(
            f"Number of replaced vertices ({vertex_index}) "
            f"does not match vertices_new ({len(vertices_new)})"
        )

    with open(output_path, "w") as f:
        f.writelines(out_lines)

for f in files:
    vertices = []

    with open(f, "r") as fi:
        for line in fi:
            parts = line.split()

            if len(parts) == 4 and parts[0] == "v":
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                vertices.append([x,y,z])

    vertices = np.array(vertices)

    z_min = vertices[:,2].min()
    z_max = vertices[:,2].max()

    vertices[:,2]=vertices[:,2]-z_min-(np.abs(z_max-z_min)/2)

    z_min = vertices[:,2].min()
    z_max = vertices[:,2].max()

    vertices=vertices* (2/(z_max-z_min))

    z_min = vertices[:,2].min()
    z_max = vertices[:,2].max()

    R = np.max(np.sqrt(vertices[:,0]**2 + vertices[:,1]**2))

    replace_vertices_in_objtxt(f,f"C:/Users/rober/Python Datengenerierung/data/adjusted obj data/{g}{i}",vertices)

    dir=f"C:/Users/rober/Python Datengenerierung/dataset/batch{g}/sample{i}"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    mesh = trimesh.load(f"C:/Users/rober/Python Datengenerierung/data/adjusted obj data/{g}{i}", file_type="obj")
    mesh.export(f"C:/Users/rober/Python Datengenerierung/dataset/batch{g}/sample{i}/asteroid{i}radius{R}.stl")

    print(R)
    print(z_max)
    print(z_min)

    i+=1

# R = 1.0201783085568914

# cylinder = trimesh.creation.cylinder(
#     radius=R,
#     height=2,
#     sections=128
# )

# cylinder.export("bounding_cylinder.stl")