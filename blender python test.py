import bpy
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import imageio
import random
import math
import alphashape
import trimesh
import torch
from scipy.signal import savgol_filter
from pathlib import Path
import os
import time

#Test Asteroid
obj_path = "C:/Users/rober/blendertest/asteroid3.stl"

#Path for Scans of Asteroid in Blender
output_path = "/Users/rober/blendertest/frames/"
#Frames
frames = 360
brightness = []

start = time.time()

def generate_asteroid(convex):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    points = []
    N = 200

    for i in range(N):
        r = np.sqrt(np.random.rand())
        theta = 2*np.pi*np.random.rand()

        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = np.random.uniform(-1,1)

        points.append([x,y,z])

    points = np.array(points)

    if convex==1:
        mesh = bpy.data.meshes.new("asteroid")
        obj = bpy.data.objects.new("asteroid", mesh)
        bpy.context.collection.objects.link(obj)

        bm = bmesh.new()

        verts = [bm.verts.new(p) for p in points]

        bmesh.ops.convex_hull(bm, input=verts)

        bm.to_mesh(mesh)
        bm.free()
        bpy.ops.wm.stl_export(filepath=obj_path)
    else:

        alpha = 1.5
        shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=shape.vertices,
                            faces=shape.faces)
        mesh.export("asteroid.stl")

# bpy.ops.mesh.primitive_cylinder_add(
#     radius=1,
#     depth=2,
#     location=(0,0,0)
# )
# obj = bpy.context.object

#generate_asteroid(1)

bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0, 0, 0, 1)  # black Background
bg.inputs[1].default_value = 0             # no brightness

# Making sure scene is empty
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


def create_camera(name, angle, mode):
    r=20
    anglexy=np.deg2rad(angle+180)
    anglez=np.deg2rad(90-angle_to_mode(angle)*mode)
    x=r*np.cos(anglexy)*np.sin(anglez)
    y=r*np.sin(anglexy)*np.sin(anglez)
    z=r*np.cos(anglez)
    bpy.ops.object.camera_add(location=(x,y,z))
    cam = bpy.context.object
    cam.name = name
    
    direction = - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()
    
    return cam

def angle_to_mode(angle):
    match angle:
        case 0:
            return 21 #24
        case 45 | 135:
            return 26
        case 90:
            return 26 #29
        case 225 | 270:
            return 24
        case 315:
            return 24 #25
        
# add Lightsource
bpy.ops.object.light_add(type='SUN', location=(-10,0,0))
sun = bpy.context.object
sun.data.energy = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

Rs=[]

for frame in range(frames):
    theta = - frame / frames * 2 * torch.pi
    theta = torch.tensor(theta, dtype=torch.float32, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)

    R = torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
        torch.stack([s,  c, torch.zeros_like(c)], dim=-1),
        torch.tensor([0.,0.,1.], device=device)
    ], dim=0)
    Rs.append(R)

cameras = []

for i in range(8):
    if i!=4:
        cam1 = create_camera(f"camera{i}mid", i*45,0)
        cam11 = create_camera(f"camera{i}mid", i*45,0)
        cam2 = create_camera(f"camera{i}top", i*45,1)
        cam3 = create_camera(f"camera{i}bot", i*45,-1)
        cameras.append(cam1)
        cameras.append(cam11)
        cameras.append(cam2)
        cameras.append(cam3)

view_dirs = []

for cam in cameras:
    v = cam.location
    v = v.normalized()
    view_dirs.append([v.x, v.y, v.z])

view_dirs_np = np.array(view_dirs)

sun_vec = sun.location
sun_vec = sun_vec.normalized()

sun_dir_np = np.array([sun_vec.x, sun_vec.y, sun_vec.z])

sun_dir = torch.tensor(sun_dir_np, dtype=torch.float32, device=device)
view_dirs = torch.tensor(view_dirs_np, dtype=torch.float32, device=device)

g=18
for lol in range(1):

    root = Path(f"C:/Users/rober/Python Datengenerierung/dataset/batch{g}")

    asteroids = list(root.rglob("asteroid*.stl"))
    for asteroid in asteroids:
        brightness=[]


        # import asteroid
        bpy.ops.wm.stl_import(filepath=f"{asteroid}")

        obj = bpy.context.selected_objects[0]

        obj.location = (0,0,0)

        mesh = obj.data

        centers = []
        normals = []
        areas = []

        for poly in mesh.polygons:
            centers.append([poly.center.x, poly.center.y, poly.center.z])
            normals.append([poly.normal.x, poly.normal.y, poly.normal.z])
            areas.append(poly.area)

        centers_np = np.array(centers)
        normals_np = np.array(normals)
        areas_np = np.array(areas)

        centers = torch.tensor(centers_np, dtype=torch.float32, device=device)
        normals = torch.tensor(normals_np, dtype=torch.float32, device=device)
        areas = torch.tensor(areas_np, dtype=torch.float32, device=device)

        for frame in range(frames):

            frame_values = []

            frame_values.append(frame)

            R=Rs[frame]

            rotated_normals = normals @ R.T
            rotated_centers = centers @ R.T

            W = 512*2
            H = 512*2

            illum = torch.clamp(rotated_normals @ sun_dir, min=0)

            for cam in cameras:
                cam_pos = torch.tensor([cam.location.x, cam.location.y, cam.location.z], device=device)

                view = (cam_pos)
                view = view / torch.linalg.norm(view)

                up = torch.tensor([0.,0.,1.], device=device)

                x = torch.cross(view, up)
                x = x / torch.linalg.norm(x)

                y = torch.cross(x, view)

                rel = rotated_centers - cam_pos


                cx = rel @ x
                cy = rel @ y
                cz = rel @ view

                px = ((cx - cx.min())/(cx.max()-cx.min()) * (W-1)).long()
                py = ((cy - cy.min())/(cy.max()-cy.min()) * (H-1)).long()

                px = torch.clamp(px,0,W-1)
                py = torch.clamp(py,0,H-1)

                # zbuf = torch.full((H,W), float('inf'), device=device)
                # face_id = torch.full((H,W), -1, dtype=torch.long, device=device)

                # for i in range(len(px)):
                #     if cz[i] < zbuf[py[i],px[i]]:
                #         zbuf[py[i],px[i]] = cz[i]
                #         face_id[py[i],px[i]] = i
                # visible_faces = torch.unique(face_id[face_id >= 0])

                flat_index = py * W + px
                zbuf_flat = torch.full((H*W,), float('inf'), device=device)
                face_id_flat = torch.full((H*W,), -1, dtype=torch.long, device=device)
                zbuf_flat = zbuf_flat.scatter_reduce(
                    0,
                    flat_index,
                    cz,
                    reduce="amin",
                    include_self=True
                )
                visible_mask = cz == zbuf_flat[flat_index]
                visible_faces = torch.where(visible_mask)[0]

                vis = torch.clamp(rotated_normals @ view, min=0)

                brightness_val = torch.sum(
                    areas[visible_faces] *
                    illum[visible_faces] *
                    vis[visible_faces]
                )
                frame_values.append(brightness_val.item())
            brightness.append(frame_values)
        #brightness = savgol_filter(brightness, 11, 3, axis=0)
        
        path=os.path.dirname(asteroid)
        asteroid=os.path.basename(asteroid)

        print(asteroid)
        print(path)

        np.savetxt(f"{path}/brightness{asteroid}.csv", brightness, delimiter=",")
    g+=1
    

end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", length, "seconds!")


# for i,b in enumerate(brightness):
#     print(i, len(b))


# frames = brightness[:,0]      # erste Spalte
# values = brightness[:,1:]     # Helligkeiten

# mean = values.mean(axis=0)    # Durchschnitt pro Kamera

# normalized = values / mean

# brightness_norm = np.column_stack((frames, normalized))


#x=1
# for f in files:
#     img = Image.open(f).convert("L")
#     arr = np.array(img)
#     b = arr.mean()
#     brightness.append(b)

#     plt.figure(figsize=(4,3))
#     plt.plot(brightness, color="white")
#     plt.xlim(x, frames+x)
#     #plt.ylim(0,255)
#     plt.gca().set_facecolor("black")
#     plt.xlabel("frame")
#     plt.ylabel("brightness")

#     plot_path = f"/Users/rober/blendertest/plots/plot_{x}.png"
#     plt.savefig(plot_path, facecolor="black")
#     plt.close()

#     asteroid = Image.open(f)
#     plot = Image.open(plot_path)

#     w = asteroid.width + plot.width
#     h = max(asteroid.height, plot.height)

#     combined = Image.new("RGB",(w,h))
#     combined.paste(asteroid,(0,0))
#     combined.paste(plot,(asteroid.width,0))

#     combined_path = f"/Users/rober/blendertest/plots/plot_{x}.png"
#     combined.save(combined_path)

#     gif_frames.append(imageio.imread(combined_path))
#     x+=1

#imageio.mimsave("/Users/rober/blendertest/asteroid_lightcurve.gif", gif_frames, fps=10)