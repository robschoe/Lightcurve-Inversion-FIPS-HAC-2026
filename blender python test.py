import bpy
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import imageio
import random
import math
import bmesh
import alphashape
import trimesh

#Test Asteroid
obj_path = "C:/Users/rober/blendertest/asteroid.stl"

#Path for Scans of Asteroid in Blender
output_path = "/Users/rober/blendertest/frames/"
#Frames
frames = 120
gif_frames=[]
brightness = []

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

generate_asteroid(1)

bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0, 0, 0, 1)  # black Background
bg.inputs[1].default_value = 0             # no brightness

# Making sure scene is empty
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# import asteroid
bpy.ops.wm.stl_import(filepath=obj_path)

obj = bpy.context.selected_objects[0]

obj.location = (0,0,0)

def create_camera(name, angle, mode):
    anglexy=((angle/360)+1/2)*2*np.pi
    anglez=(np.pi/4)*mode*(-1)+np.pi/2
    x=5*np.cos(anglexy)*np.sin(anglez)
    y=5*np.sin(anglexy)*np.sin(anglez)
    z=5*np.cos(anglez)
    bpy.ops.object.camera_add(location=(x,y,z))
    cam = bpy.context.object
    cam.name = name
    
    direction = obj.location - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()
    
    return cam

cameras = []

# for i in range(8):
#     cam1 = create_camera(f"camera{i}mid", i*45,0)
#     cam2 = create_camera(f"camera{i}top", i*45,1)
#     cam3 = create_camera(f"camera{i}bot", i*45,-1)
#     cameras.append(cam1)
#     cameras.append(cam2)
#     cameras.append(cam3)
#     brightness1=[]
#     brightness2=[]
#     brightness3=[]
#     brightness.append(brightness1)
#     brightness.append(brightness2)
#     brightness.append(brightness3)

cam1 = create_camera(f"camera", 45,1)
cameras.append(cam1)


# add Lightsource
bpy.ops.object.light_add(type='SUN', location=(-10,0,0))
sun = bpy.context.object
sun.data.energy = 5
direction = obj.location - sun.location
sun.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()

# Render Settings
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 512
scene.render.resolution_y = 512

# render Animation
for frame in range(frames):
    i=0
    for cam in cameras:

        scene.camera = cam
        angle = 2 * math.pi * frame / frames
        obj.rotation_euler[2] = angle

        scene.render.filepath = f"{output_path}{cam.name}{frame:03d}.png"
        bpy.ops.render.render(write_still=True)
        img = Image.open(f"{output_path}{cam.name}{frame:03d}.png").convert("L")
        img_array = np.array(img)
        print(img_array.mean())
        # calculate mean brightness
        brightness.append(img_array.mean())
        i+=1

files = sorted(glob.glob("C:/Users/rober/blendertest/frames/*.png"))


for f in files:
    img = Image.open(f).convert("L")
    img_array = np.array(img)
    print(img_array.mean())
    # calculate mean brightness
    brightness.append(img_array.mean())
   
time = np.arange(len(brightness))

#Plot of lightcurve
plt.figure()
plt.plot(time, brightness)
plt.xlabel("Frame / Time")
plt.ylabel("Brightness")
plt.title("Synthetic Lightcurve")
plt.show()

#brightness = np.array(brightness)

#np.savetxt("brightness.csv", brightness, delimiter=",")

x=1
for f in files:
    img = Image.open(f).convert("L")
    arr = np.array(img)
    b = arr.mean()
    brightness.append(b)

    plt.figure(figsize=(4,3))
    plt.plot(brightness, color="white")
    plt.xlim(x, frames+x)
    #plt.ylim(0,255)
    plt.gca().set_facecolor("black")
    plt.xlabel("frame")
    plt.ylabel("brightness")

    plot_path = f"/Users/rober/blendertest/plots/plot_{x}.png"
    plt.savefig(plot_path, facecolor="black")
    plt.close()

    asteroid = Image.open(f)
    plot = Image.open(plot_path)

    w = asteroid.width + plot.width
    h = max(asteroid.height, plot.height)

    combined = Image.new("RGB",(w,h))
    combined.paste(asteroid,(0,0))
    combined.paste(plot,(asteroid.width,0))

    combined_path = f"/Users/rober/blendertest/plots/plot_{x}.png"
    combined.save(combined_path)

    gif_frames.append(imageio.imread(combined_path))
    x+=1

imageio.mimsave("/Users/rober/blendertest/asteroid_lightcurve.gif", gif_frames, fps=10)