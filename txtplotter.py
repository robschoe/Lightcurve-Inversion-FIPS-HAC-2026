import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt("brightnessasteroid10007radius2.02317048971541.stl.csv", delimiter=",")
#data = np.loadtxt("Asteroid03_lightcurve_intensity.txt", delimiter=",", skiprows=1)

camsel=5

cam1 = data1[:,camsel]

cam1 = cam1 / np.mean(cam1)

# frames = data[:,0]
# cam1 = data[:,camsel]

plt.plot(range(len(cam1)), cam1)
plt.xlabel("Frame")
plt.ylabel("Brightness")
plt.title("Lightcurve Camera 1")

data2 = np.loadtxt("brightnessasteroid10005radius1.462970141641174.stl.csv", delimiter=",")

frames = data2[:,0]
cam2 = data2[:,camsel]

cam2 = cam2 / np.mean(cam2)

plt.plot(range(len(cam2)), cam2)
plt.xlabel("Frame")
plt.ylabel("Brightness")
plt.title("Lightcurve Camera 1")
plt.show()

# x=[]
# for h in range(29):
#     for k in range(30):
#         cam2 = data2[k::30,h]
#         cam2 = cam2 / np.mean(cam2)
#         cam1 = data1[:,h]
#         cam1 = cam1 / np.mean(cam1)
#         y=0
#         for j in range(len(cam1)):
#             y+=np.abs(cam2[j]-cam1[j])
#         x.append(y)
# print(np.argsort(x)%30)

# x=[]
# for j in range(11):
#     x.append(np.abs(cam2[j]-cam1[j]))
# print(np.max(x))

# for j in range(7):
#     y=4*j+1
#     x=0
#     for i in range(len(frames)):
#         x+=data[i,y]-data[i,y+1]
#     print(x)

# objects=[]
# objects.append("C:/Users/rober/Parametersuche blender/asteroid1.stl")
# objects.append("C:/Users/rober/Parametersuche blender/asteroid2.stl")
# objects.append("C:/Users/rober/Parametersuche blender/asteroid3.stl")
# asteroiddata=[]
# data2 = np.loadtxt("Asteroid01_lightcurve_intensity_blender.txt", delimiter=",")
# asteroiddata.append(data2)
# data3 = np.loadtxt("Asteroid02_lightcurve_intensity_blender.txt", delimiter=",")
# asteroiddata.append(data3)
# data4 = np.loadtxt("Asteroid03_lightcurve_intensity_blender.txt", delimiter=",")
# asteroiddata.append(data4)

# for l in range(7):
#     v=[]
#     for j in range(20):
#         y=0
#         i=0
#         for x in objects:
#             for h in range(2):
#                 data1 = np.loadtxt( f"{x}{j}.csv" , delimiter=",")
#                 data2=asteroiddata[i]
#                 cam1 = data1[:,h+l*4+3]
#                 if np.mean(cam1)!=0:
#                     cam1 = cam1 / np.mean(cam1)
#                 cam2 = data2[0::30,h+l*4+3]
#                 cam2 = cam2 / np.mean(cam2)
#                 for z in range(len(cam1)):
#                     y+=np.abs(cam2[z]-cam1[z])
#             i=(i+1)%3
#         v.append(y)
#     print(v)

#     plt.figure()
#     plt.plot(range(len(v)), v)
#     plt.xlabel("Frame / Time")
#     plt.ylabel("Brightness")
#     plt.title("Synthetic Lightcurve")
#     plt.show()
