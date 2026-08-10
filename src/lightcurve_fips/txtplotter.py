import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#data1 = np.loadtxt("brightnessasteroid5radius2.0475867806576935.stl.csv", delimiter=",")
data = np.loadtxt("brightnessasteroid3.stl.csv", delimiter=",", skiprows=1)

camsel=20

cam1 = data[:,camsel]

cam1 = cam1 / np.mean(cam1)

# frames = data[:,0]
# cam1 = data[:,camsel]

plt.plot(range(len(cam1)), cam1)
plt.xlabel("Frame")
plt.ylabel("Brightness")
plt.title(f"Lightcurve Camera {camsel}")

data2 = np.loadtxt("Asteroid03_lightcurve_intensity_blender.txt", delimiter=",")

frames = data2[:,0]
cam2 = data2[:,camsel]

cam2 = cam2 / np.mean(cam2)

plt.plot(range(len(cam2)), cam2)
plt.xlabel("Frame")
plt.ylabel("Brightness")
plt.title(f"Lightcurve Camera {camsel}")
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

def load_brightness_file(path):
    path = Path(path)

    # Liest CSV/TXT mit Komma, Semikolon, Tab oder Leerzeichen als Trenner
    df = pd.read_csv(
        path,
        sep=r"[,\s;]+",
        engine="python",
        header=None,
        comment="#"
    )

    # Falls durch Trennzeichen leere Spalten entstehen
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != 29:
        raise ValueError(f"{path} hat {df.shape[1]} Spalten, erwartet werden 29.")

    columns = ["frame"] + [f"cam_{i:02d}" for i in range(1, 29)]
    df.columns = columns

    return df


def compare_curves(y1, y2):
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    # NaN/Inf entfernen
    mask = np.isfinite(y1) & np.isfinite(y2)
    y1 = y1[mask]
    y2 = y2[mask]

    diff = y1 - y2

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    # Korrelation nur möglich, wenn beide Kurven nicht konstant sind
    if np.std(y1) > 0 and np.std(y2) > 0:
        corr = np.corrcoef(y1, y2)[0, 1]
    else:
        corr = np.nan

    # Normalisierter RMSE bezogen auf Wertebereich von Kurve 1
    value_range = np.max(y1) - np.min(y1)
    if value_range > 0:
        nrmse = rmse / value_range
    else:
        nrmse = np.nan

    return rmse, mae, corr, nrmse

root=Path.cwd()
file_a = root/r"brightnessasteroid3.stl.csv"
file_b = root/r"Asteroid03_lightcurve_intensity_blender.txt"

df_a = load_brightness_file(file_a)
df_b = load_brightness_file(file_b)

# Nach Frame-Nummer zusammenführen
df = pd.merge(df_a, df_b, on="frame", suffixes=("_a", "_b"))

results = []

for cam in [f"cam_{i:02d}" for i in range(1, 29)]:
    y_a = df[f"{cam}_a"].values
    y_b = df[f"{cam}_b"].values

    rmse, mae, corr, nrmse = compare_curves(y_a, y_b)

    results.append({
        "camera": cam,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr,
        "NRMSE": nrmse
    })

results_df = pd.DataFrame(results)

print(results_df)

# Durchschnitt über alle Kameras
print("\nMittelwerte:")
print(results_df[["RMSE", "MAE", "Correlation", "NRMSE"]].mean())

# Optional speichern
results_df.to_csv("vergleich_ergebnisse.csv", index=False)
