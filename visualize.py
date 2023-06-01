import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("guided_filter_psnr.csv")

# rの値のリストを取得する
# r_values = df["r"].unique()

# rの値ごとに3Dグラフを作成する
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# rに対応するデータのみをフィルタリングする


# グリッドを作成
x = np.linspace(df["r"].min(), df["r"].max(), num=100)
y = np.linspace(df["eps"].min(), df["eps"].max(), num=100)
X, Y = np.meshgrid(x, y)

# PSNRを補間
Z = griddata((df["r"], df["eps"]), df["psnr"], (X, Y), method='cubic')

# 3D曲面を作成する
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

ax.set_xlabel('r')
ax.set_ylabel('eps')
ax.set_zlabel('PSNR')
fig.colorbar(surf, ax=ax, label='PSNR')

plt.show()
