# -*- coding: utf-8 -*-
import os
from PIL import Image

d = r"C:\Users\26378\Desktop\惩戒打击场景设计"
T = r"C:\Users\26378\AppData\Local\Temp"

def km_to_px(kmx, kmy, W, H):
    return int(W * ((kmx + 120) / 1170)), int(H * (1 - (kmy - 40) / 520))

# 图2：打击标签区
im = Image.open(os.path.join(d, "图2_打击实施图.png")).convert("RGB")
W, H = im.size
print("fig2 W,H", W, H)
for name, (kx0, ky0, kx1, ky1) in {
    "fig2_strike1": (150, 470, 500, 330),
    "fig2_strike2": (250, 300, 560, 200),
    "fig2_sub":     (60, 220, 320, 90),
}.items():
    x0, y0 = km_to_px(kx0, ky0, W, H); x1, y1 = km_to_px(kx1, ky1, W, H)
    c = im.crop((x0, y0, x1, y1))
    c.save(os.path.join(T, "zz_" + name + ".png")); print(name, c.size)

# 图1：前出哨舰 + 950km文字
im1 = Image.open(os.path.join(d, "图1_初始态势图.png")).convert("RGB")
W1, H1 = im1.size
print("fig1 W,H", W1, H1)
x0, y0 = km_to_px(400, 340, W1, H1); x1, y1 = km_to_px(760, 180, W1, H1)
c1 = im1.crop((x0, y0, x1, y1)); c1.save(os.path.join(T, "zz_fig1_sentinel.png")); print("fig1_sentinel", c1.size)
print("OK")
