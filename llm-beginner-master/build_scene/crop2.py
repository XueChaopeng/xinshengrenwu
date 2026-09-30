# -*- coding: utf-8 -*-
from PIL import Image
import os
d = r"C:\Users\26378\Desktop\惩戒打击场景设计"
T = r"C:\Users\26378\AppData\Local\Temp"

def kmpx(kmx, kmy, W, H):
    return int(W * ((kmx + 120) / 1170)), int(H * (1 - (kmy - 40) / 520))

def crop(src, name, box_km, out):
    im = Image.open(os.path.join(d, src)).convert("RGB")
    W, H = im.size
    kx0, ky0, kx1, ky1 = box_km
    x0, y0 = kmpx(kx0, ky0, W, H); x1, y1 = kmpx(kx1, ky1, W, H)
    c = im.crop((x0, y0, x1, y1))
    c.save(os.path.join(T, out)); print(out, c.size)

# 图1：左半（岸基白框、潜艇、600km白框、图例、编队标签）
crop("图1_初始态势图.png", "a", (-120, 420, 520, 40), "zz_f1_left.png")
# 图1：右上（舰载高超覆盖、CSG警戒圈、350km）
crop("图1_初始态势图.png", "b", (430, 560, 1050, 240), "zz_f1_topright.png")
# 图2：左上（①③白框、苏-35S、A-50U、米格-31K）
crop("图2_打击实施图.png", "c", (-120, 560, 480, 340), "zz_f2_topleft.png")
# 图2：左下（②白框、潜艇、前哨-R、S-70、巡航弹覆盖、图例）
crop("图2_打击实施图.png", "d", (-120, 340, 640, 60), "zz_f2_botleft.png")
print("DONE")
