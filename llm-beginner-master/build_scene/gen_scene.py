# -*- coding: utf-8 -*-
"""
场景设计图生成脚本（优化版）：
  图1 初始态势图   —— 红蓝距离、我方编队间距、技战参数（主图+我方编队间距放大条带）
  图2 打击实施图   —— 无人机+巡航导弹+弹道导弹多域协同，实现链路切割
  图3 行动时间甘特图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, lines, patheffects as pe
import numpy as np
import os

font_manager.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 输出到脚本自身所在文件夹（自包含，可随处运行）
OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = "#1f5aa6"; BLUE_L = "#6fa8dc"
RED = "#c0392b"; RED_L = "#e74c3c"
GREEN = "#1e8449"; ORANGE = "#d68910"
GREY = "#5d6d7e"; OCEAN = "#e3f0f9"


def draw_ship(ax, x, y, color=BLUE, w=26, h=7, label="", fs=7, dy=0, z=6):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    hull = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.2,rounding_size=1.6",
                          fc=color, ec="white", lw=0.7, zorder=z)
    ax.add_patch(hull)
    sup = Rectangle((x - w*0.16, y - h*0.7), w*0.32, h*1.4, fc="white", ec="white", zorder=z+2)
    ax.add_patch(sup)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color="white",
                zorder=z+3, fontweight="bold")


def draw_aircraft(ax, x, y, color=BLUE, label="", fs=7):
    ax.plot(x, y, marker="^", markersize=6, color=color, zorder=7)
    ax.plot([x-5, x+5], [y, y], color=color, lw=1.0, zorder=6)
    ax.plot([x, x], [y-4, y+4], color=color, lw=1.0, zorder=6)
    ax.text(x, y+7, label, ha="center", va="bottom", fontsize=fs, color=color, zorder=10,
            fontweight="bold", path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def dashed_circle(ax, x, y, r, color, ls=(0, (6, 4)), lw=1.1, alpha=0.6, label="", lbl_angle=30, dx=0, dy=0, fs=8):
    ax.add_patch(plt.Circle((x, y), r, fill=False, ec=color, ls=ls, lw=lw, alpha=alpha, zorder=2))
    if label:
        ang = np.deg2rad(lbl_angle)
        lx, ly = x + r*np.cos(ang) + dx, y + r*np.sin(ang) + dy
        ax.text(lx, ly, label, color=color, fontsize=fs, ha="center", va="center", zorder=9,
                fontweight="bold", path_effects=[pe.withStroke(linewidth=3, foreground="white")])


# ============================================================ 图1 初始态势图
def fig1():
    fig = plt.figure(figsize=(15, 10.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.35, 1.0], hspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(OCEAN)
    ax.set_xlim(-120, 1050); ax.set_ylim(40, 560)
    ax.set_aspect("auto")
    for gx in range(0, 1100, 100):
        ax.axvline(gx, color="#9fb8cc", lw=0.7, ls=":", zorder=1)
    for gy in range(0, 600, 100):
        ax.axhline(gy, color="#9fb8cc", lw=0.7, ls=":", zorder=1)
    for gx in range(0, 1100, 200):
        ax.text(gx, 46, f"{gx}km", color="#7d8b96", fontsize=7.5, ha="center", zorder=3)
    ax.set_xlabel("相对海区距离 (km) —— 东西向", fontsize=10)
    ax.set_ylabel("南北向 (km)", fontsize=10)
    ax.set_title("图1   初始态势图 —— 敌航母打击群前出哨舰（伯克2A）滋事，距我方(俄军)约600km",
                 fontsize=15, fontweight="bold", pad=10, color="#1a1a2e")

    bx, by = 0, 300
    sx, sy = 600, 300
    cx, cy = 950, 300
    Y = by

    # ---------- 我方编队 ----------
    fleet = [("22350(指挥)", 0, 300), ("光荣级", 26, 318), ("无畏级", -25, 282),
             ("20380-1", 48, 344), ("20380-2", -45, 340), ("补给舰", 10, 258)]
    for nm, fx, fy in fleet:
        draw_ship(ax, fx, fy, color=BLUE, w=24, h=7, label=nm.split("(")[0], fs=6.5)
    ax.add_patch(patches.Ellipse((0, 300), 150, 150, fill=False, ec=BLUE, lw=1.5, ls="--", zorder=4))
    ax.text(0, 400, "俄方海上编队(蓝军)\n(指挥舰22350 / 光荣级 / 无畏级×2 / 20380×2 / 补给)\n舰间距8~15km，正面约40km(见下方放大)",
            ha="center", va="bottom", fontsize=9, color=BLUE, zorder=10, fontweight="bold")
    draw_aircraft(ax, 70, 450, color=BLUE, label="A-50U预警机", fs=7)
    draw_aircraft(ax, 115, 490, color=BLUE, label="伊尔-20电子战飞机", fs=7)
    draw_aircraft(ax, 150, 535, color=BLUE, label="米格-31K(匕首·远程)", fs=7)
    draw_aircraft(ax, 300, 370, color=BLUE, label="苏-35S/苏-34(中距·反辐射)", fs=7)
    draw_aircraft(ax, 340, 200, color=BLUE, label="前哨-R无人机(侦察)", fs=7)
    draw_aircraft(ax, 385, 250, color=BLUE, label="S-70猎人无人机(察打)", fs=7)
    # 潜艇（隐蔽轴线）
    ax.add_patch(patches.FancyBboxPatch((150-22, 140-5), 44, 10, boxstyle="round,pad=0.6,rounding_size=4",
                                        fc="#34495e", ec="white", lw=0.7, zorder=6))
    ax.plot([172, 172], [130, 150], color="#34495e", lw=1.4, zorder=6)
    ax.text(172, 128, "潜艇(隐蔽轴线: 3M54口径/P-800缟玛瑙)", ha="center", va="top", fontsize=7.5,
            color="#34495e", zorder=10, fontweight="bold")
    ax.text(30, 175, "岸基高超/反舰导弹\n(3M22锆石/K-300P棱堡-缟玛瑙)", ha="center", va="center", fontsize=8,
            color=BLUE, zorder=10, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # ---------- 敌前出哨舰 ----------
    draw_ship(ax, sx, sy, color=RED, w=40, h=11, label="伯克2A", fs=9)
    ax.text(sx, sy-26, "前出哨舰（滋事）\n阿利·伯克2A型驱逐舰 DDG-72\n★ 打击目标", ha="center", va="top",
            fontsize=9.5, color=RED, zorder=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec=RED, lw=1.4))

    # ---------- 敌航母打击群 ----------
    csg = [("CVN-1", 950, 300), ("CG-52", 982, 322), ("DDG-53", 924, 280),
           ("AOE", 1000, 262), ("DDG-XX", 908, 338)]
    for nm, gx, gy in csg:
        draw_ship(ax, gx, gy, color=RED, w=26, h=7, label=nm, fs=6.5)
    ax.add_patch(patches.Ellipse((950, 300), 175, 165, fill=False, ec=RED, lw=1.6, ls="--", zorder=4))
    ax.text(950, 205, "敌航母打击群(旗舰CVN-1, 距我方约950km·航母殿后)\n舰载E-2D预警机/F/A-18/EA-18G\n天基侦察+远程通信+电子信息支援",
            ha="center", va="top", fontsize=8.6, color=RED, zorder=10, fontweight="bold")

    # ---------- 距离标注 ----------
    ax.annotate("", xy=(560, Y), xytext=(60, Y),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=2, shrinkA=0, shrinkB=0))
    ax.text(310, Y+20, "约600km（进入侦察/监视范围）", ha="center", fontsize=10.5, color="#2c3e50",
            zorder=10, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.annotate("", xy=(905, Y), xytext=(645, Y),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=2, shrinkA=0, shrinkB=0))
    ax.text(775, Y+20, "约350km", ha="center", fontsize=10, color="#2c3e50", zorder=10, fontweight="bold")

    # ---------- 火力/探测覆盖 ----------
    dashed_circle(ax, bx, by, 600, "#d68910", ls=(0, (6, 4)), lw=1.4, alpha=0.75,
                  label="舰载高超覆盖(3M22锆石 ~600km)", lbl_angle=20, dy=-4, fs=8.5)
    dashed_circle(ax, bx, by, 500, BLUE, ls=(0, (4, 3)), lw=1.1, alpha=0.6,
                  label="巡航反舰覆盖(口径/缟玛瑙 岸/潜/舰 ~500km)", lbl_angle=-30, dx=-6, dy=6, fs=8)
    dashed_circle(ax, cx, cy, 350, RED, ls=(0, (5, 3)), lw=1.2, alpha=0.6,
                  label="敌CSG防空警戒圈(~350km)", lbl_angle=152, dx=4, dy=0, fs=8)
    dashed_circle(ax, sx, sy, 200, RED_L, ls=(0, (2, 2)), lw=1.0, alpha=0.5,
                  label="哨舰宙斯盾防空监视(~200km)", lbl_angle=-105, dx=4, dy=-4, fs=7.5)

    handles = [
        patches.Patch(color=BLUE, label="我方(蓝军)"), patches.Patch(color=RED, label="敌方(红军)"),
        lines.Line2D([0], [0], color=GREY, lw=2, ls="--", label="距离量"),
        lines.Line2D([0], [0], color=BLUE, lw=1.4, ls=(0, (6, 4)), label="我方火力/探测覆盖"),
        lines.Line2D([0], [0], color=RED, lw=1.2, ls=(0, (5, 3)), label="敌方防空警戒"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95, ncol=2)

    # ---------- 下方：我方编队间距放大条带 ----------
    axi = fig.add_subplot(gs[1, 0])
    axi.set_facecolor("#f5fafd")
    axi.set_xlim(-58, 58); axi.set_ylim(-16, 16)
    axi.set_xticks(range(-50, 60, 25)); axi.set_yticks([])
    axi.tick_params(labelsize=7)
    axi.grid(color="#b9d1e3", lw=0.6, ls=":")
    axi.set_title("俄方编队间距放大示意 (km)：舰间距约8~15km，编队正面约40km，防空/反舰综合队形",
                  fontsize=10, fontweight="bold", pad=6)
    zoom_n = [("22350(指挥)", -48), ("光荣级", -26), ("无畏级", -6), ("20380-1", 16), ("20380-2", 34), ("补给舰", 50)]
    for nm, zx in zoom_n:
        draw_ship(axi, zx, 0, color=BLUE, w=9, h=3.6, label=nm.split("(")[0], fs=6)
        axi.text(zx, -7.5, nm, ha="center", va="top", fontsize=6, color=BLUE)
    for a, b, t in [(-48, -26, "~22km"), (-26, -6, "~20km"), (-6, 16, "~22km")]:
        axi.annotate("", xy=(b+0.2, 6.5), xytext=(a+0.4, 6.5),
                     arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
        axi.text((a+b)/2, 9.2, t, fontsize=6.5, color=GREY, ha="center")
    axi.text(0, -13.5, "编队正面约40km；各舰遂行区域防空/对海打击/反潜，数据链协同交战",
             ha="center", fontsize=6.6, color="#2c3e50")

    fig.tight_layout()
    fig.savefig(OUT + r"\图1_初始态势图.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] 图1")


# ============================================================ 图2 打击实施图
def fig2():
    fig, ax = plt.subplots(figsize=(15, 9.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eaf3fa")
    ax.set_xlim(-120, 1050); ax.set_ylim(40, 560)
    ax.set_aspect("auto")
    for gx in range(0, 1100, 100):
        ax.axvline(gx, color="#9fb8cc", lw=0.7, ls=":", zorder=1)
    for gy in range(0, 600, 100):
        ax.axhline(gy, color="#9fb8cc", lw=0.7, ls=":", zorder=1)
    ax.set_xlabel("相对海区距离 (km)", fontsize=10)
    ax.set_ylabel("南北向 (km)", fontsize=10)
    ax.set_title("图2   打击实施阶段 —— 高超/弹道＋巡航弹＋反辐射/电子压制＋无人机 多域协同，实现链路切割",
                 fontsize=15, fontweight="bold", pad=10, color="#1a1a2e")

    bx, by = 0, 300
    sx, sy = 600, 300
    cx, cy = 950, 300

    # 我方
    draw_aircraft(ax, 60, 440, color=BLUE, label="A-50U(引导)", fs=7)
    draw_aircraft(ax, 90, 480, color=BLUE, label="伊尔-20EW(压制卫通/数据链)", fs=7)
    draw_ship(ax, 0, 300, color=BLUE, w=32, h=9, label="22350", fs=7)
    draw_ship(ax, 26, 318, color=BLUE, w=24, h=7, label="光荣级", fs=6)
    draw_ship(ax, -24, 282, color=BLUE, w=24, h=7, label="无畏级", fs=6)
    ax.text(0, 365, "俄方编队(22350/光荣级/无畏级/20380/补给)", ha="center", fontsize=8.6, color=BLUE, zorder=10, fontweight="bold")
    draw_aircraft(ax, 200, 520, color=BLUE, label="米格-31K(匕首·远程)", fs=7)
    draw_aircraft(ax, 285, 405, color=BLUE, label="苏-35S(反辐射/护航)", fs=7)
    draw_aircraft(ax, 480, 170, color=BLUE, label="前哨-R(抵近侦察)", fs=7)
    draw_aircraft(ax, 520, 230, color=BLUE, label="S-70猎人(察打/中继)", fs=7)
    # 潜艇（隐蔽轴线）
    ax.add_patch(patches.FancyBboxPatch((150-22, 150-5), 44, 10, boxstyle="round,pad=0.6,rounding_size=4",
                                        fc="#34495e", ec="white", lw=0.7, zorder=6))
    ax.plot([172, 172], [140, 160], color="#34495e", lw=1.4, zorder=6)
    ax.text(172, 138, "潜艇(隐蔽轴线) 3M54口径/缟玛瑙", ha="center", va="top", fontsize=7,
            color="#34495e", zorder=10, fontweight="bold")

    # 敌
    draw_ship(ax, sx, sy, color=RED, w=40, h=11, label="伯克2A", fs=9)
    ax.text(sx, sy-24, "前出哨舰 DDG-72（回撤中）\n★ 敌方打击目标", ha="center", va="top", fontsize=9,
            color=RED, zorder=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec=RED, lw=1.4))
    draw_ship(ax, 950, 300, color=RED, w=32, h=9, label="CVN-1", fs=7)
    draw_ship(ax, 985, 322, color=RED, w=24, h=7, label="CG-52", fs=6)
    draw_ship(ax, 920, 280, color=RED, w=24, h=7, label="DDG-53", fs=6)
    ax.text(950, 210, "敌航母打击群(殿后, 不进入对抗)", ha="center", fontsize=8.6, color=RED, zorder=10, fontweight="bold")

    # 打击弹道（带箭头直线），标签靠近发射点并向外偏移，避免重叠
    def strike(x0, y0, x1, y1, color, ls="-", lw=2.2, label="", lx=None, ly=None, fs=8):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0), zorder=11)
        if label:
            if lx is None:
                lx, ly = (x0+x1)/2, (y0+y1)/2
            ax.text(lx, ly, label, fontsize=fs, color=color, ha="left", va="center", zorder=12,
                    fontweight="bold", path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # ① 高超/弹道反舰（防区外，锆石+匕首）
    strike(0, 300, 600, 310, "#d68910", lw=2.4, label="① 高超/弹道反舰\n(锆石/匕首)→AEGIS/SPY-1/指挥/卫通", lx=170, ly=380)
    strike(200, 520, 600, 320, "#d68910", ls="--", lw=1.8)   # 匕首空射
    # ② 巡航弹（岸/潜/舰，口径/缟玛瑙）
    strike(172, 150, 600, 295, BLUE_L, ls="--", lw=1.8, label="② 巡航弹(口径/缟玛瑙 岸潜舰, 接力)\n→上层建筑/通信/动力", lx=380, ly=200)
    # ③ 反辐射+电子压制（软硬结合）
    strike(285, 405, 600, 305, "#e67e22", ls=":", lw=2.0, label="③ 反辐射+电子压制(苏-35S·Kh-31P/伊尔-20EW)", lx=200, ly=460)
    # ④ 无人机抵近/BDA
    strike(520, 230, 600, 290, GREEN, ls="-.", lw=1.6, label="④ 无人机抵近/BDA", lx=620, ly=120)

    # 链路切割目标区
    ax.add_patch(patches.Rectangle((660, 390), 300, 110, fill=True, fc="#fffbe6",
                                   ec=ORANGE, lw=1.6, zorder=9, alpha=0.85))
    ax.text(810, 480, "链路切割目标区", ha="center", fontsize=10.5, color=ORANGE, zorder=12, fontweight="bold")
    ax.text(810, 445, "瘫痪作战系统(AEGIS/SPY-1/指挥室)\n切断与CSG的CEC/Link-16链路\n哨舰失去协同交战与前方警戒能力",
            ha="center", fontsize=8.6, color="#b9770e", zorder=12)
    ax.annotate("", xy=(600, 320), xytext=(665, 420),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6, ls="--"), zorder=12)

    dashed_circle(ax, 0, 300, 600, "#d68910", ls=(0, (6, 4)), lw=1.4, alpha=0.7,
                  label="高超覆盖(锆石 ~600km)", lbl_angle=20, dy=-4)
    dashed_circle(ax, 0, 300, 500, BLUE, ls=(0, (4, 3)), lw=1.1, alpha=0.55,
                  label="巡航弹覆盖(口径/缟玛瑙 ~500km)", lbl_angle=12, dx=-4, dy=-6)

    handles = [
        patches.Patch(color=BLUE, label="我方"), patches.Patch(color=RED, label="敌方"),
        lines.Line2D([0], [0], color="#d68910", lw=2.4, label="① 高超/弹道反舰"),
        lines.Line2D([0], [0], color=BLUE_L, lw=1.8, ls="--", label="② 巡航弹(岸/潜/舰)"),
        lines.Line2D([0], [0], color="#e67e22", lw=2.0, ls=":", label="③ 反辐射+电子压制"),
        lines.Line2D([0], [0], color=GREEN, lw=1.6, ls="-.", label="④ 无人机BDA"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.95, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT + r"\图2_打击实施图.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] 图2")


def fig3():
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    t_start = 0; t_end = 110
    ax.set_xlim(t_start, t_end); ax.set_ylim(0, 12)

    phases = [
        ("侦察监视", 0, 30, "#d6eaf8"), ("研判决策", 25, 55, "#d5f5e3"),
        ("任务筹划", 45, 62, "#fdebd0"), ("打击实施", 60, 82, "#fadbd8"),
        ("评估撤收", 78, 100, "#e8daef"),
    ]
    for name, s, e, col in phases:
        ax.add_patch(patches.Rectangle((s, 0.2), e-s, 11.5, fc=col, ec="none", zorder=0, alpha=0.55))
        ax.text((s+e)/2, 11.6, name, ha="center", va="bottom", fontsize=10, color="#5d6d7e", zorder=3)

    rows = [
        ("侦察监视与目标发现", 2, 0, 28, BLUE, "A-50U/前哨-R/电子侦察发现并识别哨舰"),
        ("多源情报融合与威胁研判", 3, 20, 26, BLUE, "识别为前出哨舰/伯克2A滋事, 建立稳定航迹"),
        ("指挥决策(惩戒打击决心)", 4, 45, 10, "#8e44ad", "哨舰回撤→下达有限打击决心"),
        ("任务筹划与火力规划", 5, 50, 12, ORANGE, "平台-载荷匹配, 目标指示, 形成打击方案"),
        ("兵力展开(反辐射/护航、无人机、电子战就位)", 6, 50, 18, ORANGE, "多域兵力前出与就位"),
        ("打击实施①-高超/弹道反舰", 7, 62, 8, "#d68910", "锆石/匕首 → AEGIS/SPY-1/指挥/卫通"),
        ("打击实施②-巡航弹(岸/潜/舰)", 8, 66, 10, "#c0392b", "口径/缟玛瑙 → 上层建筑/通信/动力"),
        ("打击实施③-反辐射/电子压制", 9, 60, 16, ORANGE, "苏-35S·Kh-31P/伊尔-20EW → 压制雷达/卫通/数据链"),
        ("无人机抵近侦察与战果评估(BDA)", 10, 72, 10, GREEN, "影像/电子确认毁伤与链路状态"),
        ("链路切割确认与兵力撤收恢复警戒", 11, 80, 18, "#5d6d7e", "确认与CSG链路中断, 编队撤收"),
    ]

    for label, y, s, dur, col, note in rows:
        ax.barh(y, dur, left=s, height=0.62, color=col, ec="white", lw=0.8, zorder=4)
        ax.text(s+0.5, y, label, va="center", ha="left", fontsize=9.5, fontweight="bold", zorder=6,
                color="white" if col in ("#d4a017", "#c0392b", "#8e44ad", "#5d6d7e") else "#1a1a2e")
        ax.text(s+dur+0.6, y, note, va="center", ha="left", fontsize=8.2, color="#34495e", zorder=6)

    mark = {0: "T0 哨舰滋事(600km)", 45: "哨舰回撤(触发)", 62: "打击开火", 82: "链路切割确认"}
    for tm, lab in mark.items():
        ax.axvline(tm, color="#e67e22", lw=1.2, ls="--", zorder=5)
        ax.text(tm, 1.1, lab, ha="center", va="bottom", fontsize=8.5, color="#b9770e", fontweight="bold", zorder=6)

    ax.add_patch(patches.Rectangle((62, 0.2), 20, 11.5, fc="none", ec=ORANGE, lw=2.2, zorder=5))
    ax.text(72, 0.3, "有效任务窗口(~8~15min仿真)", ha="center", va="bottom", fontsize=9.5,
            color=ORANGE, fontweight="bold", zorder=6)

    ax.set_yticks([])
    ax.set_xticks(range(0, 111, 10))
    ax.set_xlabel("行动时间 (分钟)", fontsize=11)
    ax.set_ylabel("作战行动条目", fontsize=11)
    ax.set_title("图3   行动过程甘特图 (对抗过程时间线)", fontsize=15, fontweight="bold", pad=14, color="#1a1a2e")

    handles = [
        patches.Patch(color=BLUE, label="侦察/情报"), patches.Patch(color="#8e44ad", label="决策"),
        patches.Patch(color=ORANGE, label="筹划/展开"), patches.Patch(color="#d68910", label="高超/弹道反舰"),
        patches.Patch(color="#c0392b", label="巡航弹(岸/潜/舰)"), patches.Patch(color=GREEN, label="无人机BDA"),
        patches.Patch(color=ORANGE, fc="none", ec=ORANGE, lw=2.2, label="任务机会窗口"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT + r"\图3_行动时间甘特图.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK] 图3")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("完成")
