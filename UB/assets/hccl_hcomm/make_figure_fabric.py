# -*- coding: utf-8 -*-
"""950 超节点：18 个 x4 端口怎么用（带宽账本示意）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))

C_UB = "#2F7D4F"
C_CLOS = "#2E5E8C"
C_ETH = "#C1662F"
C_CHIP = "#6A4C93"
C_GRAY = "#8A8F98"


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title="", body="", ec=C_UB, fc="white", ts=12, bs=9,
        lw=1.5, ls="-", tcolor=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=lw, edgecolor=ec, facecolor=fc, ls=ls, zorder=2))
    cx = x + w / 2
    if title and body:
        ax.text(cx, y + h * 0.72, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", color=tcolor or ec, zorder=3)
        ax.text(cx, y + h * 0.36, body, ha="center", va="center", fontsize=bs,
                color="#2B2B2B", zorder=3, linespacing=1.5)
    elif title:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", color=tcolor or ec, zorder=3)
    else:
        ax.text(cx, y + h / 2, body, ha="center", va="center", fontsize=bs,
                color="#2B2B2B", zorder=3, linespacing=1.5)


def arrow(ax, p1, p2, color=C_GRAY, lw=1.8, style="-|>", ms=13):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, shrinkA=1, shrinkB=1, zorder=4))


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("saved", path)


fig, ax = new_fig(13.4, 7.8)
ax.text(0.5, 0.955, "超节点组网：一颗 950 的 18 个 x4 口是怎么分的",
        ha="center", fontsize=16, fontweight="bold", color="#1B1B1B")
ax.text(0.5, 0.905, "一个 x4 口 = 一条车道；协议 = 交通规则；拓扑（1DMESH / CLOS）= 这些车道连到哪张路网",
        ha="center", fontsize=10.5, color="#555555")

# ---------------- 左：NPU 芯片
ax.add_patch(FancyBboxPatch((0.030, 0.470), 0.300, 0.385,
                            boxstyle="round,pad=0,rounding_size=0.014",
                            fc="#F8FAFB", ec="#B9C4CF", lw=1.4, zorder=1))
ax.text(0.045, 0.830, "一颗 Ascend 950（2 AI Die + 2 IO Die）", fontsize=9.6,
        fontweight="bold", color="#4A5560")

box(ax, 0.045, 0.720, 0.130, 0.080, "AI Die 0", "Cube + Vector", ec=C_CHIP, ts=9, bs=7.6)
box(ax, 0.185, 0.720, 0.130, 0.080, "AI Die 1", "Cube + Vector", ec=C_CHIP, ts=9, bs=7.6)
box(ax, 0.045, 0.590, 0.130, 0.100, "IO Die 0", "9 个 x4 口", ec=C_UB, ts=9, bs=7.8)
box(ax, 0.185, 0.590, 0.130, 0.100, "IO Die 1", "9 个 x4 口", ec=C_UB, ts=9, bs=7.8)

# 端口小方块：每 Die 9 个
for die_x in (0.045, 0.185):
    for i in range(9):
        ax.add_patch(Rectangle((die_x + 0.006 + i * 0.0138, 0.600), 0.0095, 0.017,
                               fc=C_UB, ec="none", alpha=0.75, zorder=3))
ax.text(0.180, 0.562, "← 每个小方块 = 1 个 x4 口（4 lane × 112 Gbps ≈ 56 GB/s 单向）",
        ha="center", fontsize=7.8, color=C_UB)

box(ax, 0.045, 0.485, 0.130, 0.062, "UBoE 2 × 400 G", "占 2 个口", ec=C_ETH, ts=8.2, bs=7.4)
box(ax, 0.185, 0.485, 0.130, 0.062, "PCIe 5.0 x16", "128 GB/s，不占 UB 口", ec=C_GRAY, ts=8.2, bs=7.4)

# ---------------- 中：UB 交换网络
box(ax, 0.395, 0.610, 0.230, 0.215, "超节点内 UB 交换网络",
    "1DMESH（逻辑全互联）\nnetLayer 0 / PEER2PEER\n任意两卡一跳可达", ec=C_UB, ts=11, bs=8.8,
    fc="#F4F9F5")
ax.text(0.510, 0.575, "示例：Server 内链路只占 1 个口\n（Die1 的 0 号口，即 \"1/0\"）",
        ha="center", fontsize=8.0, color="#4A5560")

# ---------------- 右：CLOS
box(ax, 0.700, 0.610, 0.265, 0.215, "CLOS 交换网络",
    "胖树 / 经交换机 → 其他 Server\nnetLayer 1 / PEER2NET\n示例：6 个口（Die0 4~7 + Die1 5~6）",
    ec=C_CLOS, ts=11, bs=8.6, fc="#F5F8FC")
ax.text(0.832, 0.575, "6 口 ≈ 336 GB/s 单向（每卡上行）",
        ha="center", fontsize=8.2, color=C_CLOS)

# ---------------- 箭头
arrow(ax, (0.335, 0.700), (0.392, 0.700), color=C_UB, lw=2.4, ms=15)
ax.text(0.363, 0.735, "18 口", ha="center", fontsize=8.6, color=C_UB)
arrow(ax, (0.628, 0.700), (0.697, 0.700), color=C_CLOS, lw=2.2, ms=15)
ax.text(0.662, 0.735, "分口", ha="center", fontsize=8.6, color=C_CLOS)


# ---------------- 右下：带宽账本
ax.add_patch(FancyBboxPatch((0.395, 0.245), 0.570, 0.245,
                            boxstyle="round,pad=0,rounding_size=0.012",
                            fc="#FDF6F4", ec="#E8CFC7", lw=1.2))
ax.text(0.415, 0.455, "带宽账本（所有数字都是“单向”，双向请 ×2）", fontsize=9.8,
        fontweight="bold", color="#B03A2E")
lines = [
    "① 单口：4 lane × 112 Gbps = 448 Gbps = 56 GB/s",
    "② 整芯片：18 口 × 56 = 1008 GB/s  →  双向 2016 GB/s",
    "③ 分给 CLOS 6 口：6 × 56 = 336 GB/s（剩下 12 口留给 Server 内）",
    "④ 若 2 口改 UBoE（2×400 G = 50 GB/s/口）：UB 侧只剩 16 口 ≈ 896 GB/s",
]
for i, t in enumerate(lines):
    ax.text(0.415, 0.398 - i * 0.047, t, fontsize=8.7, color="#2B2B2B")

# ---------------- 左下：lane / port 概念
ax.add_patch(FancyBboxPatch((0.030, 0.230), 0.340, 0.225,
                            boxstyle="round,pad=0,rounding_size=0.012",
                            fc="#F1F4F7", ec="#D5DCE4", lw=1.2))
ax.text(0.050, 0.425, "三个词的区别", fontsize=9.8, fontweight="bold", color="#333333")
concepts = [
    "lane：一对差分线，112 Gbps 是它的“限速”",
    "port（x4）：4 条 lane 捆一起，448 Gbps",
    "aggregate：18 口带宽之和，即整芯片出口",
]
for i, t in enumerate(concepts):
    ax.text(0.050, 0.372 - i * 0.045, t, fontsize=8.7, color="#2B2B2B")

# ---------------- 底部注
ax.text(0.5, 0.170,
        "为什么 2016 GB/s 是“上限”而不是“实测”：它假设 18 个口全开、每口跑满 112G，且光模块不掉速。",
        ha="center", fontsize=9.4, color="#333333")
ax.text(0.5, 0.115,
        "真实集合通信能拿多少，取决于这次用了几个口（算法/拓扑决定）、走什么协议、开几条 jetty。",
        ha="center", fontsize=9.4, color="#333333")
ax.text(0.5, 0.055,
        "印证：HCCL 的 MeshMultiLink 按“各链路包含的 port 数”分流，MeshConcur 按 level0/level1 两侧带宽比分流。",
        ha="center", fontsize=9.6, color="#B03A2E")

save(fig, "f9_supernode_fabric.png")
