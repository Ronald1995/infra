# -*- coding: utf-8 -*-
"""
生成《HCCL 与 HCOMM 在通信流程中的分工（UB 篇）》配图。
用法： python make_figures.py
依赖： matplotlib（微软雅黑）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

C_FW = "#6C757D"
C_HCCL = "#2E5E8C"
C_HCOMM = "#C1662F"
C_ENG = "#6A4C93"
C_HW = "#2F7D4F"
C_GRAY = "#8A8F98"
C_LIGHT = "#F4F6F8"
C_RED = "#B03A2E"


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title="", body="", ec=C_HCCL, fc="white",
        ts=13, bs=9.5, lw=1.5, bold=True, tcolor=None, bs_lines_gap=0.0):
    """矩形框（坐标即实际视觉范围，无 pad 膨胀）。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    cx = x + w / 2
    if title and body:
        nlines = body.count("\n") + 1
        ty = y + h * (0.76 if nlines <= 2 else 0.80)
        by = y + h * (0.36 if nlines <= 2 else 0.38)
        ax.text(cx, ty, title, ha="center", va="center", fontsize=ts,
                fontweight="bold" if bold else "normal", color=tcolor or ec, zorder=3)
        ax.text(cx, by, body, ha="center", va="center", fontsize=bs,
                color="#2B2B2B", zorder=3, linespacing=1.5)
    elif title:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=ts,
                fontweight="bold" if bold else "normal", color=tcolor or ec, zorder=3)
    elif body:
        ax.text(cx, y + h / 2, body, ha="center", va="center", fontsize=bs,
                color="#2B2B2B", zorder=3, linespacing=1.5)


def lbox(ax, x, y, w, h, title, body="", ec=C_HCCL, fc="white", ts=12, bs=8.8, lw=1.5):
    """左对齐标题的框（标题左上，正文左下）。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + 0.014, y + h * 0.66, title, ha="left", va="center", fontsize=ts,
            fontweight="bold", color=ec, zorder=3)
    if body:
        ax.text(x + 0.014, y + h * 0.29, body, ha="left", va="center", fontsize=bs,
                color="#2B2B2B", zorder=3)


def arrow(ax, p1, p2, color=C_GRAY, lw=1.6, style="-|>", ls="-", rad=0.0, ms=12):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, linewidth=lw,
                        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                        shrinkA=1, shrinkB=1, zorder=4)
    ax.add_patch(a)


def title(ax, main, sub=""):
    ax.text(0.5, 0.962, main, ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1B1B1B")
    if sub:
        ax.text(0.5, 0.914, sub, ha="center", va="center", fontsize=10, color="#5A5A5A")


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------- 图1 类比
def fig1():
    fig, ax = new_fig(12, 6.2)
    title(ax, "三方分工：HCCL 定方案，HCOMM 供运力，UB 是路网")

    xs = [0.055, 0.365, 0.675]
    w = 0.27
    heads = [("HCCL", "集合通信库"), ("HCOMM", "通信基础库"), ("UB 硬件", "统一总线")]
    cols = [C_HCCL, C_HCOMM, C_HW]
    tech = [
        "算子入口 → 算法选择 → 算法执行\n决定：用哪个算法、数据切几份、分几步",
        "通信域 + 拓扑 + 资源 + 引擎\n提供：跟谁连、走哪条路、谁来搬",
        "Jetty / SQE / CQE / 端口 / 链路\n保证：字节真的过去了",
    ]
    life = [
        "物流调度中心\n规划线路、把一批货拆成几单、排班",
        "物流园区\n仓库、车队、装卸泊位、签收回执",
        "高速路网\n车道、收费站、限速",
    ]
    for i in range(3):
        box(ax, xs[i], 0.60, w, 0.24, heads[i][0], heads[i][1], ec=cols[i], ts=16, bs=10.5)
        box(ax, xs[i], 0.38, w, 0.19, "技术职责", tech[i], ec=cols[i], ts=10.5, bs=9.2)
        box(ax, xs[i], 0.14, w, 0.19, "一句话类比", life[i], ec=C_GRAY, ts=10.5, bs=9.2,
            tcolor="#444444")

    for i in range(2):
        arrow(ax, (xs[i] + w + 0.002, 0.72), (xs[i + 1] - 0.002, 0.72),
              color="#999999", lw=1.8, ms=14)

    ax.text(0.5, 0.072, "记忆口诀：HCCL 说“怎么算”，HCOMM 说“怎么搬”，UB 说“搬得动”。",
            ha="center", fontsize=11, color=C_RED)
    save(fig, "f1_roles.png")


# ---------------------------------------------------------------- 图2 分层
def fig2():
    fig, ax = new_fig(11.8, 7.6)
    title(ax, "软件分层：一次通信要穿过 6 层",
          "左边这列是“谁在回答”，右边这列是“回答什么问题”")

    layers = [
        ("AI 框架 / 推理引擎", "torch_npu / MindSpore / vLLM-Ascend", C_FW,
         "调 all_reduce 的人"),
        ("HCCL 集合通信算子层", "入口 → 算法选择 → 算法执行（hccl/src/ops）", C_HCCL,
         "用哪个算法？切几份？分几步？"),
        ("HCOMM 通信域管理（HCCM）", "通信域 / 拓扑 RankGraph / 资源配置", C_HCOMM,
         "和谁通信？有哪些路可走？"),
        ("HCOMM 基础通信（base_comm）", "Endpoint / Channel / Jetty / Notify / CommMem", C_HCOMM,
         "这条路怎么建、怎么用？"),
        ("通信引擎", "AICPU_TS / AIV / CCU / CPU_TS", C_ENG,
         "谁写工单？谁按门铃？"),
        ("UB 硬件", "HiLink / x4 端口 / UB_CTP / UB_RTP / UBoE / UB_MEM", C_HW,
         "电信号怎么过去？"),
    ]
    ytop, hh, gap = 0.845, 0.105, 0.018
    for i, (name, detail, color, question) in enumerate(layers):
        y = ytop - i * (hh + gap) - hh
        lbox(ax, 0.085, y, 0.475, hh, name, detail, ec=color,
             fc=C_LIGHT if i % 2 == 0 else "white", ts=11.5, bs=8.6)
        ax.text(0.585, y + hh / 2, question, ha="left", va="center", fontsize=9.4,
                color="#333333")

    # 控制面 / 数据面 竖带
    ax.add_patch(Rectangle((0.020, 0.384), 0.040, 0.330, fc="#EAF0F6", ec="none", zorder=1))
    ax.text(0.040, 0.549, "控制面（一次性）", rotation=90, ha="center", va="center",
            fontsize=9.6, color=C_HCCL)
    ax.add_patch(Rectangle((0.020, 0.120), 0.040, 0.228, fc="#F7EEE8", ec="none", zorder=1))
    ax.text(0.040, 0.234, "数据面（每次都走）", rotation=90, ha="center", va="center",
            fontsize=9.6, color=C_HCOMM)

    ax.annotate("", xy=(0.955, 0.845), xytext=(0.955, 0.135),
                arrowprops=dict(arrowstyle="<->", color="#BBBBBB", lw=1.2))
    ax.text(0.975, 0.49, "越往下越接近硬件", rotation=90, ha="center", va="center",
            fontsize=8.6, color="#999999")

    ax.text(0.5, 0.060,
            "注意：base_comm 横跨两面——建资源属控制面，执行 Write/Read/Notify 属数据面。",
            ha="center", fontsize=9.8, color="#333333")
    ax.text(0.5, 0.024,
            "HCCL 只在上面两层做“决策”；真正把字节搬走的是下面四层。",
            ha="center", fontsize=10.2, color=C_RED)
    save(fig, "f2_stack.png")


# ---------------------------------------------------------------- 图3 对象模型
def fig3():
    fig, ax = new_fig(12.2, 7.2)
    title(ax, "对象模型：从“通信域”一路收敛到“一个 Jetty”",
          "Channel = 两端 Endpoint + 协议 + N 个 Notify；UB 下 1 个 Channel 绑 1 个 Jetty")

    box(ax, 0.045, 0.70, 0.245, 0.125, "通信域 HcclComm", "一次 init 得到，管 N 个 rank",
        ec=C_HCOMM, ts=11.2, bs=8.8)
    box(ax, 0.045, 0.51, 0.245, 0.125, "Rank（每张卡一个）", "rankId / rankSize",
        ec=C_HCOMM, ts=11.2, bs=8.8)
    box(ax, 0.045, 0.32, 0.245, 0.125, "Endpoint（网口）", "协议 + 地址（950 上是 EID）",
        ec=C_HCOMM, ts=11.2, bs=8.8)
    box(ax, 0.045, 0.13, 0.245, 0.125, "Channel（数据通道）", "一对 Endpoint 间可建多条",
        ec=C_HCOMM, ts=11.2, bs=8.8)

    for y0, y1 in [(0.70, 0.635), (0.51, 0.445), (0.32, 0.255)]:
        arrow(ax, (0.1675, y0), (0.1675, y1), color=C_HCOMM)

    ax.add_patch(FancyBboxPatch((0.345, 0.10), 0.615, 0.585,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                fc="#FBFBFB", ec=C_HCOMM, lw=1.3, ls="--", zorder=1))
    ax.text(0.360, 0.655, "把一条 UB Channel 打开", fontsize=10.8, fontweight="bold",
            color=C_HCOMM)

    box(ax, 0.365, 0.385, 0.27, 0.235, "Jetty（收发队列）",
        "SQ：软件写 SQE\nCQ：硬件写 CQE\nJFS / JFC：门铃", ec=C_HW, ts=10.5, bs=9)
    box(ax, 0.665, 0.495, 0.275, 0.125, "Notify × N", "跨 rank 的同步信号\n按序号寻址",
        ec=C_ENG, ts=10.5, bs=8.8)
    box(ax, 0.665, 0.360, 0.275, 0.115, "CommMem（window）", "已注册内存，对端可直读写",
        ec=C_ENG, ts=10.5, bs=8.8)
    box(ax, 0.365, 0.145, 0.575, 0.165, "通信引擎 CommEngine",
        "AICPU_TS（默认）/ AIV / CCU / CPU_TS\n决定“谁来写 SQE、谁来按铃”",
        ec=C_ENG, ts=10.5, bs=8.8)

    arrow(ax, (0.292, 0.192), (0.342, 0.290), color=C_HCOMM, rad=-0.15, ms=14)

    ax.text(0.5, 0.045,
            "Channel = 两端 Endpoint + 通信协议 + N 个 Notify        UB 下：1 个 Channel = 1 个 Jetty",
            ha="center", fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F1F4F7", ec="#D5DCE4"))
    save(fig, "f3_objects.png")


# ---------------------------------------------------------------- 图4 六步
def fig4():
    fig, ax = new_fig(14, 6.6)
    title(ax, "一次 AllReduce 的六步旅程：每一步该找谁",
          "第 1～3 步偏控制面（可复用），第 4～6 步是数据面（每次调用都走）")

    steps = [
        ("① 进算子", "HCCL", "HcclAllReduce()\n填 OpParam", C_HCCL),
        ("② 查拓扑", "HCOMM", "RankGraph：层数、\n1DMESH / CLOS、Links", C_HCOMM),
        ("③ 选算法 + 引擎", "HCCL", "Selector + CostModel\n→ algName / engine", C_HCCL),
        ("④ 建资源", "HCOMM", "Endpoint / Channel /\nJetty / Notify / Thread", C_HCOMM),
        ("⑤ 任务编排", "HCCL", "Executor + Template\n切 slice、排步骤", C_HCCL),
        ("⑥ 下发执行", "HCOMM + 引擎", "HcommWriteOnThread\n→ SQE → UB 搬运", C_ENG),
    ]
    w, gap = 0.135, 0.020
    x0 = 0.045
    ybox, hbox = 0.375, 0.335
    for i, (name, owner, detail, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, ybox, w, hbox, "", "", ec=color)
        ax.text(x + w / 2, ybox + 0.265, owner, ha="center", va="center", fontsize=8.2,
                color="white", bbox=dict(boxstyle="round,pad=0.30", fc=color, ec="none"),
                zorder=3)
        ax.text(x + w / 2, ybox + 0.165, name, ha="center", va="center", fontsize=9.6,
                fontweight="bold", color=color, zorder=3)
        ax.text(x + w / 2, ybox + 0.078, detail, ha="center", va="center", fontsize=8.2,
                color="#2B2B2B", linespacing=1.45, zorder=3)
        if i < 5:
            arrow(ax, (x + w + 0.001, ybox + hbox / 2), (x + w + gap - 0.001, ybox + hbox / 2),
                  color="#9AA0A6", ms=10)

    # 控制面 / 数据面 分组
    ctrl_right = x0 + 2 * (w + gap) + w
    ax.add_patch(FancyBboxPatch((x0 - 0.010, 0.318), ctrl_right - x0 + 0.020, 0.038,
                                boxstyle="round,pad=0,rounding_size=0.006",
                                fc="#EAF0F6", ec="none"))
    ax.text((x0 + ctrl_right) / 2, 0.337, "偏控制面：建链、建资源，可复用 / 可缓存",
            ha="center", va="center", fontsize=9, color=C_HCCL)
    data_left = x0 + 3 * (w + gap)
    data_right = x0 + 6 * w + 5 * gap
    ax.add_patch(FancyBboxPatch((data_left - 0.010, 0.318), data_right - data_left + 0.020, 0.038,
                                boxstyle="round,pad=0,rounding_size=0.006",
                                fc="#F7EEE8", ec="none"))
    ax.text((data_left + data_right) / 2, 0.337, "数据面：每次通信都真的搬数据",
            ha="center", va="center", fontsize=9, color=C_HCOMM)

    # 责任带
    ax.add_patch(FancyBboxPatch((x0 - 0.010, 0.145), (ctrl_right - x0) + 0.020, 0.10,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                fc=C_HCCL, ec="none"))
    ax.text((x0 + ctrl_right) / 2, 0.195, "HCCL：①③ 入口与选路，⑤ 切片与编排",
            ha="center", va="center", fontsize=9.6, color="white")
    ax.add_patch(FancyBboxPatch((data_left - 0.010, 0.145), (data_right - data_left) + 0.020, 0.10,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                fc=C_HCOMM, ec="none"))
    ax.text((data_left + data_right) / 2, 0.195, "HCOMM：②④ 拓扑与资源，⑥ 搬运与同步",
            ha="center", va="center", fontsize=9.6, color="white")

    ax.text(0.5, 0.075, "② 与 ④ 由 HCOMM 提供，但“要不要查、建几条”由 HCCL 的算法决定。",
            ha="center", fontsize=10, color=C_RED)
    save(fig, "f4_six_steps.png")


# ---------------------------------------------------------------- 图5 AllReduce 分解
def fig5():
    fig, ax = new_fig(12.6, 7.4)
    title(ax, "AllReduce 为什么要拆成三段：分级通信",
          "以 2 台 Server × 4 张卡为例（三级拓扑时，中间再多一段超节点间）")

    for s, x in [(0, 0.055), (1, 0.545)]:
        ax.add_patch(FancyBboxPatch((x, 0.395), 0.395, 0.275,
                                    boxstyle="round,pad=0,rounding_size=0.010",
                                    fc="#F7F9FB", ec="#B9C4CF", lw=1.2, zorder=1))
        ax.text(x + 0.197, 0.635, f"Server {s}", ha="center", fontsize=10.2, color="#4A5560")
        for r in range(4):
            cx = x + 0.030 + r * 0.089
            box(ax, cx, 0.455, 0.070, 0.130, f"R{r}", "", ec=C_HW, ts=9.2)
        ax.text(x + 0.197, 0.425, "Server 内：1DMESH 直连（netLayer 0）", ha="center",
                fontsize=8.6, color=C_HW)

    arrow(ax, (0.452, 0.520), (0.543, 0.520), color=C_HW, lw=2.0, style="<|-|>", ms=14)
    ax.text(0.498, 0.560, "CLOS\n(netLayer 1)", ha="center", va="bottom", fontsize=8.4,
            color=C_HW)

    stages = [
        ("阶段一  节点内 ReduceScatter", "每张卡把自己那份切 4 片，与邻居两两归约\n→ 每卡只剩 1/4 数据（已归约）", C_HCCL),
        ("阶段二  节点间 AllReduce", "同一位置的卡跨机交换并归约\n（NHR：log2(N) 步；Ring：N-1 步）", C_HCOMM),
        ("阶段三  节点内 AllGather", "把归约好的 1/4 拼回完整结果\n→ 每卡都拿到全量 AllReduce 结果", C_ENG),
    ]
    for i, (name, detail, color) in enumerate(stages):
        x = 0.050 + i * 0.315
        box(ax, x, 0.160, 0.285, 0.155, name, detail, ec=color, ts=10.0, bs=8.6)
        if i < 2:
            arrow(ax, (x + 0.287, 0.238), (x + 0.313, 0.238), color="#9AA0A6", ms=10)

    ax.text(0.5, 0.098,
            "同样一次 AllReduce，950 上还可换成 MeshOneShot（小数据）/ MeshTwoShot / AHC / MeshMultiJetty…… 选谁由 HCCL 决定。",
            ha="center", fontsize=9.2, color="#333333")
    ax.text(0.5, 0.045,
            "为什么要分级？段内链路快、段间链路慢：让“慢链路少传、快链路多传”就是 HCCL 算法的主要工作。",
            ha="center", fontsize=9.6, color=C_RED)
    save(fig, "f5_allreduce.png")


# ---------------------------------------------------------------- 图6 引擎
def fig6():
    fig, ax = new_fig(13.2, 7.0)
    title(ax, "四种通信引擎：同样一条 SQE，由谁写出来",
          "展开模式由 HCCL_OP_EXPANSION_MODE 决定；950 默认 AICPU_TS")

    rows = [
        ("AICPU_TS（默认）", "Host 提交 AICPU Kernel → TS 调度到 AICPU →\nAICPU 提交通信 Task → TS 下发执行器",
         "AICPU 写\nSQE / 描述符", C_HCCL, "大数据、高带宽"),
        ("AIV", "Host 提交 AIV Kernel → TS 调度到 Vector Core →\nVector Core 执行通信算子、直接写 SQE",
         "Vector Core\n写 SQE", C_ENG, "小数据、低时延"),
        ("CCU", "Host 下发 CCU 指令序列 + CCU Kernel →\nCCU 执行微码，用 URMA 搬数据",
         "CCU 硬件\n发 WQE", C_HCOMM, "高带宽低时延"),
        ("CPU_TS（A2）", "Host CPU 跑通信逻辑 → TS 下发 → 硬件执行",
         "Host CPU\n写任务", C_GRAY, "A2 专用"),
    ]
    ytop, hh, gap = 0.760, 0.140, 0.026
    for i, (name, how, who, color, fit) in enumerate(rows):
        y = ytop - i * (hh + gap) - hh
        box(ax, 0.045, y, 0.185, hh, name, fit, ec=color, ts=10.2, bs=8.2)
        box(ax, 0.250, y, 0.395, hh, "", how, ec="#C9CED4", bs=8.5)
        box(ax, 0.665, y, 0.135, hh, "", who, ec=color, bs=8.4)
        box(ax, 0.820, y, 0.135, hh, "", "UB\nJetty", ec=C_HW, bs=8.6)
        arrow(ax, (0.231, y + hh / 2), (0.249, y + hh / 2), color="#9AA0A6", ms=9)
        arrow(ax, (0.646, y + hh / 2), (0.664, y + hh / 2), color="#9AA0A6", ms=9)
        arrow(ax, (0.801, y + hh / 2), (0.819, y + hh / 2), color="#9AA0A6", ms=9)

    ax.text(0.5, 0.070,
            "四种引擎的差别不在“算法”，而在“谁来执行算法”：HCCL 选算法，引擎决定执行体，HCOMM 提供执行体与队列。",
            ha="center", fontsize=9.6, color=C_RED)
    save(fig, "f6_engines.png")


# ---------------------------------------------------------------- 图7 UB 协议
def fig7():
    fig, ax = new_fig(13.0, 7.6)
    title(ax, "UB 硬件画像与协议地图（Ascend 950）",
          "同一张卡上，不同引擎、不同语义走不同协议：不是“一条 UB 走天下”")

    ax.add_patch(FancyBboxPatch((0.035, 0.500), 0.425, 0.360,
                                boxstyle="round,pad=0,rounding_size=0.010",
                                fc="#F4F9F5", ec=C_HW, lw=1.3))
    ax.text(0.052, 0.825, "硬件画像：2 AI Die + 2 IO Die", fontsize=10.2,
            fontweight="bold", color=C_HW)
    hw = [
        "HiLink SerDes：72 lane × 112 Gbps",
        "切成 18 个 x4 端口；双向 2016 GB/s（受光模块限制）",
        "UBoE：2 × 400 Gbps，与 UB 共享 2 个端口",
        "片内：每个 IO Die 9 个 x4 端口转发，不过 DRAM",
        "调度器 STARS2.0：64 UB jetty / 32 CCU 任务",
    ]
    for i, t in enumerate(hw):
        ax.text(0.050, 0.772 - i * 0.052, "• " + t, fontsize=8.5, color="#2B2B2B")

    box(ax, 0.035, 0.235, 0.425, 0.215, "两种语义，两条路径",
        "网络语义：写 SQE 去读/写远端（UB_CTP / UB_RTP）\n"
        "内存语义：远端内存映射到本地地址空间，\n直接 load/store（UB_MEM）",
        ec=C_ENG, ts=10.2, bs=8.5)

    ax.text(0.520, 0.845, "协议 × 语义 × 引擎", fontsize=10.8, fontweight="bold",
            color="#1B1B1B")
    headers = ["协议", "语义", "950 上谁可驱动", "典型场景"]
    colx = [0.520, 0.595, 0.650, 0.815]
    ax.add_patch(Rectangle((0.512, 0.760), 0.475, 0.042, fc="#E8EDF3", ec="none"))
    for x, h in zip(colx, headers):
        ax.text(x, 0.781, h, fontsize=8.8, fontweight="bold", color="#2B2B2B")

    rows = [
        ("UB_CTP", "网络", "AICPU_TS / AIV\n/ CCU / CPU", "超节点内主力\n（1DMESH 直连）", C_HW),
        ("UB_RTP", "网络", "AICPU_TS", "跨框多端口\n（CLOS）", C_HW),
        ("UBoE", "网络", "AICPU_TS", "以太封装\n（IPv4 建链）", C_HCOMM),
        ("UB_MEM", "内存", "AICPU_TS / AIV", "对称内存\n直接 load/store", C_ENG),
        ("RoCE", "网络", "CPU / AICPU_TS / AIV", "兼容路径", C_GRAY),
        ("HCCS", "内存", "—（A2/A3）", "非 950 的\n片间互连", C_GRAY),
    ]
    for i, (p, sem, eng, scene, color) in enumerate(rows):
        y = 0.712 - i * 0.068
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.512, y - 0.026), 0.475, 0.068, fc="#FAFBFC", ec="none"))
        ax.text(colx[0], y, p, fontsize=8.8, fontweight="bold", color=color)
        ax.text(colx[1], y, sem, fontsize=8.4, color="#2B2B2B")
        ax.text(colx[2], y, eng, fontsize=7.8, color="#2B2B2B", linespacing=1.35)
        ax.text(colx[3], y, scene, fontsize=7.8, color="#2B2B2B", linespacing=1.35)

    ax.text(0.75, 0.250,
            "协议不是用户随便挑的：由拓扑（1DMESH / CLOS）\n+ 引擎（AICPU / AIV / CCU）+ 语义需求共同决定。",
            ha="center", fontsize=9.0, color=C_RED,
            bbox=dict(boxstyle="round,pad=0.45", fc="#FDF6F4", ec="#E8CFC7"))
    ax.text(0.75, 0.140,
            "记住两条：CCU 只走 UB_CTP；\n多 jetty（多 channel）只在 CLOS + UB(CTP/RTP) + AICPU 下生效。",
            ha="center", fontsize=9.0, color="#333333",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F1F4F7", ec="#D5DCE4"))
    save(fig, "f7_ub_map.png")


# ---------------------------------------------------------------- 图8 排错
def fig8():
    fig, ax = new_fig(12.6, 6.8)
    title(ax, "排错地图：先定位“该找谁”，再查细节",
          "同一句“通信卡住”，落在不同层，处理方式完全不同")

    layers = [
        ("框架层", "调用序列不一致 / 某个 rank 没进集合通信", "EI0002 超时", C_FW),
        ("HCCL 层", "算法与数据量/拓扑不匹配、图捕获并发过多", "EI0003 / 精度异常", C_HCCL),
        ("HCOMM 域层", "rank table 不一致、拓扑探测失败、建链失败", "EI0004 / EI0014 / EI0015", C_HCOMM),
        ("HCOMM 资源层", "Channel / Jetty / Notify 不够或泄漏", "EI0007 / no notify resource", C_HCOMM),
        ("引擎 / 传输层", "Jetty 超时、UB CQE 错误、RoCE CQE 错误", "EI0018 / EI0013 / HCOMM_TA_*", C_ENG),
        ("硬件 / 链路层", "光模块、端口闪断、对端进程退出", "链路告警 + CQE 错误", C_HW),
    ]
    ytop, hh, gap = 0.800, 0.090, 0.022
    for i, (name, symptom, code, color) in enumerate(layers):
        y = ytop - i * (hh + gap) - hh
        box(ax, 0.045, y, 0.170, hh, name, "", ec=color, ts=10.0)
        ax.text(0.235, y + hh / 2, symptom, ha="left", va="center", fontsize=9.0,
                color="#2B2B2B")
        ax.text(0.700, y + hh / 2, code, ha="left", va="center", fontsize=8.6, color=color)

    ax.annotate("", xy=(0.912, 0.135), xytext=(0.912, 0.795),
                arrowprops=dict(arrowstyle="-|>", color="#BBBBBB", lw=1.2))
    ax.text(0.930, 0.465, "越往下：越先怀疑链路/资源", rotation=90, ha="center",
            va="center", fontsize=8.4, color="#999999")

    ax.text(0.5, 0.055,
            "顺序建议：先确认平台（950 才看 Jetty/UB CQE）→ 再看是“超时”还是“错误 CQE” → 最后才动环境变量。",
            ha="center", fontsize=9.4, color=C_RED)
    save(fig, "f8_troubleshoot.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
