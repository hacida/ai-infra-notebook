"""Rebuild the beginner diagrams used by transformer-evolution/*.md.

Run: python draw_transformer_evolution.py
The output is PNG; no SVG assets are needed by the articles.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent / "images" / "transformer-evolution"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1600, 900
NAVY, BLUE, TEAL, ORANGE = "#14243d", "#2976d2", "#159d94", "#ee8b36"
PALE, WHITE, GRAY = "#f5f8fb", "#ffffff", "#536579"
FONT = "C:/Windows/Fonts/msyh.ttc"


def f(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def canvas(title, subtitle):
    im = Image.new("RGB", (W, H), PALE)
    d = ImageDraw.Draw(im)
    d.text((60, 40), title, font=f(46, True), fill=NAVY)
    d.text((62, 112), subtitle, font=f(25), fill=GRAY)
    return im, d


def box(d, xy, title, color=BLUE, fill=WHITE, size=29):
    left, top, right, bottom = xy
    d.rounded_rectangle((left+5, top+7, right+5, bottom+7), radius=18,
                        fill="#e7edf3")
    d.rounded_rectangle(xy, radius=18, fill=fill, outline=color, width=4)
    bounds = d.textbbox((0, 0), title, font=f(size, True))
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    d.text((left + (right-left-tw)/2, top + (bottom-top-th)/2 - 4), title, font=f(size, True), fill=NAVY)


def arrow(d, start, end, color=TEAL, width=6):
    import math
    d.line([start, end], fill=color, width=width)
    a = math.atan2(end[1]-start[1], end[0]-start[0])
    l = 17
    p1 = (end[0]-l*math.cos(a-.55), end[1]-l*math.sin(a-.55))
    p2 = (end[0]-l*math.cos(a+.55), end[1]-l*math.sin(a+.55))
    d.polygon([end, p1, p2], fill=color)


def label(d, xy, s, color=GRAY, size=25):
    d.text(xy, s, font=f(size), fill=color)


# 1: visibility is a query-by-key rule, not a left-to-right arrow metaphor.
im, d = canvas("一张表看懂“能看见谁”", "行是当前 Query，列是可读取的 Key；绿色格表示可见")
xs = [470, 740, 1010]
ys = [290, 430, 570]
for left, name in [(100, "Encoder：双向"), (840, "Decoder：因果")]:
    label(d, (left, 205), name, NAVY, 31)
    for j, tok in enumerate(["猫", "追", "狗"]):
        label(d, (left+155+j*115, 258), tok, NAVY, 29)
        label(d, (left+75, 325+j*115), tok, NAVY, 29)
    for i in range(3):
        for j in range(3):
            visible = left == 100 or j <= i
            x0, y0 = left+145+j*115, 318+i*115
            d.rounded_rectangle((x0, y0, x0+86, y0+86), radius=13,
                                fill="#d8f3ec" if visible else "#e6eaf0",
                                outline=TEAL if visible else "#ccd4dd", width=3)
            if visible:
                d.ellipse((x0+29, y0+29, x0+57, y0+57), fill=TEAL)
            else:
                d.line((x0+31, y0+31, x0+55, y0+55), fill=GRAY, width=4)
                d.line((x0+55, y0+31, x0+31, y0+55), fill=GRAY, width=4)
box(d, (120, 740, 710, 820), "读完整输入：右侧也已知", TEAL, "#e9f7f3", 26)
box(d, (875, 740, 1480, 820), "逐词预测：未来必须遮住", ORANGE, "#fff2e7", 26)
im.save(OUT / "10-visibility-masks.png")


# 2: storage versus retrieval in MLA, with no unsupported size ranking.
im, d = canvas("MLA：历史留下“底稿”，当前再按头提问", "只画内容路径；实际 MLA 还单独缓存 RoPE 位置 Key")
box(d, (90, 320, 340, 450), "历史 Token", BLUE, "#eaf2ff")
arrow(d, (340, 385), (475, 385))
box(d, (475, 295, 810, 475), "共享 latent  c_j", TEAL, "#e9f7f3")
label(d, (535, 495), "每个历史位置存一份", TEAL)
for y, head in [(240, "Head 1 的读取参数"), (470, "Head 2 的读取参数"), (700, "Head 3 的读取参数")]:
    box(d, (1070, y, 1470, y+110), head, BLUE, WHITE, 27)
    arrow(d, (810, 385), (1060, y+55), TEAL)
box(d, (90, 680, 625, 780), "MQA：各头共用同一对完整 K/V", ORANGE, "#fff2e7", 23)
label(d, (710, 835), "MLA：同一底稿，各头用不同参数解读", NAVY, 27)
im.save(OUT / "11-mla-latent-reading.png")


# 3: elementary feature gating, versus changing a scalar activation.
im, d = canvas("SwiGLU：同一个 Token 走两条路，再逐通道相乘", "这里没有跨 Token 的信息交换；门值也不是二值开关")
box(d, (80, 340, 300, 470), "输入 x", BLUE, "#eaf2ff")
box(d, (500, 215, 845, 355), "值分支  Wu x", BLUE, WHITE)
box(d, (500, 475, 845, 615), "门分支  SiLU(Wg x)", ORANGE, WHITE, 27)
arrow(d, (300, 405), (500, 285), BLUE)
arrow(d, (300, 405), (500, 545), ORANGE)
box(d, (1010, 335, 1230, 485), "逐元素 ×", TEAL, "#e9f7f3")
arrow(d, (845, 285), (1010, 395), TEAL)
arrow(d, (845, 545), (1010, 425), TEAL)
box(d, (1340, 340, 1530, 480), "Wo → 输出", BLUE, WHITE, 24)
arrow(d, (1230, 410), (1340, 410), TEAL)
label(d, (535, 715), "示例：值 [2, 3] × 门 [0.1, 0.9] = [0.2, 2.7]", NAVY, 30)
im.save(OUT / "12-swiglu-two-paths.png")


# 4: route a token to two experts and combine, separating parameter storage and work.
im, d = canvas("MoE：八套 Expert 放在模型里，每个 Token 只走两套", "示意 Top-2 路由；不同 Token 可能选不同 Expert")
box(d, (65, 365, 295, 485), "Token A", BLUE, "#eaf2ff")
box(d, (390, 365, 620, 485), "Router", ORANGE, "#fff2e7")
arrow(d, (295, 425), (390, 425), BLUE)
for i in range(8):
    x = 735 + (i % 4)*195
    y = 245 + (i // 4)*250
    selected = i in (1, 5)
    box(d, (x, y, x+155, y+110), f"E{i+1}", TEAL if selected else "#bac6d1",
        "#e9f7f3" if selected else WHITE, 30)
arrow(d, (620, 410), (930, 300), TEAL)
arrow(d, (620, 440), (930, 550), TEAL)
box(d, (1280, 375, 1560, 485), "合并为一个向量", ORANGE, "#fff2e7", 24)
arrow(d, (1005, 355), (1280, 415), TEAL)
d.line([(1005, 495), (1150, 410)], fill=TEAL, width=6)
arrow(d, (1150, 410), (1280, 440), TEAL)
label(d, (710, 130), "淡灰：本次未激活，参数仍需存放", GRAY, 27)
im.save(OUT / "13-moe-token-route.png")


# 5: prefill versus decode, with cache timing clear.
im, d = canvas("KV Cache：旧位置不重算，新 Query 仍要读历史", "图只画一层；每层都有自己的缓存")
box(d, (90, 215, 420, 350), "Prefill：A B C", BLUE, "#eaf2ff")
arrow(d, (420, 280), (610, 280))
box(d, (610, 215, 1100, 350), "缓存 K/V：A B C", TEAL, "#e9f7f3")
box(d, (90, 505, 420, 640), "下一轮输入：D", BLUE, "#eaf2ff")
arrow(d, (420, 570), (610, 570))
box(d, (610, 505, 1100, 640), "追加缓存：D 的 K/V", TEAL, "#e9f7f3")
arrow(d, (1100, 285), (1255, 520), TEAL)
arrow(d, (1100, 570), (1255, 570), TEAL)
box(d, (1255, 480, 1520, 660), "D 的 Q\n读 A～D", ORANGE, "#fff2e7", 27)
label(d, (115, 752), "刚选出 D 时，D 的 K/V 还没有写入；下一轮输入 D 才追加。", NAVY, 28)
im.save(OUT / "14-kv-cache-timeline.png")


# 6: the storage arithmetic behind head sharing, not an end-to-end speed chart.
im, d = canvas("同样 32 个 Query 头，历史 KV 占用可以不同", "一层、一条 4096 Token 序列、每头 128 维、每元素 2 字节；只比较 KV Cache")
rows = [("MHA：32 组 KV", 64, BLUE), ("GQA：8 组 KV", 16, TEAL),
        ("MQA：1 组 KV", 2, ORANGE)]
for idx, (name, mib, color) in enumerate(rows):
    y = 275 + idx*170
    label(d, (95, y+19), name, NAVY, 29)
    d.rounded_rectangle((450, y, 450+int(mib/64*930), y+82), radius=15, fill=color)
    label(d, (1410, y+18), f"{mib} MiB", NAVY, 30)
box(d, (105, 790, 1470, 855), "只说明每历史位置存储更窄；模型整体速度与显存需另算", ORANGE, "#fff2e7", 25)
im.save(OUT / "15-kv-head-storage-bars.png")


# 7: same relative displacement after moving the whole segment.
im, d = canvas("整段后移 100 位：哪些位置线索保持相同？", "两个词对的内容相同，位置分别是 (10, 8) 与 (110, 108)，相对位移都为 2")
for y, a, b in [(280, "第 10 位：当前词", "第 8 位：历史词"),
                (465, "第 110 位：当前词", "第 108 位：历史词")]:
    box(d, (95, y, 475, y+95), a, BLUE, "#eaf2ff", 25)
    box(d, (790, y, 1170, y+95), b, TEAL, "#e9f7f3", 25)
    arrow(d, (475, y+48), (790, y+48), ORANGE)
    label(d, (525, y+5), "相隔 2 位", ORANGE, 28)
box(d, (90, 685, 570, 810), "绝对位置：编号改变", BLUE, WHITE, 25)
box(d, (625, 685, 1500, 810), "相对 Bias / RoPE / ALiBi：位移线索保持相同", TEAL, "#e9f7f3", 24)
im.save(OUT / "16-position-shift-example.png")
