"""Generate the three 16:9 PNG diagrams for the residual and norm chapter."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).parent / "images" / "transformer-evolution"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1600, 900
NAVY, BLUE, TEAL, ORANGE = "#152841", "#2878ce", "#139b91", "#e88b3d"
PALE, WHITE, GRAY = "#f6f9fc", "#ffffff", "#52657a"
FONT = "C:/Windows/Fonts/msyh.ttc"


def ft(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def base(title, subtitle):
    im = Image.new("RGB", (W, H), PALE)
    d = ImageDraw.Draw(im)
    d.text((68, 46), title, font=ft(43, True), fill=NAVY)
    d.text((70, 111), subtitle, font=ft(25), fill=GRAY)
    d.line((70, 167, 1530, 167), fill="#dce6ef", width=3)
    return im, d


def text(d, pos, value, color=NAVY, size=29, bold=False):
    d.text(pos, value, font=ft(size, bold), fill=color)


def card(d, xy, value, edge=BLUE, fill=WHITE, size=32):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle((x0+5, y0+7, x1+5, y1+7), radius=20,
                        fill="#e7edf3")
    d.rounded_rectangle(xy, radius=20, fill=fill, outline=edge, width=4)
    b = d.textbbox((0, 0), value, font=ft(size, True))
    tw, th = b[2]-b[0], b[3]-b[1]
    d.text((x0+(x1-x0-tw)/2, y0+(y1-y0-th)/2-4), value,
           font=ft(size, True), fill=NAVY)


def arr(d, start, end, color=TEAL, width=6):
    d.line((start, end), fill=color, width=width)
    a = math.atan2(end[1]-start[1], end[0]-start[0])
    length = 17
    p = (end[0]-length*math.cos(a-.55), end[1]-length*math.sin(a-.55))
    q = (end[0]-length*math.cos(a+.55), end[1]-length*math.sin(a+.55))
    d.polygon((end, p, q), fill=color)


# 17: one plain layer versus a residual layer. The plus node is a sum, never a switch.
im, d = base("残差：让子层学习增量，保留输入的直通路", "两条路同时参与前向计算；图只画一个 Attention 或 FFN 子层")
text(d, (100, 230), "没有残差", NAVY, 31, True)
card(d, (110, 355, 330, 485), "输入 x", BLUE, "#eaf3ff")
card(d, (600, 355, 850, 485), "子层 F", BLUE)
card(d, (1160, 355, 1450, 485), "输出 F(x)", BLUE, "#eaf3ff")
arr(d, (330, 420), (600, 420), BLUE)
arr(d, (850, 420), (1160, 420), BLUE)
text(d, (100, 600), "有残差", NAVY, 31, True)
card(d, (110, 680, 330, 805), "输入 x", BLUE, "#eaf3ff")
card(d, (595, 680, 820, 805), "子层 F", BLUE)
card(d, (1105, 680, 1380, 805), "相加 +", ORANGE, "#fff2e6")
arr(d, (330, 742), (595, 742), BLUE)
arr(d, (820, 742), (1105, 742), BLUE)
d.line([(330, 742), (410, 742), (410, 605), (990, 605), (990, 700)],
       fill=TEAL, width=7)
arr(d, (990, 700), (1105, 700), TEAL)
text(d, (575, 558), "输入 x 直接保留", TEAL, 28)
arr(d, (1380, 742), (1500, 742), ORANGE)
text(d, (1390, 690), "x + F(x)", ORANGE, 25)
im.save(OUT / "17-residual-why.png")


# 18: repeated additive trunk and the reverse gradient path.
im, d = base("深层残差：信息有主线，子层不断贡献改动", "正向自左向右；反向梯度沿同一主线自右向左传播")
y = 490
for i, x in enumerate([340, 700, 1060]):
    card(d, (x-80, y-50, x+80, y+50), "相加 +", ORANGE, "#fff2e6", 28)
    card(d, (x-95, 260, x+95, 365), f"子层 F{i+1}", BLUE, WHITE, 27)
    arr(d, (x-170, y), (x-80, y), TEAL)
    d.line([(x-170, y), (x-170, 310)], fill=BLUE, width=6)
    arr(d, (x-170, 310), (x-95, 310), BLUE)
    arr(d, (x, 365), (x, y-50), BLUE)
    arr(d, (x+80, y), (x+190, y), TEAL)
text(d, (85, 455), "输入", NAVY, 29)
text(d, (1355, 455), "输出", NAVY, 29)
arr(d, (1320, 700), (175, 700), ORANGE, 7)
text(d, (555, 724), "反向信号：存在一条不必穿过各 F 的路径", ORANGE, 27)
card(d, (140, 795, 1460, 855), "直通路径有助于优化，但不保证梯度永远稳定", TEAL, "#e9f7f3", 24)
im.save(OUT / "18-residual-gradient-path.png")


# 19: the statistics are across channels of one token, not across the token axis.
im, d = base("LayerNorm：统计一枚 Token 的隐藏通道", "序列里的每枚 Token 独立计算；这里用三个通道做教学示意")
for row, (name, values) in enumerate([("Token A", [2, 4, 6]),
                                       ("Token B", [1, 3, 5]),
                                       ("Token C", [4, 7, 2])]):
    y = 265 + row*145
    focus = row == 1
    text(d, (120, y+28), name, ORANGE if focus else GRAY, 30, focus)
    for col, v in enumerate(values):
        card(d, (410+col*165, y, 530+col*165, y+92), str(v),
             ORANGE if focus else BLUE, "#fff1e5" if focus else WHITE, 29)
    if focus:
        d.rounded_rectangle((390, y-16, 885, y+108), radius=24,
                            outline=ORANGE, width=3)
        arr(d, (885, y+46), (1050, y+46), ORANGE)
        card(d, (1050, y-10, 1490, y+105), "均值 = 3", ORANGE, "#fff1e5", 27)
text(d, (410, 742), "本例 Token B：[1, 3, 5] → 减均值 → [-2, 0, 2]", NAVY, 29)
text(d, (410, 796), "再按标准差缩放；不读取 Token A/C 的数值", GRAY, 26)
im.save(OUT / "19-normalization-one-token.png")


# 20: decoder query reads the completed source, while decoder self-attention stays causal.
im, d = base("Cross-Attention：正在写的目标，回头查已读完的源文", "Query 来自 Decoder；Key / Value 来自 Encoder 输出")
text(d, (115, 235), "源序列：已完整给定", NAVY, 30, True)
for i, value in enumerate(["猫", "追", "狗"]):
    card(d, (140+i*175, 315, 270+i*175, 425), value, BLUE, "#eaf3ff")
d.rounded_rectangle((110, 295, 650, 445), radius=22, outline="#a7bfd6", width=3)
card(d, (790, 310, 1130, 430), "由 Encoder 输出生成 K/V", TEAL, "#e9f7f3", 23)
arr(d, (650, 370), (790, 370), TEAL, 5)
text(d, (115, 585), "目标序列：正在生成", NAVY, 30, True)
for i, value in enumerate(["The", "cat", "…"]):
    card(d, (140+i*175, 655, 270+i*175, 765), value,
         ORANGE if i == 2 else BLUE, "#fff2e6" if i == 2 else WHITE, 27)
card(d, (790, 650, 1130, 770), "Decoder 当前 Q", ORANGE, "#fff2e6", 27)
arr(d, (620, 710), (790, 710), ORANGE)
card(d, (1250, 470, 1510, 610), "Cross-\nAttention", TEAL, WHITE, 25)
arr(d, (1130, 370), (1250, 520), TEAL)
arr(d, (1130, 710), (1250, 560), ORANGE)
im.save(OUT / "20-cross-attention-sources.png")


# 21: the precise scope of token mixing versus per-token feature processing.
im, d = base("Attention 交换位置间的信息；FFN 加工每个位置", "两种子层交替出现；FFN 使用同一套参数，但不直接读取其他 Token")
text(d, (100, 225), "Attention：跨位置取信息", NAVY, 30, True)
for i, value in enumerate(["苹果", "发布", "新品"]):
    y = 295+i*90
    card(d, (110, y, 295, y+72), value, BLUE, "#eaf3ff", 25)
    arr(d, (295, y+36), (625, 405), TEAL, 5)
card(d, (625, 350, 950, 460), "Attention（新品）", TEAL, "#e9f7f3", 25)
card(d, (1120, 350, 1490, 460), "新品的上下文表示", TEAL, WHITE, 24)
arr(d, (950, 405), (1120, 405), TEAL)
text(d, (100, 590), "FFN：逐位置加工特征", NAVY, 30, True)
for i, value in enumerate(["苹果", "发布", "新品"]):
    x = 105+i*495
    card(d, (x, 680, x+150, 780), value, BLUE, "#eaf3ff", 26)
    card(d, (x+205, 680, x+365, 780), "同一 FFN", ORANGE, "#fff2e6", 23)
    arr(d, (x+150, 730), (x+205, 730), BLUE)
im.save(OUT / "21-attention-vs-ffn.png")


# 22: repeated full forward passes without cache versus one-token decode with cache.
im, d = base("KV Cache：同样生成 D、E，重复计算发生在哪里？", "每行是一次模型前向；这里只比较历史位置是否重算")
text(d, (110, 225), "不缓存：每轮重跑前缀", NAVY, 30, True)
text(d, (880, 225), "使用 Cache：只算新增位置", NAVY, 30, True)
for idx, (left, right) in enumerate([("A  B  C", "A  B  C → 写入 Cache"),
                                      ("A  B  C  D", "D → 追加 Cache"),
                                      ("A  B  C  D  E", "E → 追加 Cache")]):
    y = 320+idx*155
    card(d, (100, y, 720, y+105), left, BLUE, "#eaf3ff", 30)
    card(d, (870, y, 1500, y+105), right, TEAL if idx else BLUE,
         "#e9f7f3" if idx else "#eaf3ff", 28)
card(d, (170, 805, 1420, 865), "省掉旧位置的重复前向；当前 Query 仍要读取可见历史 K/V", ORANGE, "#fff2e6", 25)
im.save(OUT / "22-cache-repeat-work.png")


# 23: a decoder block is a sequence of choices, not one combined trick.
im, d = base("把现代组件放回一个 Decoder Block", "按数据流读：位置进入 Attention；FFN 或 MoE 随后加工；Norm/残差包住子层")
items = [
    ("输入 Token", BLUE, "#eaf3ff"),
    ("Norm", BLUE, WHITE),
    ("Attention", TEAL, "#e9f7f3"),
    ("残差相加", ORANGE, "#fff2e6"),
    ("Norm", BLUE, WHITE),
    ("FFN / MoE", TEAL, "#e9f7f3"),
    ("残差相加", ORANGE, "#fff2e6"),
]
for i, (name, edge, fill) in enumerate(items):
    x = 70+i*220
    card(d, (x, 345, x+185, 480), name, edge, fill, 24)
    if i < len(items)-1:
        arr(d, (x+185, 412), (x+220, 412), TEAL, 5)
d.line([(255, 412), (265, 412), (265, 270), (825, 270), (825, 345)],
       fill=ORANGE, width=5)
arr(d, (825, 305), (825, 345), ORANGE, 5)
d.line([(915, 412), (930, 412), (930, 215), (1480, 215), (1480, 345)],
       fill=ORANGE, width=5)
arr(d, (1480, 305), (1480, 345), ORANGE, 5)
card(d, (335, 595, 795, 710), "RoPE：改变 Q/K 位置关系", ORANGE, "#fff2e6", 24)
arr(d, (565, 595), (565, 480), ORANGE)
card(d, (810, 595, 1515, 710), "MHA / GQA / MLA：决定历史信息的组织方式", BLUE, "#eaf3ff", 23)
arr(d, (1050, 595), (610, 480), BLUE)
text(d, (270, 788), "生成时：Attention 历史状态进入 KV Cache；MoE Router 选择本次 Expert。", GRAY, 26)
im.save(OUT / "23-decoder-component-map.png")
