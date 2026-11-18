"""Render the original technical film at 1280 x 720 from the timed voice track."""

import argparse, json, math, subprocess
from pathlib import Path
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/video"
SCENES = json.loads((ROOT / "video/script.json").read_text())
PAPER = "#f5f1e7"
INK = "#222923"
MUTED = "#555c52"
RULE = "#c2c3b4"
RED = "#b53c24"
FADE = "#e8e4d7"
LIGHT = "#fffdf7"
W, H = 1280, 720
FPS = 24


@lru_cache(maxsize=40)
def font(size, style="regular"):
    name = {
        "regular": "lato/Lato-Regular",
        "bold": "lato/Lato-Heavy",
        "mono": "dejavu/DejaVuSansMono",
        "serif": "dejavu/DejaVuSerif",
    }[style]
    return ImageFont.truetype("/usr/share/fonts/truetype/" + name + ".ttf", size)


def words(d, x, y, s, size=26, color=INK, style="regular", anchor="lt"):
    d.text((x, y), s, fill=color, font=font(size, style), anchor=anchor)


def line(d, xy, c=RULE, w=2):
    d.line(xy, fill=c, width=w, joint="curve")


def box(d, xy, fill=LIGHT, edge=RULE, r=6):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=edge, width=2)


def pill(d, x, y, w, s, active=False):
    box(d, (x, y, x + w, y + 70), RED if active else LIGHT, RED if active else RULE)
    words(d, x + w / 2, y + 36, s, 29, LIGHT if active else INK, "mono", "mm")


def clamp(v):
    return max(0, min(1, v))


def ease(v):
    v = clamp(v)
    return v * v * (3 - 2 * v)


def arrow(d, a, b, c=INK):
    x, y = a
    u, v = b
    line(d, (x, y, u, v), c, 3)
    angle = math.atan2(v - y, u - x)
    d.polygon(
        [(u, v)]
        + [
            (u - 13 * math.cos(angle + da), v - 13 * math.sin(angle + da))
            for da in (-0.47, 0.47)
        ],
        fill=c,
    )


def dot(d, x, y, r, color):
    d.ellipse((x - r, y - r, x + r, y + r), fill=color)


CUP = Image.open(ROOT / "desktop/franok.png").convert("RGBA")


def chapter(i, p, without_footer=False):
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    words(d, 72, 37, "FRANOK", 24, INK, "bold")
    line(d, (190, 38, 190, 63))
    words(d, 210, 40, "MODEL NOTES", 17, MUTED, "mono")
    words(d, 1208, 40, f"{i + 1:02} / {len(SCENES):02}", 17, MUTED, "mono", "rt")
    line(d, (72, 84, 1208, 84))
    if i == 0:
        art = CUP.resize((366, 366), Image.Resampling.LANCZOS)
        im.paste(art, (810, 163), art)
        words(d, 72, 153, "An English–Turkish language model", 27, MUTED)
        words(d, 66, 225, "Franok,", 100, INK, "serif")
        words(d, 66, 340, "from scratch.", 100, INK, "serif")
        words(d, 74, 529, "Yunus Emre Vurgun", 31, INK, "bold")
        words(d, 75, 579, "franok.yunusemrevurgun.com", 23, RED, "mono")
    else:
        words(d, 70, 119, SCENES[i]["title"], 56, INK, "serif")
        if i == 1:
            rows = [
                (
                    "TR",
                    "Kahve hazır.",
                    ["K", "ah", "ve", " hazır", "."],
                    [49, 368, 1057, 2059, 20],
                ),
                (
                    "EN",
                    "The coffee is hot.",
                    ["The", " coffee", " is", " hot", "."],
                    [598, 7188, 346, 6180, 20],
                ),
            ]
            for row, (lang, raw, toks, ids) in enumerate(rows):
                y = 216 + row * 185
                words(d, 74, y, lang, 19, RED, "mono")
                words(d, 143, y - 6, raw, 37, INK, "bold")
                for j, (piece, n) in enumerate(zip(toks, ids)):
                    x = 146 + j * 187
                    pill(d, x, y + 48, 162, piece, (row * 5 + j) == min(9, int(p * 11)))
                    words(d, x + 81, y + 136, "ID " + str(n), 19, MUTED, "mono", "mt")
        elif i == 2:
            parts = [
                ("Token vector", "512 coordinates"),
                ("Causal attention", "reads the prefix"),
                ("One FF bank", "updates this slot"),
                ("Next layer", "residual stream"),
            ]
            for j, (a, b) in enumerate(parts):
                x = 75 + 294 * j
                y = 289
                if j:
                    arrow(d, (x - 40, y + 58), (x - 10, y + 58), RULE)
                on = min(3, int(p * 4.2)) == j
                box(
                    d,
                    (x, y, x + 247, y + 120),
                    INK if on else LIGHT,
                    INK if on else RULE,
                )
                words(d, x + 17, y + 20, a, 28, LIGHT if on else INK, "bold")
                words(d, x + 17, y + 70, b, 22, PAPER if on else MUTED)
            words(d, 73, 457, "Layers", 21, MUTED, "mono")
            for j in range(12):
                x = 180 + j * 79
                box(
                    d,
                    (x, 454, x + 61, 494),
                    RED if j <= int(p * 12) else FADE,
                    RED if j <= int(p * 12) else RULE,
                    2,
                )
                words(
                    d,
                    x + 30,
                    474,
                    str(j + 1),
                    19,
                    LIGHT if j <= int(p * 12) else MUTED,
                    "mono",
                    "mm",
                )
            words(
                d,
                74,
                556,
                "Eight query heads share two key/value heads in each attention layer.",
                26,
                MUTED,
            )
        elif i == 3:
            xs = [122 + j * 103 for j in range(8)]
            words(d, 76, 207, "TOKEN SLOT", 17, MUTED, "mono")
            words(d, 76, 524, "FEED-FORWARD BANK", 17, MUTED, "mono")
            for sel in (False, True):
                for j, x in enumerate(xs):
                    if (j == 3) != sel:
                        continue
                    dest = xs[(j + 6) % 8]
                    path = []
                    for k in range(66):
                        t = k / 65
                        curve = ease(t)
                        path.append((x + (dest - x) * curve, 292 + 161 * t))
                    line(d, path, RED if sel else RULE, 5 if sel else 2)
                    if sel:
                        px, py = path[int(65 * ease((p - 0.18) / 0.58))]
                        dot(d, px, py, 9, RED)
            for j, x in enumerate(xs):
                pill(d, x - 32, 226, 64, str(j), j == 3)
                pill(d, x - 32, 450, 64, str(j), j == 1)
            line(d, (957, 222, 957, 540))
            words(d, 990, 230, "ANCHOR", 17, MUTED, "mono")
            words(d, 990, 264, "17", 48, INK, "bold")
            words(d, 990, 337, "LAYER", 17, MUTED, "mono")
            words(d, 990, 371, "2", 48, INK, "bold")
            words(d, 990, 446, "ROTATION", 17, MUTED, "mono")
            words(d, 990, 480, "+6", 48, RED, "bold")
            words(d, 75, 588, "(3 + 6) mod 8 = 1", 29, RED, "mono")
        elif i == 4:
            words(d, 74, 218, "Loss = −log p(correct piece)", 44, INK, "serif")
            words(d, 75, 300, "ILLUSTRATIVE PROBABILITIES", 17, MUTED, "mono")
            for j, (desc, v, val) in enumerate(
                [("Before", 0.10, "2.30"), ("After", 0.50, "0.69")]
            ):
                y = 366 + 108 * j
                words(d, 74, y + 8, desc, 27, INK, "bold")
                x = 230
                bw = 588
                box(d, (x, y, x + bw, y + 55), FADE, RULE, 3)
                drawv = v * (0.20 + 0.80 * ease(p * 3 - j * 0.3))
                d.rectangle((x + 2, y + 2, x + max(3, drawv * bw), y + 53), fill=RED)
                words(d, 853, y + 12, f"p = {v:.2f}", 25, INK, "mono")
                words(d, 1024, y + 12, "L = " + val, 25, RED, "mono")
            words(
                d,
                74,
                586,
                "Increasing probability on the observed next piece reduces its loss.",
                26,
                MUTED,
            )
        elif i == 5:
            rows = [
                (
                    "01",
                    "Supervised fine-tuning",
                    "Prompt gives context. Assistant tokens contribute to loss.",
                ),
                (
                    "02",
                    "Low-rank updates",
                    "W′ = W + (α/r) BA. Small matrices adapt the parent.",
                ),
                (
                    "03",
                    "On-policy reinforcement",
                    "Six sampled replies share a prompt. Score each reply.",
                ),
            ]
            for j, (n, name, sub) in enumerate(rows):
                x = 74
                y = 217 + j * 124
                active = min(2, int(p * 3.2)) == j
                box(
                    d,
                    (x, y, 1206, y + 105),
                    INK if active else LIGHT,
                    INK if active else RULE,
                )
                words(d, x + 24, y + 24, n, 23, "#efa488" if active else RED, "mono")
                words(d, x + 96, y + 16, name, 32, LIGHT if active else INK, "bold")
                words(d, x + 97, y + 64, sub, 23, PAPER if active else MUTED)
        elif i == 6:
            for j, (kind, mb, caption) in enumerate(
                [
                    ("FP16", 628.1, "PyTorch reference"),
                    ("Q8", 353.4, "Ubuntu desktop"),
                    ("Q4", 196.3, "Web browser"),
                ]
            ):
                y = 224 + j * 108
                words(d, 75, y + 6, kind, 33, INK, "bold")
                width = 663 * mb / 628.1
                box(
                    d,
                    (219, y, 219 + width, y + 49),
                    [RULE, INK, RED][j],
                    [RULE, INK, RED][j],
                    2,
                )
                words(d, 924, y + 2, f"{mb:.1f} MB", 27, INK, "mono")
                words(d, 220, y + 60, caption, 22, MUTED)
            words(d, 74, 583, "franok.yunusemrevurgun.com", 34, RED, "mono")
    if not without_footer:
        line(d, (72, 659, 1208, 659))
        line(d, (72, 659, 72 + 1136 * (i + p) / len(SCENES), 659), RED, 3)
        words(d, 72, 677, "Yunus Emre Vurgun", 16, MUTED)
        words(d, 1208, 677, "EN / TR     •     OPEN MODEL", 15, MUTED, "mono", "rt")
    return im


def main():
    t = json.loads((OUT / "timing.json").read_text())
    (OUT / "frames").mkdir(exist_ok=True)
    for i in range(len(t)):
        chapter(i, 0.52).save(OUT / "frames" / f"{i:02}.png")
    chapter(0, 0.4, True).save(OUT / "franok-poster.jpg", quality=91)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-i",
            str(OUT / "voice.wav"),
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(OUT / "franok-from-scratch.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    i = 0
    for k in range(math.ceil(t[-1]["end"] * FPS)):
        sec = k / FPS
        if i + 1 < len(t) and sec >= t[i]["end"]:
            i += 1
        p = clamp((sec - t[i]["start"]) / (t[i]["end"] - t[i]["start"]))
        ff.stdin.write(chapter(i, p).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("Render failed")


if __name__ == "__main__":
    main()
