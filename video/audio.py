"""Synthesize each caption separately for accurate bilingual timing."""

import argparse, json, subprocess, wave, os, math
from pathlib import Path
from contextlib import closing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/video"


def stamp(seconds):
    ms = round(seconds * 1000)
    return (
        f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000:03}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--piper", required=True, type=Path)
    p.add_argument("--voice", required=True, type=Path)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((ROOT / "video/script.json").read_text())
    rate = 0
    frames = 0
    sections = []
    caption = {k: ["WEBVTT", ""] for k in ("en", "tr")}
    with wave.open(str(OUT / "voice.wav"), "wb") as mix:
        for i, sc in enumerate(scenes):
            start = frames
            for j, line in enumerate(sc["en"]):
                tmp = OUT / f"voice-{i}-{j}.wav"
                subprocess.run(
                    [
                        str(args.piper),
                        "--model",
                        str(args.voice),
                        "--output_file",
                        str(tmp),
                        "--length_scale",
                        "0.98",
                        "--quiet",
                    ],
                    input=(line + "\n").encode(),
                    check=True,
                    env={**os.environ, "OMP_NUM_THREADS": "2"},
                )
                with closing(wave.open(str(tmp), "rb")) as w:
                    assert w.getnchannels() == 1 and w.getsampwidth() == 2
                    if not rate:
                        rate = w.getframerate()
                        mix.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                    assert rate == w.getframerate()
                    lead = int(rate * (0.18 if j == 0 else 0.10))
                    tail = int(rate * (0.48 if j == len(sc["en"]) - 1 else 0.06))
                    t0 = (frames + lead) / rate
                    n = w.getnframes()
                    t1 = t0 + n / rate
                    mix.writeframes(b"\0\0" * lead + w.readframes(n) + b"\0\0" * tail)
                    frames += lead + n + tail
                    for k in caption:
                        caption[k].extend(
                            [f"{stamp(t0)} --> {stamp(t1)}", sc[k][j], ""]
                        )
            sections.append({"scene": i, "start": start / rate, "end": frames / rate})
    (OUT / "timing.json").write_text(json.dumps(sections, indent=2))
    for k, lines in caption.items():
        (OUT / f"franok-{k}.vtt").write_text("\n".join(lines).rstrip() + "\n")
    print("Voice duration", frames / rate, flush=True)


if __name__ == "__main__":
    main()
