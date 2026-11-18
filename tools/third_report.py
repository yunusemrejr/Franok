"""Assemble a release report from measured native, web and Ubuntu results."""

import argparse
import json
from pathlib import Path
import shutil

from tools.paired_results import compare, records


def read(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def passages(report, metric, lang=None):
    return sum(
        int(row[metric]) for row in report["items"] if not lang or row["lang"] == lang
    )


def agreement(native, runtime):
    def predictions(data):
        return {
            (r["lang"], json.dumps(r["key"])): r["prediction"] for r in data["items"]
        }

    a, b = predictions(native), predictions(runtime)
    if a.keys() != b.keys():
        raise ValueError("Incomplete quantized passage test")
    return sum(a[k] == b[k] for k in a)


def write_report(out, info, pretrain, post, paired):
    allocation = info["allocation"]
    reinforcement = info["rl"]
    lines = [
        "# Franok 0.3 results",
        "",
        "Both languages use the same 314,061,312-parameter decoder and tokenizer. "
        "The 16 two-turn conversations feed each model’s actual first answer back into its second prompt. "
        "The passage and natural-text tests use fixed bilingual inputs. [Test methods](README.md)",
        "",
        "## Training",
        "",
        "| Stage | Optimizer steps | Training targets | Selected adapter scale |",
        "|---|---:|---:|---:|",
        f"| Continued pretraining | {pretrain['steps']:,} | {pretrain['targets']:,} | — |",
    ]
    for name, stage in post["stages"].items():
        factor = stage["scale"] if stage["scale"] is not None else "—"
        lines.append(
            f"| {name} | {stage['steps']:,} | {stage['targets']:,} | {factor} |"
        )
    lines += [
        "",
        f"One Colab L4 was allocated for {allocation['elapsed_seconds'] / 60:.1f} minutes, including preparation and transfer. "
        f"The reinforcement stage sampled {reinforcement['rollouts']:,} replies and {reinforcement['tokens']:,} tokens. "
        f"It made {reinforcement['optimizer_updates']:,} updates and selected adapter scale {reinforcement['scale']}. "
        "Supervised target counts include answer positions used to compare preferred and rejected responses. "
        "A sampled position can recur during training. [Training receipts](post.jsonl) · [Reinforcement record](rl-finished.json) · [Allocation](allocation.json)",
        "",
        "## Direct model comparisons",
        "",
        "| Model | Conversation criteria / 32 | Belebele / 240 | XQuAD exact EN / 64 | XQuAD exact TR / 64 | English bits/byte ↓ | Turkish bits/byte ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "parent": "Franok 0.2",
        "peer": "SmolLM2-360M-Instruct",
        "lfm2": "LFM2-350M",
        "final": "Franok 0.3 FP32",
    }
    for r in info["native"]:
        name = r["model"]
        bp = r["bits_per_byte"]
        lines.append(
            f"| [{labels[name]}]({name}-open.json) | {r['natural_dialogue']} | {r['belebele']} | {r['xquad_en']} | {r['xquad_tr']} | {bp['en']:.3f} | {bp['tr']:.3f} |"
        )
    lo, hi = paired["belebele-vs-peer"]["percentile_95"]
    lines += [
        "",
        "Conversation criteria were graded by one model-assisted reviewer against the published rubric. "
        "Belebele ranks four supplied choices, while XQuAD generates an answer freely. "
        f"The paired 95% bootstrap interval for Franok 0.3 minus SmolLM2 on Belebele is {100 * lo:+.1f} to {100 * hi:+.1f} percentage points. "
        "Both translations of each passage stay together during resampling. [Rubric judgments](final-rubric.json) · [Paired differences](paired.json)",
        "",
        "## Browser and Ubuntu",
        "",
        "| Runtime | Belebele / 240 | Choices matching FP32 / 240 | Free XQuAD exact / 128 | Passage-reader exact / 128 | Top token matching FP32 | Change in target NLL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fmt, r in info["runtime"].items():
        lines.append(
            f"| [{fmt.upper()} WASM](final-{fmt}-quant.json) | {r['reading']} | {r['matching_choices']} | {r['free_exact']} | {r['reader_exact']} | {100 * r['top1']:.1f}% | {r['nll_difference']:+.5f} |"
        )
    s = info["reader"]
    lines += [
        "",
        f"The optional FP32 passage reader found {s['en']['exact']}/64 exact English answers and {s['tr']['exact']}/64 Turkish answers. "
        f"It returned a quotation on {s['en']['answered']}/64 English and {s['tr']['answered']}/64 Turkish questions. "
        "Abstentions count as misses on this answerable set. [Native reader output](final-reader.json) · [Q4 outputs](final-q4-reader.json) · [Q8 outputs](final-q8-reader.json)",
        "",
        f"The distribution check covers {info['runtime']['q4']['tokens']:,} target positions on 96 validation prefixes. "
        "A small average loss change can still alter a particular sampled sentence. The earlier 12 automatic dialogue checks also have full transcripts. "
        "[Q4 conversations](final-q4-conversations.json) · [Q8 conversations](final-q8-conversations.json)",
        "",
        "## Changing the system prompt",
        "",
        "Each language has 24 distinct held-out passage questions with two choices. Each question is asked twice at temperature zero. Only the system instruction changes. "
        "One version requires the correct letter and the other requires the exact answer text. Each question balances the same choices across the two requests, and the correct letter is balanced across questions.",
        "",
        "| Runtime | EN letter / 24 | EN words / 24 | EN both / 24 | TR letter / 24 | TR words / 24 | TR both / 24 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, file in (
        ("0.2 Q4", "parent-q4"),
        ("SmolLM2-360M-Instruct", "peer"),
        ("0.3 Q4", "final-q4"),
        ("0.3 Q8", "final-q8"),
    ):
        scores = read(out / (file + "-controls-score.json"))
        cells = [
            str(scores[lang][k])
            for lang in ("en", "tr")
            for k in ("letter", "words", "both")
        ]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Long sessions",
        "",
        "The native memory task asks the model which of four words was assigned to a folder. Sixteen cases per length cover eight labels in each language. These are forced choices with conversation history inside the model window.",
        "",
        "| Model | 128-position budget / 16 | 384-position budget / 16 | 960-position budget / 16 |",
        "|---|---:|---:|---:|",
    ]
    for name, depths in info["native_long"].items():
        lines.append(
            f"| {labels[name]} | {depths[128]} | {depths[384]} | {depths[960]} |"
        )
    lines += [
        "",
        "The workspace also searches earlier saved messages outside that window. Users can inspect and attach a source or let bilingual search select earlier context. These operations have their own retrieval measurements. [Native memory cases](final-long.json) · [Bilingual session search](meaning-search.json)",
    ]
    (out / "final.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path("work/v3"))
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation/v3"))
    args = parser.parse_args()
    root, out = args.experiment, args.out
    ev, run = root / "evaluation", root / "run"
    out.mkdir(exist_ok=True, parents=True)

    def evaluation(stem):
        return read(ev / (stem + ".json"))

    precision = read(root / "fp32/reconstruction.json")
    lm, post, rl = (
        read(run / f)
        for f in ("recovery.json", "post-finished.json", "rl-finished.json")
    )
    if (
        precision["sha256"] != rl["fp32_sha256"]
        or evaluation("final-belebele")["weights"] != precision["sha256"]
    ):
        raise ValueError("Training and evaluation identify different final weights")
    info = dict(
        parameters=314061312, active_parameters=49820160, sha256=precision["sha256"]
    )
    info["allocation"] = read(out / "allocation.json")
    info["pretraining_targets"] = lm["targets"]
    info["supervised_targets"] = sum(st["targets"] for st in post["stages"].values())
    paired = {}
    rows = []
    for model in ("parent", "peer", "lfm2", "final"):
        choices, generation = (
            evaluation(f"{model}-{s}") for s in ("belebele", "xquad")
        )
        chat = evaluation(model + "-rubric")
        fluency = evaluation(model + "-byte")["summary"]
        rows.append(
            dict(
                model=model,
                belebele=passages(choices, "correct"),
                natural_dialogue=chat["passed"],
                xquad=passages(generation, "exact"),
                xquad_en=passages(generation, "exact", "en"),
                xquad_tr=passages(generation, "exact", "tr"),
                bits_per_byte={
                    lang: fluency[lang]["bits_per_byte"] for lang in ("en", "tr")
                },
            )
        )
    for suite in ("belebele", "xquad", "rubric"):
        candidate = records(evaluation("final-" + suite))
        for baseline in ("parent", "peer", "lfm2"):
            paired[suite + "-vs-" + baseline] = compare(
                candidate, records(evaluation(baseline + "-" + suite))
            )
    fp_choices = evaluation("final-belebele")
    reader = evaluation("final-reader")
    info["native"] = rows
    info["reader"] = reader["summary"]
    info["runtime"] = {}
    for fmt in ("q4", "q8"):
        choices = evaluation(f"final-{fmt}-belebele")
        quant = evaluation(f"final-{fmt}-quant")
        span = evaluation(f"final-{fmt}-reader")
        transcripts = evaluation(f"final-{fmt}-conversations")
        stats = quant["metrics"]["all"]
        info["runtime"][fmt] = dict(
            sha256=quant["sha256"],
            reading=passages(choices, "correct"),
            matching_choices=agreement(fp_choices, choices),
            free_exact=passages(evaluation(f"final-{fmt}-xquad"), "exact"),
            reader_exact=passages(span, "exact"),
            conversation_checks=sum(
                check["passed"] is True
                for row in transcripts["conversations"]
                for check in row["checks"]
            ),
            controls=evaluation(f"final-{fmt}-controls-score"),
            top1=stats["top1"],
            nll_difference=stats["modelNll"] - stats["fullNll"],
            grouped_kl=stats["coarseKl"],
            tokens=stats["n"],
        )
    info["rl"] = {
        k: rl[k]
        for k in (
            "scale",
            "rollouts",
            "optimizer_updates",
            "tokens",
            "dialogue_reward_active",
        )
    }
    info["native_long"] = {}
    for name, location in (
        ("parent", out.parent / "v2/franok-long-context.json"),
        ("peer", out.parent / "v2/smollm2-long-context.json"),
        ("final", ev / "final-long.json"),
    ):
        trials = read(location)
        info["native_long"][name] = {
            depth: sum(int(r["correct"]) for r in trials if r["budget"] == depth)
            for depth in (128, 384, 960)
        }
    write_json(out / "summary.json", info)
    write_json(out / "paired.json", paired)
    reports = [
        f"{m}-{s}"
        for m in ("parent", "peer", "lfm2", "final")
        for s in ("rubric", "open", "belebele", "xquad", "byte")
    ]
    reports += [
        "continued-text",
        "final-text",
        "final-long",
        "final-short",
        "final-reader",
        "sft-open",
        "pre-rl-open",
    ]
    for fmt in ("q4", "q8"):
        reports += [
            f"final-{fmt}-{s}"
            for s in (
                "quant",
                "belebele",
                "xquad",
                "reader",
                "conversations",
                "transfer_conversations",
                "controls-score",
            )
        ]
    reports += ["parent-q4-controls-score", "peer-controls-score"]
    for name in reports:
        shutil.copyfile(ev / (name + ".json"), out / (name + ".json"))
    for name in (
        "pretrain.jsonl",
        "post.jsonl",
        "recovery.json",
        "sft.json",
        "dialogue.json",
        "evidence.json",
        "rl-finished.json",
        "rl.jsonl",
    ):
        shutil.copyfile(run / name, out / name)
    shutil.copyfile(root / "preparation.json", out / "data-preparation.json")
    shutil.copyfile(run / "reader/training.json", out / "reader-training.json")
    shutil.copyfile(
        run / "dialogue-reward/validation.json", out / "reward-validation.json"
    )
    shutil.copyfile(root / "fp32/reconstruction.json", out / "fp32-reconstruction.json")
    write_report(out, info, lm, post, paired)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
