"""Build a readable release report from the recorded training and inference results."""

import json, shutil
from pathlib import Path

D = Path("docs/evaluation/v2")


def read(file):
    return json.loads(Path(file).read_text())


def conversation(r, lang):
    checks = [
        c["passed"]
        for item in r["conversations"]
        if item["lang"] == lang
        for c in item["checks"]
        if c["passed"] is not None
    ]
    return f"{sum(checks)}/{len(checks)}"


def main():
    run = Path("work/v2/run")
    for name in (
        "metrics.jsonl",
        "progress.json",
        "finished.json",
        "rl.jsonl",
        "rl-finished.json",
    ):
        shutil.copyfile(run / name, D / name)
    p = read(run / "progress.json")
    rl = read(run / "rl-finished.json")
    allocation = read(D / "allocation.json")
    proof = read("work/v2/fp32/reconstruction.json")
    adapter_check = read(D / "adapter-reconstruction.json")
    shutil.copyfile("work/v2/fp32/reconstruction.json", D / "fp32-reconstruction.json")
    native = read(D / "franok-belebele.json")["summary"]
    short = read(D / "franok-short.json")["scores"]
    full_read = read(D / "fp32-belebele.json")
    full_short = read(D / "fp32-short.json")
    native_choices = {
        ("reading", r["language"], r["link"], r["question"]): r["pred"] - 1
        for r in full_read["items"]
    }
    native_choices.update(
        {
            ("short", r["lang"], i): 0 if r["win"] else 1
            for i, r in enumerate(full_short["rows"])
        }
    )
    p_read = read("docs/evaluation/franok-belebele.json")["summary"]
    p_short = read("docs/evaluation/franok-dialogue.json")["scores"]
    b_read = read("docs/evaluation/smollm2-belebele.json")["summary"]
    b_short = read("docs/evaluation/smollm2-360m.json")["scores"]
    q = {bits: read(D / f"runtime-{bits}.json") for bits in ("q4", "q8")}
    report = [
        "# Extended Colab release",
        "",
        "Franok has 314,061,312 parameters and a 1,024-token neural context. This report uses the same 48 aligned reading questions per language as the first release, plus newly recorded conversations, controlled long-context inputs and numerical comparisons against the unrounded final FP32 merge.",
        "",
        "## Training on one NVIDIA L4",
        "",
        "| Stage | Optimizer steps | Input positions | Supervised or next-token targets | Elapsed |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("lm", "Continued bilingual pretraining"),
        ("sft", "Full supervised tuning"),
        ("dialogue", "Dialogue LoRA"),
        ("evidence", "Contrastive and supervised LoRA"),
    ]:
        st = p["phases"][key]
        report.append(
            f"| {label} | {st['steps']:,} | {st['positions']:,} | {st['targets']:,} | {st['elapsed_seconds'] / 60:.1f} min |"
        )
    report += [
        "",
        f"The extension used {p['positions']:,} input positions and {p['targets']:,} unmasked targets in the stages above. The original run contributed another 53,859,151 next-token targets. Positions can repeat across draws. The separate reinforcement stage sampled {rl['rollouts']:,} replies containing {rl['tokens']:,} generated tokens and made {rl['optimizer_updates']:,} parameter updates in {rl['elapsed'] / 60:.1f} minutes. Its selected adapter scale was {rl['scale']}.",
        "",
        f"The additional L4 allocation lasted {allocation['elapsed_seconds'] / 60:.1f} minutes, including setup, evaluation on the accelerator and transfer. Colab confirmed termination at {allocation['termination_confirmed_utc']}. [Allocation record](allocation.json)",
        "",
        "## Reading and short tasks",
        "",
        "| Model | Reading EN / 48 | Reading TR / 48 | Short EN / 16 | Short TR / 16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, a, b in [
        ("SmolLM2-360M-Instruct", b_read, b_short),
        ("Franok v0.1 native", p_read, p_short),
        ("Franok v0.2 native FP16", native, short),
        ("Franok v0.2 native FP32", full_read["summary"], full_short["scores"]),
    ]:
        report.append(f"| {name} | {a['en']} | {a['tr']} | {b['en']} | {b['tr']} |")
    for bits, r in q.items():
        vals = [
            r["choices"]["summary"][k]["correct"]
            for k in ["reading-en", "reading-tr", "short-en", "short-tr"]
        ]
        report.append(
            "| Franok " + bits.upper() + " WASM | " + " | ".join(map(str, vals)) + " |"
        )
    report += [
        "",
        "Reading uses the mean log probability per answer character across four alternatives. The short set compares two continuations by mean token log probability. All Franok choice tests use the same short user-only template to preserve the comparison to v0.1. The peer uses its native chat template. The paired reading sample, question IDs, scorer and comparison revision are described in the [original release report](../README.md). These are small tests.",
        "",
        "## Actual conversations",
        "",
        "Ten scripted sessions have independent English and Turkish prompts. Each response becomes real history for the next user turn. Twelve turns have predeclared keyword or exact-match checks. The remaining turns show free-form behavior without an automatic grade. Identity questions use each model’s own name. Keyword checks can pass an incoherent answer or miss a valid paraphrase. Read the full exchanges before drawing conclusions.",
        "",
        "| Model and decoder | English checks | Turkish checks | Transcript |",
        "|---|---:|---:|---|",
    ]
    for name, file in [
        ("Franok v0.1 Q4", "parent-conversations.json"),
        ("SmolLM2-360M-Instruct", "smollm2-conversations.json"),
        ("Franok v0.2 native FP16", "franok-conversations.json"),
        ("Franok v0.2 native FP32", "fp32-conversations.json"),
        ("Franok v0.2 Q4", "runtime-q4.json"),
        ("Franok v0.2 Q8", "runtime-q8.json"),
    ]:
        v = read(D / file)
        report.append(
            f"| {name} | {conversation(v, 'en')} | {conversation(v, 'tr')} | [Raw replies]({file}) |"
        )
    report += [
        "",
        "### What changed during post-training",
        "",
        "These intermediate checkpoints answered the same prompts with the same sampler and seed. The six scored turns per language cover only a few dialogue behaviors. Every row links the actual replies, including unscored turns.",
        "",
        "| Native checkpoint | English checks / 6 | Turkish checks / 6 | Short EN / 16 | Short TR / 16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, chat_file, short_file in [
        ("After extended pretraining", "continued-chat.json", "continued-short.json"),
        (
            "After full supervised tuning",
            "supervised-chat.json",
            "supervised-short.json",
        ),
        ("After dialogue and evidence LoRAs", "pre-rl-chat.json", "pre-rl-short.json"),
        (
            "After reinforcement learning",
            "franok-conversations.json",
            "franok-short.json",
        ),
    ]:
        chat = read(D / chat_file)
        choices = read(D / short_file)["scores"]
        report.append(
            f"| [{name}]({chat_file}) | {conversation(chat, 'en')} | {conversation(chat, 'tr')} | {choices['en']} | {choices['tr']} |"
        )
    report += [
        "",
        "The conversational sampler uses temperature 0.58, top-p 0.89, top-k 48, a repetition penalty of 0.55 on recent tokens and a blocked fourth-order repetition. Its random generator resets to the same seed at the start of every session. Franok receives the application system prompt. The peer receives parallel language instructions under its own name. These runs bypass the local calculator and language notices.",
        "",
        "## Precision on the actual runtime",
        "",
        "| Format | Held-out tokens | Native FP32 NLL | WASM NLL | Top-token agreement | Mean logit cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summaries = {}
    for bits, r in q.items():
        slices = r["numeric"]
        n = sum(s["tokens"] for s in slices.values())
        avg = lambda k: sum(s[k] * s["tokens"] for s in slices.values()) / n
        summaries[bits] = {
            k: avg(k) for k in ["native_nll", "wasm_nll", "agreement", "cosine"]
        }
        report.append(
            f"| {bits.upper()} | {n:,} | {avg('native_nll'):.5f} | {avg('wasm_nll'):.5f} | {avg('agreement'):.1%} | {avg('cosine'):.5f} |"
        )
    report += [
        "",
        "| Paired choices | Same reading answer as native FP32 / 96 | Same short answer as native FP32 / 32 |",
        "|---|---:|---:|",
    ]
    for bits, r in q.items():
        matches = {"short": 0, "reading": 0}
        for item in r["choices"]["items"]:
            indices = item["key"] if isinstance(item["key"], list) else [item["key"]]
            key = (item["suite"], item["lang"], *indices)
            matches[item["suite"]] += item["pred"] == native_choices[key]
        summaries[bits]["same_reading_choices"] = matches["reading"]
        summaries[bits]["same_short_choices"] = matches["short"]
        report.append(f"| {bits.upper()} | {matches['reading']} | {matches['short']} |")
    report += [
        "",
        f"The reference reconstructs the unrounded final update from the pre-RL parent and released FP32 adapter. Rounding that reconstruction to FP16 differed from the recorded Colab FP16 checkpoint at {proof['fp16_different_elements']:,} of 314,061,312 elements (maximum absolute difference {proof['fp16_max_absolute_difference']:.8g}). Validation uses 16 seeded 129-token fragments of held-out English and Turkish text and chat, with every next position scored. Generated answers can diverge even when their next-token probability distributions are close.",
        "",
        f"Rebuilding from the published supervised base and all three adapters with the CPU merge tool differed from the Colab checkpoint at {adapter_check['different_after_fp16_rounding']:,} of {adapter_check['parameters']:,} FP16 values. The largest absolute difference was {adapter_check['max_absolute_difference']:.8g}. [Merge comparison](adapter-reconstruction.json)",
        "",
        "## Neural recall over longer conversations",
        "",
        "An early user message assigns a word to a folder. Unrelated turns intervene before the model chooses one of four words. Correct words are balanced across eight trials per language per depth. Both models receive the same dialogue content at each depth. The bounds below use Franok tokenization, so actual token counts for the peer differ. No session search or attachment participates.",
        "",
        "| Model | Budget in Franok tokens | English / 8 | Turkish / 8 |",
        "|---|---:|---:|---:|",
    ]
    for name, file in [
        ("Franok v0.2", "franok-long-context.json"),
        ("SmolLM2-360M-Instruct", "smollm2-long-context.json"),
    ]:
        rows = read(D / file)
        for depth in (128, 384, 960):
            scores = [
                sum(
                    x["correct"]
                    for x in rows
                    if x["lang"] == lang and x["budget"] == depth
                )
                for lang in ["en", "tr"]
            ]
            report.append(f"| {name} | {depth} | {scores[0]} | {scores[1]} |")
    report += [
        "",
        "The separate session-library test attaches the source from a 1,000-turn transcript. It checks whether the selected text actually reaches the model’s prompt. It makes no claim that the model can reliably recall or interpret that text.",
        "",
        "## Reproduction",
        "",
        "Prepare the pinned held-out data with `python -m training.data`. Use `tools.eval_inputs` to regenerate the shared choice set. `tools.numeric` generates native references. Run `node tools/measure_runtime.mjs EXPORT q4 REFERENCE OUT.json` and repeat with `q8`. The scripts `tools.reading`, `tools.probe`, `tools.compare_chat` and `tools.long_context` take checkpoint paths and output paths. The peer uses transformers 4.57.6. The [fixed conversation scripts](../../../tests/conversations.json) and all individual numerical and generation results are available alongside this report.",
    ]
    (D / "README.md").write_text("\n".join(report) + "\n")
    (D / "summary.json").write_text(
        json.dumps(
            dict(
                positions=p["positions"],
                targets=p["targets"],
                rl_updates=rl["optimizer_updates"],
                rl_rollouts=rl["rollouts"],
                reading=native,
                short=short,
                quantization=summaries,
            ),
            indent=2,
        )
        + "\n"
    )
    card = Path("docs/MODEL_CARD.md")
    before, rest = card.read_text().split("<!-- measured -->")
    _, after = rest.split("<!-- end measured -->")
    native_chat = read(D / "fp32-conversations.json")
    measured = [
        f"Extended pretraining and supervised stages consumed {p['targets']:,} target tokens. Reinforcement learning sampled {rl['rollouts']:,} replies and made {rl['optimizer_updates']:,} optimizer updates.",
        "",
        "| Fixed test | Franok EN | Franok TR | SmolLM2-360M-Instruct EN | SmolLM2-360M-Instruct TR |",
        "|---|---:|---:|---:|---:|",
        f"| Paired Belebele reading | {native['en']}/48 | {native['tr']}/48 | {b_read['en']}/48 | {b_read['tr']}/48 |",
        f"| Short two-choice tasks | {short['en']}/16 | {short['tr']}/16 | {b_short['en']}/16 | {b_short['tr']}/16 |",
        "",
        f"The native FP32 model passed {conversation(native_chat, 'en')} English and {conversation(native_chat, 'tr')} Turkish checks in direct multi-turn generation. Its answers frequently lose the topic. The broader conversational and reading results fall below the comparison model on these tests.",
        "",
    ]
    for bits, qs in summaries.items():
        agreed = qs["same_reading_choices"] + qs["same_short_choices"]
        measured.append(
            f"{bits.upper()} chose the same answer as FP32 on {agreed}/128 paired questions. Across 2,048 bilingual next-token predictions its top token matched FP32 on {qs['agreement']:.1%}, and mean negative log likelihood changed by {qs['wasm_nll'] - qs['native_nll']:+.5f}."
        )
        measured.append("")
    card.write_text(
        before
        + "<!-- measured -->\n"
        + "\n".join(measured)
        + "<!-- end measured -->"
        + after
    )


if __name__ == "__main__":
    main()
