"""Blind automatic judge for chat transcripts: rubric and sensibleness verdicts from an open model on vLLM."""

import argparse
import json
import random
from pathlib import Path

from .compare_chat import meets

LANGUAGES = dict(en="English", tr="Turkish")

RUBRIC = """You are a strict judge of a small chat assistant. The conversation below is in {language}. Decide whether the assistant's last reply satisfies the requirement.
Requirement: {rubric}
Answer FAIL if the reply is garbled, off-topic, self-contradictory, in the wrong language, ignores the user's latest message, or does not meet the requirement. A short correct reply is fine.
Answer with one word: PASS or FAIL.

{chat}"""

SENSIBLE = """You are a strict judge of a small chat assistant. The conversation below is in {language}. Decide whether the assistant's last reply is sensible: coherent, on the topic of the user's latest message, and written in {language}.
Answer FAIL if the reply is garbled, off-topic, self-contradictory, in the wrong language, or ignores the user's latest message. A short correct reply is fine.
Answer with one word: PASS or FAIL.

{chat}"""


def render(history, reply):
    lines = [f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}" for t in history]
    return "\n".join(lines + [f"Assistant (reply to judge): {reply}"])


def load_transcript(name, path, cases):
    data = json.loads(Path(path).read_text())
    by_id = {c["id"]: c for c in data["conversations"]}
    for case in cases:
        got = by_id.get(case["case"])
        if got is None:
            raise SystemExit(f"transcript {name} ({path}) has no case {case['case']!r}")
        users = [t["content"] for t in got["turns"] if t["role"] == "user"]
        if len(got["turns"]) != 2 * len(case["turns"]) or users != case["turns"]:
            raise SystemExit(f"transcript {name} case {case['case']!r} does not match the user turns in the cases file")
    return by_id


def build_items(cases, transcripts):
    items = []
    for case in cases:
        for system, by_id in transcripts.items():
            got = by_id[case["case"]]
            for j, user in enumerate(case["turns"]):
                criteria = case["expect"][j]
                if criteria is not None and got.get("checks"):
                    criteria = got["checks"][j]["criterion"]
                reply = got["turns"][2 * j + 1]["content"]
                items.append(
                    dict(
                        case=case["case"],
                        turn=j,
                        system=system,
                        lang=case["lang"],
                        user=user,
                        reply=reply,
                        rubric=case["rubric"][j],
                        history=got["turns"][: 2 * j + 1],
                        expect=None if criteria is None else meets(reply, criteria, case),
                    )
                )
    return items


def prompts_for(item):
    fields = dict(language=LANGUAGES[item["lang"]], chat=render(item["history"], item["reply"]))
    return RUBRIC.format(rubric=item["rubric"], **fields), SENSIBLE.format(**fields)


def judge(items, ask, seed=0):
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    pairs = [prompts_for(items[i]) for i in order]
    raw = {}
    for k, name in enumerate(("rubric", "sensible")):
        answers = ask([p[k] for p in pairs])
        if len(answers) != len(pairs):
            raise RuntimeError(f"ask returned {len(answers)} answers for {len(pairs)} prompts")
        for i, a in zip(order, answers):
            raw[i, name] = a
    out = []
    for i, item in enumerate(items):
        row = {k: v for k, v in item.items() if k != "history"}
        for name in ("rubric", "sensible"):
            row[f"{name}_raw"] = raw[i, name]
            row[name] = raw[i, name].strip().upper().startswith("PASS")
        out.append(row)
    return out


def tally(rows, key):
    rows = [r for r in rows if r[key] is not None]
    count = lambda rs: dict(passed=sum(bool(r[key]) for r in rs), of=len(rs))
    return dict(**count(rows), **{lang: count([r for r in rows if r["lang"] == lang]) for lang in LANGUAGES})


def totals(rows):
    systems = dict.fromkeys(r["system"] for r in rows)
    return {
        s: {f"{k}_passed": tally([r for r in rows if r["system"] == s], k) for k in ("rubric", "sensible", "expect")}
        for s in systems
    }


def table(summary):
    cell = lambda t: f"{t['passed']}/{t['of']} (en {t['en']['passed']}/{t['en']['of']}, tr {t['tr']['passed']}/{t['tr']['of']})"
    width = max(map(len, summary), default=6)
    lines = [f"{'system':<{width}}  {'rubric':<30}  {'sensible':<30}  expect"]
    for s, t in summary.items():
        lines.append(f"{s:<{width}}  {cell(t['rubric_passed']):<30}  {cell(t['sensible_passed']):<30}  {cell(t['expect_passed'])}")
    return "\n".join(lines)


def make_ask(args):
    from vllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        **({"quantization": args.quantization} if args.quantization else {}),
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=3)

    def ask(prompts):
        chats = [
            tok.apply_chat_template(
                [dict(role="user", content=p)], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for p in prompts
        ]
        return [o.outputs[0].text for o in llm.generate(chats, params)]

    return ask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--transcripts", nargs="+", required=True, metavar="NAME=PATH")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--quantization", default=None)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text())
    named = [t.partition("=") for t in args.transcripts]
    if any(not (n and sep and path) for n, sep, path in named) or len({n for n, _, _ in named}) != len(named):
        raise SystemExit("--transcripts takes distinct NAME=PATH pairs")
    transcripts = {n: load_transcript(n, path, cases) for n, _, path in named}
    rows = judge(build_items(cases, transcripts), make_ask(args), args.seed)
    summary = totals(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(dict(model=args.model, revision=args.revision, seed=args.seed, totals=summary, items=rows), ensure_ascii=False, indent=2) + "\n"
    )
    print(table(summary))


if __name__ == "__main__":
    main()
