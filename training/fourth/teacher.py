"""Generate seeded conversations with an open instruct model on vLLM. Resumable and chunked."""

import argparse
import json
import time
from pathlib import Path

from .seeds import build_prompt, build_seeds

GRADE = """You check a short chat between a user and an assistant, written in {language}. Judge every assistant reply.
Answer BAD if any reply: says something factually wrong, unsafe or doubtful; states an invented detail about the user's own situation or about local facts (opening hours, prices, travel times) that it cannot know; contains a grammar mistake, a made-up word or unnatural phrasing; or ignores what the user asked or the answer format the user requested.
Otherwise answer GOOD.
Answer with one word: GOOD or BAD.

{chat}"""

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=40404)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--lang", choices=("en", "tr"))
    p.add_argument("--grade", action="store_true", help="after generating, grade each parsed conversation GOOD or BAD")
    p.add_argument("--max-model-len", type=int, default=1536)
    p.add_argument("--max-seqs", type=int, default=384)
    p.add_argument("--quantization", default=None)
    p.add_argument("--gpu-util", type=float, default=0.92)
    args = p.parse_args()
    from vllm import LLM, SamplingParams

    seeds = build_seeds(args.count, args.seed, args.lang)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["seed"]["id"] for line in args.out.read_text().splitlines() if line}
    todo = [s for s in seeds if s["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    llm = LLM(
        args.model,
        revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_seqs,
        enable_prefix_caching=True,
        **({"quantization": args.quantization} if args.quantization else {}),
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=args.temperature, top_p=0.95, top_k=50, max_tokens=700, seed=None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, produced = time.time(), 0
    for lo in range(0, len(todo), args.chunk):
        batch = todo[lo : lo + args.chunk]
        prompts = [
            tok.apply_chat_template(
                [dict(role="user", content=build_prompt(s))],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for s in batch
        ]
        outputs = llm.generate(prompts, params)
        with args.out.open("a") as f:
            for s, o in zip(batch, outputs):
                produced += len(o.outputs[0].token_ids)
                f.write(
                    json.dumps(
                        dict(seed=s, text=o.outputs[0].text, finish=o.outputs[0].finish_reason, model=args.model, revision=args.revision),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        elapsed = time.time() - started
        print(
            f"{lo + len(batch)}/{len(todo)} elapsed={elapsed:.0f}s output_tokens={produced} "
            f"tokens_per_s={produced / elapsed:.0f} conv_per_s={(lo + len(batch)) / elapsed:.2f}",
            flush=True,
        )
    if args.grade:
        grade(llm, tok, args.out)


def grade(llm, tok, out):
    from vllm import SamplingParams

    from .filter import parse

    path = out.with_suffix(".grades.jsonl")
    seen = {json.loads(l)["id"] for l in path.read_text().splitlines() if l} if path.exists() else set()
    records = [json.loads(l) for l in out.read_text().splitlines() if l]
    todo = []
    for r in records:
        conv, reason = parse(r)
        if reason is None and r["seed"]["id"] not in seen:
            todo.append((r["seed"], conv))
    prompts = [
        tok.apply_chat_template(
            [dict(role="user", content=GRADE.format(
                language="Turkish" if s["lang"] == "tr" else "English",
                chat="\n".join(f"User: {t['user']}\nAssistant: {t['assistant']}" for t in conv)))],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for s, conv in todo
    ]
    started = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=3))
    with path.open("a") as f:
        for (s, _), o in zip(todo, outputs):
            f.write(json.dumps(dict(id=s["id"], verdict=o.outputs[0].text.strip().upper())) + "\n")
    print(f"graded {len(todo)} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
