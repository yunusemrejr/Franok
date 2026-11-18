"""Prepare held-out multilingual retrieval queries and 976 unrelated saved-turn distractors."""

import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, decoders
from training.third.batches import Conversations
from tools.transfer import cases

p = Path("work/v3/semantic/evaluation")
p.mkdir(exist_ok=True, parents=True)
ids = json.loads(Path("tests/transfer_ids.json").read_text())
notes = json.loads(Path("tests/recall_cases.json").read_text())
rng = np.random.default_rng(67145)
tok = Tokenizer.from_file("web/tokenizer.json")
tok.decoder = decoders.ByteLevel()
corpus = Conversations("val")
distractors = []
for lang, names in [
    ("en", ["everyday-en", "human", "short-dialogue", "short-replies"]),
    ("tr", ["everyday-tr", "native-chat", "turkish-tasks", "short-replies"]),
]:
    docs = set()
    for name in names:
        r = corpus.records[lang, name]
        for i in rng.permutation(len(r.index))[:320]:
            a, m, e = r.index[i]
            s = tok.decode(r.data[m + 1 : e - 1].astype(int).tolist()).strip()
            if 20 <= len(s) <= 1300 and not s.startswith(("http", "{")):
                docs.add(s)
    chosen = rng.choice(sorted(docs), 488, replace=False).tolist()
    distractors.extend(chosen)
items = {}
for suite in ("xquad", "belebele"):
    examples = list(cases(suite, ids[suite]))
    for lang in ("en", "tr"):
        own = [r for r in examples if r["lang"] == lang]
        other = [r for r in examples if r["lang"] != lang]
        for label, rows in [("same", own), ("cross", other)]:
            assert [r["key"] for r in rows] == [r["key"] for r in own]
            assert len(set(r["passage"] for r in rows)) == len(rows)
            items[f"{suite}-{lang}-{label}"] = dict(
                docs=[r["passage"] for r in rows],
                queries=[[r["question"], j, lang] for j, r in enumerate(own)],
            )
texts = [r[0] for r in notes] + distractors
order = rng.permutation(len(texts)).tolist()
assert len(set(texts)) == 1000
docs = [texts[i] for i in order]
for kind, ix in [("same", 1), ("cross", 2)]:
    items["notes-" + kind] = dict(
        docs=docs,
        queries=[
            [r[ix], order.index(i), ("en", "tr")[(i + (ix - 1)) % 2]]
            for i, r in enumerate(notes)
        ],
    )
(p / "sets.json").write_text(json.dumps(items, ensure_ascii=False))
print({k: (len(v["docs"]), len(v["queries"])) for k, v in items.items()}, flush=True)
