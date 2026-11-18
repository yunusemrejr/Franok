"""Regenerate the pinned multiple-choice comparisons without redistributing test passages."""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from .reading import REV


def main():
    root = Path("work/measurement")
    root.mkdir(parents=True, exist_ok=True)
    prior = json.loads(Path("docs/evaluation/franok-belebele.json").read_text())
    samples = []
    for lang, dialect in [("en", "eng_Latn"), ("tr", "tur_Latn")]:
        path = hf_hub_download(
            "facebook/belebele",
            "data/" + dialect + ".jsonl",
            revision=REV,
            repo_type="dataset",
        )
        all_items = {
            (r["link"], r["question_number"]): r
            for r in map(json.loads, Path(path).read_text().splitlines())
        }
        for item in prior["items"]:
            if item["language"] != lang:
                continue
            key = item["link"], item["question"]
            r = all_items[key]
            samples.append(
                dict(
                    suite="reading",
                    lang=lang,
                    key=list(key),
                    q=r["flores_passage"] + "\n\n" + r["question"],
                    choices=[r["mc_answer" + str(j)] for j in range(1, 5)],
                    gold=int(r["correct_answer_num"]) - 1,
                )
            )
    for i, (lang, q, correct, foil) in enumerate(
        json.loads(Path("tests/dialogue.json").read_text())
    ):
        samples.append(
            dict(suite="short", lang=lang, key=i, q=q, choices=[correct, foil], gold=0)
        )
    (root / "cases.json").write_text(json.dumps(samples, ensure_ascii=False))


if __name__ == "__main__":
    main()
