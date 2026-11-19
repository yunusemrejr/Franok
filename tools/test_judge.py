"""Checks the judge with fake systems and a fake model: totals, blind mapping, and agreement with stored checks."""

import json
import tempfile
from pathlib import Path

from .compare_chat import meets
from .judge import build_items, judge, load_transcript, prompts_for, totals

CASES = [
    dict(case="a", lang="en", turns=["hi", "bye"], expect=[["hello"], None], rubric=["Greet.", "Say goodbye."]),
    dict(case="b", lang="tr", turns=["selam"], expect=[["merhaba"]], rubric=["Selam ver."]),
    dict(case="c", lang="en", turns=["x", "y"], expect=[None, ["yes"]], rubric=["R1", "R2"]),
]
GOOD, BAD = "ZEBRA", "noise"


def transcript(replies):
    convs = []
    for c in CASES:
        turns = []
        for q in c["turns"]:
            turns += [dict(role="user", content=q), dict(role="assistant", content=replies.pop(0))]
        convs.append(dict(id=c["case"], lang=c["lang"], turns=turns))
    return dict(model="fake", conversations=convs)


def fake_ask(prompts):
    return ["PASS" if GOOD in p.rpartition("reply to judge): ")[2] else "FAIL" for p in prompts]


def main():
    sysa = transcript([f"hello {GOOD}", BAD, f"merhaba {GOOD}", BAD, f"yes {GOOD}"])
    sysb = transcript([BAD, f"bye {GOOD}", BAD, f"{GOOD} x", BAD])
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for name, t in (("sysA", sysa), ("sysB", sysb)):
            paths[name] = Path(d, f"{name}.json")
            paths[name].write_text(json.dumps(t))
        loaded = {n: load_transcript(n, p, CASES) for n, p in paths.items()}
        broken = json.loads(paths["sysA"].read_text())
        broken["conversations"].pop()
        Path(d, "broken.json").write_text(json.dumps(broken))
        try:
            load_transcript("broken", Path(d, "broken.json"), CASES)
        except SystemExit as e:
            assert "no case 'c'" in str(e)
        else:
            raise AssertionError("missing case not rejected")

    items = build_items(CASES, loaded)
    assert len(items) == 2 * 5
    seen = []

    def spy(prompts):
        seen.append(prompts)
        return fake_ask(prompts)

    rows = judge(items, spy, seed=3)
    assert all("sysA" not in p and "sysB" not in p for batch in seen for p in batch)
    assert [p for p in seen[0]] != [prompts_for(i)[0] for i in items], "items were not shuffled"
    for r in rows:
        expected = GOOD in r["reply"]
        assert r["rubric"] == expected and r["sensible"] == expected, r
    t = totals(rows)
    assert t["sysA"]["rubric_passed"] == dict(passed=3, of=5, en=dict(passed=2, of=4), tr=dict(passed=1, of=1)), t
    assert t["sysB"]["rubric_passed"]["passed"] == 2 and t["sysB"]["sensible_passed"]["passed"] == 2
    assert t["sysA"]["expect_passed"] == dict(passed=3, of=3, en=dict(passed=2, of=2), tr=dict(passed=1, of=1)), t
    assert t["sysB"]["expect_passed"]["passed"] == 0 and t["sysB"]["expect_passed"]["of"] == 3

    stored = json.loads(Path("docs/evaluation/v4/franok04r2-v4-conversations.json").read_text())
    cases = {c["case"]: c for c in json.loads(Path("tests/v4_conversations.json").read_text())}
    checked = 0
    for conv in stored["conversations"]:
        case = cases[conv["id"]]
        for j, check in enumerate(conv["checks"]):
            if case["expect"][j] is not None:
                assert meets(conv["turns"][2 * j + 1]["content"], case["expect"][j], case) == check["passed"], (conv["id"], j)
                checked += 1
    print(f"ok: totals, blind mapping, {checked} stored checks agree")


if __name__ == "__main__":
    main()
