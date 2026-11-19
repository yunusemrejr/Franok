"""Publish an explicit model artifact set or an immutable Git tree to the model hub."""

import argparse
import hashlib
import json
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO = "yunusemrejr/Franok"
ROOT = Path(__file__).resolve().parents[1]


def checksum(file):
    with file.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def finetune_mapping(run, export, version):
    """Release files for a full-parameter fine-tune whose run directory holds summary.json and best/."""
    summary = json.loads((run / "summary.json").read_text())
    best = run / "best"
    sha = checksum(best / "model.safetensors")
    if sha != summary["selected_sha256"]:
        raise ValueError("Fine-tune summary does not match the selected weights")
    mapping = {
        "model.safetensors": best / "model.safetensors",
        "config.json": best / "config.json",
        "tokenizer.json": export / "tokenizer.json",
        "LICENSE": ROOT / "LICENSE",
        "README.md": (ROOT / "docs/MODEL_CARD.md").read_bytes(),
        f"training-record/{version}/summary.json": run / "summary.json",
    }
    for f in ("finetune.log", "filter-report.json"):
        if (run / f).is_file():
            mapping[f"training-record/{version}/{f}"] = run / f
    formats = {
        "fp32": dict(
            file="model.safetensors",
            bytes=(best / "model.safetensors").stat().st_size,
            sha256=sha,
        )
    }
    return mapping, formats, sha, best


def publish_weights(args):
    run, export = args.run, args.export
    version = json.loads((ROOT / "package.json").read_text())["version"]
    if (run / "summary.json").is_file() and not (run / "rl-finished.json").is_file():
        mapping, formats, base_sha, folder = finetune_mapping(run, export, version)
        # 0.3's adapters and merge stay available at its own revision; main holds only this release.
        stale = ["adapters/", "fp32/", "dialogue-reward/"]
        return publish(args, mapping, formats, base_sha, folder, version, stale)
    receipt = json.loads((run / "rl-finished.json").read_text())
    if checksum(run / "model.safetensors") != receipt["sha256"]:
        raise ValueError("RL receipt does not match root weights")
    if receipt["optimizer_updates"] == 0:
        raise ValueError("Missing reinforcement updates")
    mapping = {
        "model.safetensors": run / "model.safetensors",
        "config.json": run / "config.json",
        "tokenizer.json": export / "tokenizer.json",
        "LICENSE": ROOT / "LICENSE",
        "README.md": (ROOT / "docs/MODEL_CARD.md").read_bytes(),
        "adapters/base/model.safetensors": run / "base/model.safetensors",
        "adapters/base/config.json": run / "base/config.json",
    }
    for stage in ("dialogue", "evidence", "rl"):
        for f in ("adapter.json", "adapter.safetensors"):
            mapping[f"adapters/{stage}/{f}"] = run / "adapters" / stage / f
    version = json.loads((ROOT / "package.json").read_text())["version"]
    logs = ["rl-finished.json", "rl.jsonl"]
    if (run / "post-finished.json").exists():
        logs += [
            "pretrain.jsonl",
            "recovery.json",
            "sft.json",
            "dialogue.json",
            "evidence.json",
            "post.jsonl",
            "post-finished.json",
        ]
        continued = run.parent / "lm"
        heldout = run.parent / "evaluation/continued-text.json"
        if (
            checksum(continued / "model.safetensors")
            != json.loads(heldout.read_text())["sha256"]
        ):
            raise ValueError(
                "Pretraining and held-out text receipts refer to different weights"
            )
        mapping["pretrained/model.safetensors"] = continued / "model.safetensors"
        mapping["pretrained/config.json"] = continued / "config.json"
        mapping["pretrained/tokenizer.json"] = export / "tokenizer.json"
        mapping["pretrained/validation.json"] = heldout
        mapping[f"training-record/{version}/data-preparation.json"] = (
            run.parent / "preparation.json"
        )
    else:
        logs += ["progress.json", "finished.json", "metrics.jsonl"]
    for f in logs:
        mapping[f"training-record/{version}/{f}"] = run / f
    reward = run / "dialogue-reward"
    if (reward / "validation.json").is_file():
        for f in ("head.safetensors", "training.json", "validation.json"):
            mapping["dialogue-reward/" + f] = reward / f
    products = run / "adapters/rl/updates.safetensors"
    if products.is_file():
        mapping["adapters/rl/updates.safetensors"] = products
        mapping["adapters/pre-rl/model.safetensors"] = run / "pre-rl.safetensors"
        mapping["adapters/pre-rl/config.json"] = run / "config.json"
    full = run.parent / "fp32"
    proof = json.loads((full / "reconstruction.json").read_text())
    if checksum(full / "model.safetensors") != proof["sha256"]:
        raise ValueError("Wrong FP32 reconstruction")
    if (
        receipt.get("fp32_sha256", proof["sha256"]) != proof["sha256"]
        or proof["fp16_different_elements"]
    ):
        raise ValueError("Merged weights disagree with the training host")
    mapping.update(
        {
            "fp32/model.safetensors": full / "model.safetensors",
            "fp32/config.json": full / "config.json",
            "fp32/reconstruction.json": full / "reconstruction.json",
        }
    )
    formats = {}
    formats["fp16"] = dict(
        file="model.safetensors",
        bytes=(run / "model.safetensors").stat().st_size,
        sha256=receipt["sha256"],
    )
    formats["fp32"] = dict(
        file="fp32/model.safetensors",
        bytes=(full / "model.safetensors").stat().st_size,
        sha256=proof["sha256"],
    )
    return publish(args, mapping, formats, receipt["sha256"], run, version)


def publish(args, mapping, formats, base_sha, run, version, stale=()):
    export = args.export
    reader = None
    if args.reader:
        folder = run / "reader"
        config = json.loads((folder / "config.json").read_text())
        training = json.loads((folder / "training.json").read_text())
        if "transferred_from" not in training and training["history"][-1]["step"] <= 0:
            raise ValueError("No trained reader updates")
        for f in ("head.safetensors", "config.json", "training.json"):
            mapping["reader/" + f] = folder / f
        reader = dict(
            url="./reader.safetensors",
            baseSha256=base_sha,
            sha256=checksum(folder / "head.safetensors"),
            **{k: config[k] for k in ("width", "hidden", "max_span", "threshold")},
        )
    search = None
    if args.search:
        cfg = json.loads((args.search / "recall.json").read_text())
        if checksum(args.search / "recall.bin") != cfg["sha256"]:
            raise ValueError("Search weights disagree with their training export")
        for f in ("recall.bin", "recall.json", "training.json"):
            mapping["search/" + f] = args.search / f
        search = {
            k: cfg[k] for k in ("format", "width", "vocab", "maxTokens", "sha256")
        }
        search["url"] = "./recall.bin"
    for k, manifest in [("q4", "model.json"), ("q8", "model-q8.json")]:
        path = export / ("model." + k)
        data = json.loads((export / manifest).read_text())
        if data["config"]["window"] != 1024 or data["sha256"] != checksum(path):
            raise ValueError("Invalid " + k + " export")
        mapping[path.name] = path
        mapping[manifest] = export / manifest
        formats[k] = dict(
            file=path.name,
            manifest="./" + manifest,
            bytes=path.stat().st_size,
            sha256=data["sha256"],
        )
    api = HfApi()
    if args.revision != "main":
        api.create_branch(repo_id=REPO, branch=args.revision, exist_ok=True)
    # A delete of a folder that an earlier release already removed fails the whole commit.
    present = {f.split("/", 1)[0] for f in api.list_repo_files(REPO, revision=args.revision)}
    stale = [p for p in stale if p.rstrip("/") in present]
    info = api.create_commit(
        repo_id=REPO,
        revision=args.revision,
        operations=[
            *(CommitOperationDelete(path_in_repo=p, is_folder=True) for p in stale),
            *(
                CommitOperationAdd(path_in_repo=k, path_or_fileobj=v)
                for k, v in mapping.items()
            ),
        ],
        commit_message=f"Publish Franok {version} weights, Q4, Q8 and training record",
    )
    if args.revision != "main":
        return info
    if reader:
        shutil.copyfile(
            run / "reader/head.safetensors", ROOT / "web/reader.safetensors"
        )
    if search:
        shutil.copyfile(args.search / "recall.bin", ROOT / "web/recall.bin")
    for record in formats.values():
        record["url"] = (
            f"https://huggingface.co/{REPO}/resolve/{info.oid}/" + record.pop("file")
        )
    for f in ("model.json", "model-q8.json", "tokenizer.json"):
        shutil.copyfile(export / f, ROOT / "web" / f)
    version = json.loads((ROOT / "package.json").read_text())["version"]
    file = f"franok_{version}_amd64.deb"
    release = dict(
        version=version,
        window=1024,
        default="q4",
        systemConditioned=True,
        revision=info.oid,
        formats=formats,
        installer=dict(
            file=file,
            version=version,
            url="https://franok.yunusemrevurgun.com/downloads/" + file,
        ),
        video=dict(
            url="./media/franok-from-scratch.mp4",
            poster="./media/franok-poster.jpg",
            captions={lang: "./media/franok-" + lang + ".vtt" for lang in ["en", "tr"]},
            youtube="https://youtu.be/XtNThi_pBjM",
        ),
    )
    if search:
        release["search"] = search
    if reader:
        release["reader"] = reader
    (ROOT / "web/release.json").write_text(json.dumps(release, indent=2) + "\n")
    return info


def publish_source():
    rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "-z", rev], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    ops = []
    for path in filter(None, paths):
        content = subprocess.check_output(["git", "show", rev + ":" + path], cwd=ROOT)
        ops.append(
            CommitOperationAdd(
                path_in_repo="source/" + path, path_or_fileobj=BytesIO(content)
            )
        )
        if path == "docs/MODEL_CARD.md":
            ops.append(
                CommitOperationAdd(
                    path_in_repo="README.md", path_or_fileobj=BytesIO(content)
                )
            )
    return HfApi().create_commit(
        repo_id=REPO,
        operations=ops,
        commit_message="Publish source, film and evaluations from " + rev[:12],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=("weights", "source"))
    p.add_argument(
        "--reader",
        action="store_true",
        help="Include the separately evaluated trained passage reader",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Publish experimental weights to a separate branch without activating the app",
    )
    p.add_argument(
        "--search",
        type=Path,
        help="Directory containing the trained bilingual session-search export",
    )
    p.add_argument("--run", type=Path, default=ROOT / "work/v2/run")
    p.add_argument("--export", type=Path, default=ROOT / "work/v2/export")
    args = p.parse_args()
    print(
        (publish_weights(args) if args.part == "weights" else publish_source()).oid,
        flush=True,
    )
