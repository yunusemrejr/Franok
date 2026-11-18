"""Publish only explicit model artifacts or the tracked public Git revision."""
import argparse
import json
import subprocess
from io import BytesIO
from pathlib import Path
from huggingface_hub import CommitOperationAdd,HfApi

REPO="yunusemrejr/Franok"
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("part",choices=("weights","source"))
    args=p.parse_args()
    add=lambda target,path:CommitOperationAdd(path_in_repo=target,path_or_fileobj=path)
    if args.part=="weights":
        mapping={"model.safetensors":"work/run/model.safetensors","config.json":"work/run/config.json",
                 "model.q4":"work/export/model.q4","model.json":"work/export/model.json",
                 "tokenizer.json":"work/export/tokenizer.json","README.md":"docs/MODEL_CARD.md","LICENSE":"LICENSE"}
        ops=[add(k,ROOT/v) for k,v in mapping.items()]
        info=HfApi().create_commit(repo_id=REPO,operations=ops,commit_message="Publish trained FP16 and four-bit Franok weights")
        release=dict(weights=f"https://huggingface.co/{REPO}/resolve/{info.oid}/model.q4")
        (ROOT/"web/release.json").write_text(json.dumps(release,indent=2)+"\n")
    else:
        rev=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        paths=subprocess.check_output(["git","ls-tree","-r","--name-only","-z",rev],cwd=ROOT).decode().split("\0")
        ops=[]
        for path in filter(None,paths):
            content=subprocess.check_output(["git","show",rev+":"+path],cwd=ROOT)
            ops.append(add("source/"+path,BytesIO(content)))
            if path=="docs/MODEL_CARD.md":ops.append(add("README.md",BytesIO(content)))
        info=HfApi().create_commit(repo_id=REPO,operations=ops,commit_message="Publish complete source and evaluations from "+rev[:12])
    print(info.oid,flush=True)

if __name__=="__main__":main()
