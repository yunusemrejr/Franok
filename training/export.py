"""Export raw tensors to Franok Q4/32. Each block is float32 scale + 16 byte pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def export(run, out, tokenizer):
    out.mkdir(parents=True,exist_ok=True)
    weights = out/"model.q4"
    catalog={}
    with safe_open(run/"model.safetensors",framework="numpy") as model, weights.open("wb") as dest:
        for key in sorted(model.keys()):
            a=model.get_tensor(key)
            offset=dest.tell()
            if a.ndim==1:
                dest.write(a.astype("<f4").tobytes())
                typ="f32"
            else:
                if a.shape[-1]%32:raise ValueError(f"Unaligned {key}")
                blocks=a.reshape(-1,32)
                for first in range(0,len(blocks),32768):
                    group=blocks[first:first+32768].astype("float32")
                    scale=np.maximum(np.max(np.abs(group),axis=1,keepdims=True)/7,1e-12).astype("<f4")
                    q=(np.rint(group/scale).clip(-7,7).astype("int8")+8).astype("uint8")
                    pairs=q[:,:16] | (q[:,16:]<<4)
                    dest.write(np.concatenate([scale.view("uint8"),pairs],axis=1).tobytes())
                typ="q4"
            catalog[key]=dict(offset=offset,bytes=dest.tell()-offset,shape=list(a.shape),type=typ)
    cfg=json.loads((run/"config.json").read_text())
    spec=json.loads(tokenizer.read_text())
    spec["decoder"]={"type":"ByteLevel","add_prefix_space":False,"trim_offsets":True,"use_regex":False}
    (out/"tokenizer.json").write_text(json.dumps(spec,ensure_ascii=False,separators=(",",":")))
    with weights.open("rb") as f:digest=hashlib.file_digest(f,"sha256").hexdigest()
    (out/"model.json").write_text(json.dumps(dict(format="franok-q4-v1",config=cfg,tensors=catalog,bytes=weights.stat().st_size,sha256=digest),indent=2))
    print(weights,weights.stat().st_size,digest)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=Path("work/run"))
    p.add_argument("--out",type=Path,default=Path("work/export"))
    p.add_argument("--tokenizer",type=Path,default=Path("work/data/tokenizer.json"))
    args=p.parse_args();export(args.run,args.out,args.tokenizer)
