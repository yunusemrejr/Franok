"""Bounded Colab training, validation, and resumable checkpoints."""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from franok.model import Config, Franok

DATA = Path("work/data")
OUT = Path("work/run")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=3200)
    p.add_argument("--batch",type=int,default=24)
    p.add_argument("--length",type=int,default=256)
    p.add_argument("--minutes",type=float,default=65)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Select a Colab GPU.")
    OUT.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(1842)
    rng=np.random.default_rng(1842)
    cfg=Config()
    device="cuda"
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Keep FP32 parameters and optimizer state. Autocast only matrix operations.
    net=Franok(cfg).to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=.001,betas=(.9,.95),weight_decay=.08,fused=True)
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    arrays={p.stem:np.memmap(p,dtype="<u2",mode="r") for p in DATA.glob("*.bin")}
    first=0
    if args.resume:
        ck=torch.load(OUT/"resume.pt",map_location=device,weights_only=False)
        net.load_state_dict(ck["model"]);optim.load_state_dict(ck["opt"])
        first=ck["step"];rng.bit_generator.state=ck["rng"];scaler.load_state_dict(ck["scaler"])
    metrics=OUT/"metrics.jsonl"
    def log(**fields):
        with metrics.open("a") as f:f.write(json.dumps(fields)+"\n")
        print(fields,flush=True)
    def batch(split,chooser,bsize,chat_prob=.32,lang=None,kind=None):
        rows=[]
        for i in range(bsize):
            lng=lang or ("en" if i%2==0 else "tr")
            src=kind or ("chat" if chooser.random()<chat_prob else "text")
            data=arrays[f"{split}-{lng}-{src}"]
            offset=chooser.integers(len(data)-args.length)
            rows.append(data[offset:offset+args.length+1])
        return torch.from_numpy(np.stack(rows).astype("int64")).to(device)
    def loss(rows):
        with torch.autocast(device,dtype=dtype):
            logits=net(rows[:,:-1])
            return F.cross_entropy(logits.flatten(0,1),rows[:,1:].flatten())
    @torch.no_grad()
    def validate(step):
        net.eval();scores={}
        for lng in ["en","tr"]:
            for src in ["chat","text"]:
                vrng=np.random.default_rng(7263)
                scores[lng+"-"+src]=sum(loss(batch("val",vrng,8,lang=lng,kind=src)).item() for _ in range(4))/4
        log(step=step,validation=scores)
        net.train()
    def save(step,final):
        weights={n:v.detach().to("cpu",torch.float16).contiguous() for n,v in net.state_dict().items()}
        save_file(weights,str(OUT/"weights.tmp"));(OUT/"weights.tmp").replace(OUT/"model.safetensors")
        (OUT/"config.json").write_text(json.dumps(asdict(cfg),indent=2))
        (OUT/"progress.json").write_text(json.dumps(dict(step=step,tokens=step*args.batch*args.length,finished=final)))
        torch.save(dict(model=net.state_dict(),opt=optim.state_dict(),step=step,rng=rng.bit_generator.state,scaler=scaler.state_dict()),OUT/"resume.tmp")
        (OUT/"resume.tmp").replace(OUT/"resume.pt")
    log(hardware=torch.cuda.get_device_name(),torch=torch.__version__,parameters=sum(p.numel() for p in net.parameters()),options=vars(args))
    validate(first)
    start=time.monotonic()
    for i in range(first,args.steps):
        elapsed=time.monotonic()-start
        progress=max(i/args.steps,elapsed/(args.minutes*60))
        lr=.00085*min((i+1)/80,1)*(.16+.84*(1+math.cos(math.pi*min(1,progress)))/2)
        optim.param_groups[0]["lr"]=lr
        optim.zero_grad(set_to_none=True)
        l=loss(batch("train",rng,args.batch,chat_prob=.7 if progress>.78 else .28))
        scaler.scale(l).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
        scaler.step(optim);scaler.update()
        step=i+1
        if step==1 or step%25==0:log(step=step,loss=round(l.item(),4),lr=lr,elapsed=round(time.monotonic()-start,1))
        done=step==args.steps or time.monotonic()-start>=60*args.minutes
        if step%400==0 or done:
            validate(step);save(step,done)
        if done:break

if __name__=="__main__":main()
