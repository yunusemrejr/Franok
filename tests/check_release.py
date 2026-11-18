"""Save FP16-checkpoint reference logits on validation fragments for WASM parity."""
from pathlib import Path
import json
import numpy as np
import torch
from safetensors.torch import load_file
from franok.model import Config,Franok

p=Path("work/release-parity");p.mkdir(parents=True,exist_ok=True)
m=Franok(Config(**json.loads(Path("work/run/config.json").read_text()))).eval()
m.load_state_dict(load_file("work/run/model.safetensors"))
inputs=[]
for lang in ["en","tr"]:
    for kind in ["text","chat"]:
        a=np.memmap(f"work/data/val-{lang}-{kind}.bin",mode="r",dtype="<u2")
        inputs.append(a[101:150].astype("int64").tolist())
with torch.inference_mode():
    m(torch.tensor(inputs)[:,:-1]).numpy().astype("<f4").tofile(p/"logits.f32")
(p/"ids.json").write_text(json.dumps(inputs))
