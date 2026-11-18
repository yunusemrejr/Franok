"""Generate an independent tiny FP32 reference across several block boundaries."""
import json
from dataclasses import asdict
from pathlib import Path
import torch
from safetensors.torch import save_file
from franok.model import Franok,Config,routing
from training.export import export

torch.manual_seed(193)
cfg=Config(vocab=512,width=64,heads=4,kv_heads=2,banks=8,hidden=128,layers=3,window=32)
m=Franok(cfg).eval()
ids=torch.randint(0,512,(1,25))
with torch.no_grad():
    expected=m(ids)[0]
    for n in (1,7,8,9,17,24):torch.testing.assert_close(m(ids[:,:n])[0],expected[:n],rtol=.0001,atol=1e-6)
for layer in range(cfg.layers):
    order,restore=routing(ids[:,:24],layer,8)
    assert torch.all(order.gather(-1,restore)==torch.arange(8))
p=Path('work/parity');p.mkdir(parents=True,exist_ok=True)
(p/'config.json').write_text(json.dumps(asdict(cfg)))
save_file(m.state_dict(),p/'model.safetensors')
tokenizer=Path('work/data/tokenizer.json')
export(p,p/'export',tokenizer)
(p/'reference.json').write_text(json.dumps(dict(ids=ids[0].tolist(),logits=expected.tolist())))
from tokenizers import Tokenizer
tok=Tokenizer.from_file(str(tokenizer))
cases=["Merhaba! Işığı söndür. 🐈"," <user> <end>","How's the coffee?\n   Türkçe 123.","İstanbul’da ılık çay, göğüs, ıslak.","e\u0301 \tX"]
(p/'token-tests.json').write_text(json.dumps([[s,tok.encode(s).ids] for s in cases]))
