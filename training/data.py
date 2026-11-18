"""Prepare a reproducible two-language corpus and fit a fresh byte BPE."""
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

ROOT = Path("work/data")
FILES = {
    "en": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.en/train-00000-of-00041.parquet"),
    "tr": ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.tr/train-00000-of-00002.parquet"),
    "oasst": ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b", "data/train-00000-of-00001-b42a775f407cee45.parquet"),
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "databricks-dolly-15k.jsonl"),
    "instructions": ("merve/turkish_instructions", "ed3381a23d9328b02054456d8ca5bdc157f617d5", "instructions.csv"),
    "reading": ("GoktugD/turkish-extractive-qa-1.5m", "8d1453c6643e6c76d6ecae9a63094f47748b474b", "data/train-00000-of-00015.parquet"),
}
PATTERN = r" ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s"
SPECIAL = ["<pad>", "<bos>", "<end>", "<user>", "<assistant>", "<en>", "<tr>"]
PAD, BOS, END, USER, ASSISTANT, EN, TR = range(7)


def parquet(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        yield from batch.to_pylist()


def documents(paths):
    for lang in ("en", "tr"):
        chars = 0
        for row in parquet(paths[lang]):
            body = row["text"][:16000].strip()
            if len(body) < 150:
                continue
            yield lang, "wiki", row["id"], [body]
            chars += len(body)
            if chars >= 95_000_000:
                break
    messages = {r["message_id"]: r for r in parquet(paths["oasst"]) if r["lang"] in ("en","tr") and not r["deleted"] and not r["synthetic"]}
    for row in messages.values():
        parent = messages.get(row["parent_id"])
        if row["role"] == "assistant" and row["rank"] in (0,None) and parent and parent["lang"] == row["lang"]:
            yield row["lang"], "oasst", row["message_tree_id"], [parent["text"],row["text"]]
    with open(paths["dolly"]) as stream:
        for line in stream:
            r = json.loads(line)
            yield "en", "dolly", r["instruction"], [("\n".join(filter(None,[r["context"],r["instruction"]]))),r["response"]]
    with open(paths["instructions"]) as stream:
        for row in csv.DictReader(stream):
            r = {k.strip(): (v or "").strip() for k,v in row.items() if k}
            yield "tr", "instructions", r["talimat"], ["\n".join(filter(None,[r["giriş"],r["talimat"]])),r["çıktı"]]
    for i, r in enumerate(parquet(paths["reading"])):
        if i >= 16000:
            break
        yield "tr", "reading", r["context"], [r["context"]+"\n"+r["question"],r["answer"]]


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,(repo,rev,file) = item
        path = hf_hub_download(repo,file,revision=rev,repo_type="dataset")
        print("Downloaded",name,flush=True)
        return name,path
    with ThreadPoolExecutor(4) as pool:
        paths = dict(pool.map(fetch,FILES.items()))
    records = []
    seen = set()
    for lang,source,key,parts in documents(paths):
        if not all(p.strip() for p in parts): continue
        digest = hashlib.sha256("\n".join(parts).encode()).digest()
        if digest in seen: continue
        seen.add(digest)
        bucket = int.from_bytes(hashlib.sha256((source+lang+str(key)).encode()).digest()[:4],"little") % 100
        records.append((lang,source,parts,"val" if bucket == 0 else "train"))
    random.Random(1842).shuffle(records)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PATTERN),"isolated"),pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
    # Reserve control IDs through the trainer, then remove special-text matching:
    # a user who types a role marker gets ordinary byte tokens.
    tok.train_from_iterator((p[:500] for _,_,parts,split in records if split=="train" for p in parts), trainers.BpeTrainer(vocab_size=8192,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    spec=json.loads(tok.to_str())
    spec["added_tokens"]=[]
    tok=Tokenizer.from_str(json.dumps(spec))
    tok.save(str(ROOT/"tokenizer.json"))
    handles={}
    counts={}
    try:
        for start in range(0,len(records),256):
            batch=records[start:start+256]
            encoded=iter(tok.encode_batch([p for _,_,parts,_ in batch for p in parts]))
            for lang,source,parts,split in batch:
                segments=[next(encoded).ids for _ in parts]
                name=f"{split}-{lang}-" + ("text" if len(parts)==1 else "chat")
                ids=[BOS,EN if lang=="en" else TR]
                if len(parts)==1:
                    ids+=segments[0]+[END]
                else:
                    ids += [USER]+segments[0]+[ASSISTANT]+segments[1]+[END]
                if name not in handles: handles[name]=(ROOT/(name+".bin")).open("wb")
                handles[name].write(np.asarray(ids,dtype="<u2").tobytes())
                counts[name]=counts.get(name,0)+len(ids)
    finally:
        for handle in handles.values(): handle.close()
    (ROOT/"sources.json").write_text(json.dumps(dict(files=FILES,tokens=counts),indent=2))
    print(counts,flush=True)

if __name__ == "__main__":
    prepare()
