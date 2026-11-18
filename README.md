# Franok

Franok is an experimental 314,061,312-parameter English–Turkish language model by Yunus Emre Vurgun. It trains from random initialization. Its anchor-rotation decoder activates one of eight feed-forward banks per token per layer. The full vocabulary head included, a token uses 49,820,160 parameters.

[Chat in your browser](https://franok.yunusemrevurgun.com/) · [Weights on Hugging Face](https://huggingface.co/yunusemrejr/Franok) · [Open the training notebook in Colab](https://colab.research.google.com/github/yunusemrejr/Franok/blob/main/training/Franok.ipynb)

## Chat

Choose English or Türkçe, then send a message. The browser downloads a roughly 196 MB four-bit checkpoint on first use and generates locally in a WebAssembly SIMD worker. Chats stay in the browser or desktop app's local profile. Search any saved turn and attach it to a new question to bring it back into the model's 256-token window. The Architecture view lets you trace the routing rule. The Ubuntu app includes a GNOME application menu entry, dock icon and launcher.

```sh
npm ci
npx install-electron
npm run install:ubuntu -- --pin
gtk-launch franok
```

The desktop uses a verified local copy in `work/export/model.q4` when present. Otherwise it downloads the released weights on first use. Browser cache availability and device memory affect startup. Clearing site or application storage removes that profile's sessions.

## Train and run

The pinned dataset revisions, clean split, tokenizer, model, optimizer schedules, reference decoder and four-bit exporter are all included. Select a GPU runtime in Colab. The notebook runs the same commands.

```sh
python -m pip install -r requirements.txt
python -m training.data
python -m training.train --steps 3200 --batch 24 --minutes 65
python -m training.refine --pretrain 4800 --sft 1200 --batch 24 --minutes 35
python -m training.export
bash runtime/build.sh
```

For the PyTorch reference chat, put the released `model.safetensors`, `config.json` and `tokenizer.json` in one directory and run `python -m franok.chat DIRECTORY --lang tr`. The incremental browser runtime ships prebuilt. Rebuilding its kernels requires Clang with WebAssembly SIMD and wasm-ld.

## Technical material

- [Routing mathematics, parameter counts and quantization](docs/architecture.md)
- [Training sources, attribution and splits](docs/data.md)
- [Namecheap deployment and Ubuntu installation](docs/deployment.md)
- [Evaluation receipts](docs/evaluation/)

The MIT license covers Franok's original code and released weights. Source datasets keep their upstream licenses. The short Colab run targets a research checkpoint. Check evaluation results and sample generations before relying on its answers.
