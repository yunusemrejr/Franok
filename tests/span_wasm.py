"""Check the shipped readout arithmetic against PyTorch on several confidence margins."""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file
from franok.span import SpanReadout


def main():
    torch.manual_seed(3405)
    torch.set_num_threads(2)
    head = SpanReadout(32, 16, 7).eval()
    with TemporaryDirectory(prefix="franok-span-parity-") as folder, torch.no_grad():
        root = Path(folder)
        save_file(head.state_dict(), str(root / "head.safetensors"))
        records = []
        for n, pre in ((15, 8), (37, 13), (69, 9)):
            states = torch.randn(1, n, 32)
            legal = torch.rand(n - pre) > 0.22
            legal[0] = legal[-1] = True
            flags = torch.zeros(1, n, dtype=torch.bool)
            flags[0, pre:] = legal
            z = head(states, torch.tensor([pre]), torch.tensor([n]), flags)[0]
            k = int(z[:-1].argmax())
            j, d = divmod(k, head.max_span)
            records.append(
                dict(
                    states=states[0].tolist(),
                    offset=pre,
                    legal=legal.tolist(),
                    want=dict(
                        start=j - pre - d, end=j - pre, margin=float(z[k] - z[-1])
                    ),
                )
            )
        (root / "test.json").write_text(json.dumps(records))
        subprocess.run(["node", "tests/span_wasm.mjs", str(root)], check=True)


if __name__ == "__main__":
    main()
