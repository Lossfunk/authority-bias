"""Rebuild the frozen multiple-choice input without loading a model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.exp7.dataset_mc import create_mc_examples, save_mc_dataset  # noqa: E402

# SycophancyEval datasets/answer.jsonl at commit
# 9a1694221e3639887138f61deae344335eca6752. Fetch the immutable Git blob.
SOURCE_URL = (
    "https://api.github.com/repos/meg-tong/sycophancy-eval/git/blobs/"
    "fcf7f35fbdfdaf1f7fbb2b096936938906406652"
)
RAW_SHA256 = "3da2c2bbf685cf2c6cbfc6bf67449caad6fe1d56b67924087b3792ffba47bcc1"
OUTPUT_SHA256 = "1255ba8795b07cee961e5998c547341d0b248f032ba82ac1b56674c100d12574"


def verify(payload: bytes, expected: str, label: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label}: expected SHA-256 {expected}, got {actual}")


def prepare(raw_path: Path, output_path: Path, *, download: bool = False) -> None:
    if output_path.exists():
        verify(output_path.read_bytes(), OUTPUT_SHA256, str(output_path))
        print(f"Verified existing dataset: {output_path}")
        return

    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(f"{raw_path} is missing; supply --raw-path or --download")
        request = Request(SOURCE_URL, headers={"User-Agent": "Lossfunk-persona-vectors"})
        with urlopen(request, timeout=120) as response:
            blob = json.load(response)
        if blob.get("encoding") != "base64":
            raise ValueError("Unexpected encoding for the pinned GitHub dataset blob")
        payload = base64.b64decode(blob["content"])
        verify(payload, RAW_SHA256, "Downloaded source")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
    verify(raw_path.read_bytes(), RAW_SHA256, str(raw_path))

    with tempfile.TemporaryDirectory(prefix="persona-data-") as temporary:
        generated = Path(temporary) / "dataset.jsonl"
        save_mc_dataset(create_mc_examples(raw_path, seed=7), generated)
        payload = generated.read_bytes()
        verify(payload, OUTPUT_SHA256, "Prepared dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output:
        output.write(payload)
    print(f"Prepared and verified dataset: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=REPO_ROOT / "data/answer.jsonl")
    parser.add_argument("--output-path", type=Path, default=REPO_ROOT / "data/exp7_mc_dataset.jsonl")
    parser.add_argument("--download", action="store_true", help="Fetch the pinned public source")
    args = parser.parse_args()
    prepare(args.raw_path, args.output_path, download=args.download)


if __name__ == "__main__":
    main()
