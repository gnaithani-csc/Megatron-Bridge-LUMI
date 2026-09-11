"""Convert Energon synth-data shards to the format expected by QwenVLTaskEncoder.

The generated dataset stores one image per sample as:
  sample_XXXXXXXX.jpg   raw JPEG bytes
  sample_XXXXXXXX.json  {"id": "...", "conversations": [...]}

cook_chatml_sample expects:
  sample_XXXXXXXX.jpgs  pickle.dumps([np.ndarray, ...])  (list of numpy arrays)
  sample_XXXXXXXX.json  [{"from": "human", ...}, ...]   (bare conversation list)

This script re-packs every *.tar shard and regenerates the companion *.tar.idx file.
The .nv-meta folder is copied verbatim.

Usage:
    python scripts/data/convert_synth_data.py \\
        --input  /scratch/.../synth_data/Qwen3_VL/cap_pretrain \\
        --output /scratch/.../synth_data/Qwen3_VL/cap_pretrain_converted
"""

import argparse
import io
import json
import os
import pickle
import shutil
import struct
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def _load_shard_samples(tf: tarfile.TarFile) -> dict[str, dict[str, bytes]]:
    """Return {stem: {ext: raw_bytes}} for all members in an open TarFile."""
    samples: dict[str, dict[str, bytes]] = {}
    for member in tf.getmembers():
        if "." not in member.name:
            continue
        stem, ext = member.name.rsplit(".", 1)
        fobj = tf.extractfile(member)
        if fobj is None:
            continue
        samples.setdefault(stem, {})[ext] = fobj.read()
    return samples


def _convert_jpg_to_jpgs(jpg_bytes: bytes) -> bytes:
    """Decode a raw JPEG and pickle it as a single-element list of numpy arrays."""
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    return pickle.dumps([np.array(img)])


def _unwrap_conversation_json(raw_bytes: bytes) -> bytes:
    """Return the bare conversation list as JSON bytes.

    Accepts either:
      - {"id": "...", "conversations": [...]}  (generated format)
      - [{"from": "human", ...}, ...]          (already correct, passed through)
    """
    parsed = json.loads(raw_bytes)
    if isinstance(parsed, dict):
        parsed = parsed.get("conversations", parsed)
    return json.dumps(parsed, ensure_ascii=False).encode()


def convert_shard(in_path: Path, out_path: Path) -> int:
    """Convert one shard and write its .tar.idx.  Returns the number of samples written."""
    with tarfile.open(in_path, "r") as in_tf:
        samples = _load_shard_samples(in_tf)

    count = 0
    # sample_offsets[i] = byte offset of the first tar member for sample i.
    # A final sentinel entry (offset just past the last sample's data block) is
    # appended to match the N+1 entry format Energon expects.
    sample_offsets: list[int] = []

    with tarfile.open(out_path, "w") as out_tf:
        for stem in sorted(samples):
            fields = samples[stem]

            # Record start-of-sample offset before writing any member for this sample.
            sample_offsets.append(out_tf.offset)

            # Image: convert jpg to jpgs.
            if "jpg" in fields:
                jpgs_bytes = _convert_jpg_to_jpgs(fields["jpg"])
                info = tarfile.TarInfo(name=f"{stem}.jpgs")
                info.size = len(jpgs_bytes)
                out_tf.addfile(info, io.BytesIO(jpgs_bytes))
            elif "jpgs" in fields:
                # Already in target format, copy as-is.
                info = tarfile.TarInfo(name=f"{stem}.jpgs")
                info.size = len(fields["jpgs"])
                out_tf.addfile(info, io.BytesIO(fields["jpgs"]))

            # Conversation: unwrap the dict wrapper if present.
            if "json" in fields:
                json_bytes = _unwrap_conversation_json(fields["json"])
                info = tarfile.TarInfo(name=f"{stem}.json")
                info.size = len(json_bytes)
                out_tf.addfile(info, io.BytesIO(json_bytes))

            count += 1

        # Sentinel: offset just past the last member's data block.
        sample_offsets.append(out_tf.offset)

    # Write N+1 little-endian int64 values to shard_name.tar.idx
    idx_path = out_path.with_suffix(out_path.suffix + ".idx")
    idx_path.write_bytes(struct.pack(f"<{len(sample_offsets)}q", *sample_offsets))

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert synth-data shards to cook_chatml_sample format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, metavar="DIR",
                        help="Source Energon dataset directory (contains *.tar shards and .nv-meta)")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="Destination directory (must not be the same as --input)")
    args = parser.parse_args()

    in_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()

    if in_dir == out_dir:
        parser.error("--input and --output must be different directories")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy .nv-meta unchanged (dataset.yaml, split.yaml, index.sqlite, etc.)
    in_meta = in_dir / ".nv-meta"
    if in_meta.exists():
        out_meta = out_dir / ".nv-meta"
        if out_meta.exists():
            shutil.rmtree(out_meta)
        shutil.copytree(in_meta, out_meta)
        print(f"Copied .nv-meta → {out_meta}")
    else:
        print(f"Warning: no .nv-meta found in {in_dir}")

    shards = sorted(in_dir.glob("*.tar"))
    if not shards:
        print(f"No *.tar shards found in {in_dir}. Nothing to convert.")
        return

    total_samples = 0
    for shard in shards:
        out_shard = out_dir / shard.name
        print(f"  {shard.name} → {out_shard.name} ...", end=" ", flush=True)
        n = convert_shard(shard, out_shard)
        total_samples += n
        print(f"{n} samples")

    print(f"\nDone. {total_samples} samples across {len(shards)} shard(s) written to {out_dir}")


if __name__ == "__main__":
    main()
