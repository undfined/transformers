#!/usr/bin/env python
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compare model weights stored as safetensors.

Examples:

    python scripts/compare_safetensors_weights.py ./model_a ./model_b
    python scripts/compare_safetensors_weights.py allenai/model-a ./model_b --revision-a main
    python scripts/compare_safetensors_weights.py a.safetensors b.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


INDEX_FILENAMES = ("model.safetensors.index.json", "adapter_model.safetensors.index.json")
SINGLE_FILE_NAMES = ("model.safetensors", "adapter_model.safetensors")
DEFAULT_CHUNK_SIZE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class TensorEntry:
    filename: Path
    shape: tuple[int, ...]
    dtype: str
    data_offsets: tuple[int, int]
    data_start: int


@dataclass(frozen=True)
class TensorIndex:
    source: str
    root: Path
    tensors: dict[str, TensorEntry]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", help="Local safetensors file/directory or Hugging Face Hub model ID.")
    parser.add_argument("model_b", help="Local safetensors file/directory or Hugging Face Hub model ID.")
    parser.add_argument("--revision-a", help="Hub revision for model_a.")
    parser.add_argument("--revision-b", help="Hub revision for model_b.")
    parser.add_argument("--subfolder-a", default="", help="Subfolder containing safetensors for model_a.")
    parser.add_argument("--subfolder-b", default="", help="Subfolder containing safetensors for model_b.")
    parser.add_argument("--token", help="Hugging Face token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached Hub files.")
    parser.add_argument(
        "--include-all-safetensors",
        action="store_true",
        help="If a directory has multiple safetensors files and no index, compare all of them.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_SIZE_BYTES // (1024 * 1024),
        help="Approximate maximum tensor byte chunk to read at once.",
    )
    parser.add_argument("--max-differences", type=int, default=20, help="Stop after reporting this many differences.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    return parser.parse_args()


def path_like(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return True
    if value.endswith((".json", ".safetensors")):
        return True
    return "/" in value and value.count("/") != 1


def resolve_source(
    source: str,
    *,
    revision: str | None,
    subfolder: str,
    token: str | None,
    local_files_only: bool,
) -> tuple[Path, str]:
    path = Path(source).expanduser()
    if path.exists() or path_like(source):
        if not path.exists():
            raise FileNotFoundError(f"Local path does not exist: {source}")
        return path, source

    allow_patterns = ["*.safetensors", "*.safetensors.index.json"]
    if subfolder:
        clean_subfolder = subfolder.strip("/")
        allow_patterns = [f"{clean_subfolder}/*.safetensors", f"{clean_subfolder}/*.safetensors.index.json"]

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=source,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns,
    )
    return Path(snapshot_path), f"{source}@{revision}" if revision else source


def parse_safetensors_header(filename: Path) -> tuple[dict, int]:
    with filename.open("rb") as handle:
        header_length_bytes = handle.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"{filename} is not a valid safetensors file: missing header length")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{filename} is not a valid safetensors file: incomplete header")

    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"{filename} is not a valid safetensors file: header is not an object")
    return header, 8 + header_length


def tensor_entries_from_file(filename: Path) -> dict[str, TensorEntry]:
    header, data_start = parse_safetensors_header(filename)
    data_size = filename.stat().st_size - data_start
    tensors = {}

    for key, tensor_info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(tensor_info, dict):
            raise ValueError(f"{filename} has invalid metadata for tensor {key!r}")

        dtype = tensor_info.get("dtype")
        shape = tensor_info.get("shape")
        data_offsets = tensor_info.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"{filename} has missing or invalid dtype for tensor {key!r}")
        if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError(f"{filename} has missing or invalid shape for tensor {key!r}")
        if (
            not isinstance(data_offsets, list)
            or len(data_offsets) != 2
            or not all(isinstance(offset, int) for offset in data_offsets)
        ):
            raise ValueError(f"{filename} has missing or invalid data_offsets for tensor {key!r}")
        if not (0 <= data_offsets[0] <= data_offsets[1] <= data_size):
            raise ValueError(f"{filename} has out-of-bounds data_offsets for tensor {key!r}")

        tensors[key] = TensorEntry(
            filename=filename,
            shape=tuple(shape),
            dtype=dtype,
            data_offsets=(data_offsets[0], data_offsets[1]),
            data_start=data_start,
        )
    return tensors


def tensor_entries_from_files(files: list[Path]) -> dict[str, TensorEntry]:
    tensors: dict[str, TensorEntry] = {}
    duplicate_keys: dict[str, list[str]] = {}
    for filename in files:
        for key, entry in tensor_entries_from_file(filename).items():
            if key in tensors:
                duplicate_keys.setdefault(key, [str(tensors[key].filename)]).append(str(filename))
                continue
            tensors[key] = entry

    if duplicate_keys:
        example_key = next(iter(duplicate_keys))
        raise ValueError(f"Duplicate tensor key {example_key!r} found in: {duplicate_keys[example_key]}")
    return tensors


def tensor_entries_from_index(index_file: Path) -> dict[str, TensorEntry]:
    payload = json.loads(index_file.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_file} does not contain a 'weight_map' object")

    tensors: dict[str, TensorEntry] = {}
    open_files: dict[Path, dict[str, TensorEntry]] = {}
    for key, relative_filename in weight_map.items():
        filename = index_file.parent / relative_filename
        if filename not in open_files:
            if not filename.exists():
                raise FileNotFoundError(f"Index points to missing shard: {filename}")
            open_files[filename] = tensor_entries_from_file(filename)
        if key not in open_files[filename]:
            raise KeyError(f"Index maps {key!r} to {filename}, but that shard does not contain the key")
        tensors[key] = open_files[filename][key]
    return tensors


def discover_tensors(
    root: Path,
    *,
    source_label: str,
    subfolder: str,
    include_all_safetensors: bool,
) -> TensorIndex:
    target = root / subfolder.strip("/") if subfolder and root.is_dir() else root

    if target.is_file():
        if target.name.endswith(".safetensors.index.json"):
            tensors = tensor_entries_from_index(target)
        elif target.suffix == ".safetensors":
            tensors = tensor_entries_from_file(target)
        else:
            raise ValueError(f"Expected a .safetensors file or .safetensors.index.json file, got: {target}")
        return TensorIndex(source=source_label, root=target.parent, tensors=tensors)

    if not target.is_dir():
        raise FileNotFoundError(f"Could not find safetensors source: {target}")

    for index_filename in INDEX_FILENAMES:
        index_file = target / index_filename
        if index_file.exists():
            return TensorIndex(source=source_label, root=target, tensors=tensor_entries_from_index(index_file))

    for single_filename in SINGLE_FILE_NAMES:
        single_file = target / single_filename
        if single_file.exists():
            return TensorIndex(source=source_label, root=target, tensors=tensor_entries_from_file(single_file))

    safetensors_files = sorted(target.glob("*.safetensors"))
    if len(safetensors_files) == 1 or (safetensors_files and include_all_safetensors):
        return TensorIndex(source=source_label, root=target, tensors=tensor_entries_from_files(safetensors_files))

    if safetensors_files:
        filenames = ", ".join(file.name for file in safetensors_files[:10])
        raise ValueError(
            f"Found multiple safetensors files in {target} but no index file. "
            f"Use --include-all-safetensors to compare all of them. Files: {filenames}"
        )
    raise FileNotFoundError(f"No safetensors weights found in {target}")


def first_different_byte(chunk_a: bytes, chunk_b: bytes) -> int:
    for index, (byte_a, byte_b) in enumerate(zip(chunk_a, chunk_b)):
        if byte_a != byte_b:
            return index
    return min(len(chunk_a), len(chunk_b))


def compare_tensor_bytes(
    key: str,
    entry_a: TensorEntry,
    entry_b: TensorEntry,
    *,
    chunk_size_bytes: int,
) -> dict | None:
    start_a, stop_a = entry_a.data_offsets
    start_b, stop_b = entry_b.data_offsets
    size_a = stop_a - start_a
    size_b = stop_b - start_b
    if size_a != size_b:
        return {"kind": "data_length", "key": key, "a": size_a, "b": size_b}

    chunk_size_bytes = max(1, chunk_size_bytes)
    with entry_a.filename.open("rb") as handle_a, entry_b.filename.open("rb") as handle_b:
        handle_a.seek(entry_a.data_start + start_a)
        handle_b.seek(entry_b.data_start + start_b)

        compared = 0
        remaining = size_a
        while remaining:
            read_size = min(chunk_size_bytes, remaining)
            chunk_a = handle_a.read(read_size)
            chunk_b = handle_b.read(read_size)
            if chunk_a != chunk_b:
                return {
                    "kind": "values",
                    "key": key,
                    "byte_offset": compared + first_different_byte(chunk_a, chunk_b),
                }
            compared += read_size
            remaining -= read_size
    return None


def compare_indexes(
    index_a: TensorIndex,
    index_b: TensorIndex,
    *,
    chunk_size_bytes: int,
    max_differences: int,
) -> dict:
    differences = []
    keys_a = set(index_a.tensors)
    keys_b = set(index_b.tensors)

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    if only_a:
        differences.append({"kind": "missing_from_b", "count": len(only_a), "keys": only_a[:max_differences]})
    if only_b:
        differences.append({"kind": "missing_from_a", "count": len(only_b), "keys": only_b[:max_differences]})

    common_keys = sorted(keys_a & keys_b)
    compared_values = 0
    for key in common_keys:
        if len(differences) >= max_differences:
            break

        entry_a = index_a.tensors[key]
        entry_b = index_b.tensors[key]
        if entry_a.shape != entry_b.shape:
            differences.append({"kind": "shape", "key": key, "a": entry_a.shape, "b": entry_b.shape})
            continue
        if entry_a.dtype != entry_b.dtype:
            differences.append({"kind": "dtype", "key": key, "a": entry_a.dtype, "b": entry_b.dtype})
            continue

        difference = compare_tensor_bytes(
            key,
            entry_a,
            entry_b,
            chunk_size_bytes=chunk_size_bytes,
        )
        compared_values += 1
        if difference is not None:
            differences.append(difference)

    return {
        "same": not differences,
        "source_a": index_a.source,
        "source_b": index_b.source,
        "root_a": str(index_a.root),
        "root_b": str(index_b.root),
        "num_tensors_a": len(keys_a),
        "num_tensors_b": len(keys_b),
        "num_common_tensors": len(common_keys),
        "num_value_compared_tensors": compared_values,
        "differences": differences,
        "truncated": len(differences) >= max_differences,
    }


def print_report(report: dict) -> None:
    print(f"A: {report['source_a']} ({report['num_tensors_a']} tensors)")
    print(f"B: {report['source_b']} ({report['num_tensors_b']} tensors)")

    if report["same"]:
        print("MATCH: tensor keys, shapes, dtypes, and values are the same.")
        return

    print("DIFFERENT: tensor weights do not match.")
    for difference in report["differences"]:
        kind = difference["kind"]
        if kind in {"missing_from_a", "missing_from_b"}:
            side = "A" if kind == "missing_from_b" else "B"
            print(f"- {difference['count']} tensors only in {side}: {', '.join(difference['keys'])}")
        elif kind == "shape":
            print(f"- {difference['key']}: shape differs {difference['a']} vs {difference['b']}")
        elif kind == "dtype":
            print(f"- {difference['key']}: dtype differs {difference['a']} vs {difference['b']}")
        elif kind == "data_length":
            print(
                f"- {difference['key']}: serialized tensor byte length differs {difference['a']} vs {difference['b']}"
            )
        elif kind == "values":
            print(f"- {difference['key']}: tensor bytes differ at offset {difference['byte_offset']}")

    if report["truncated"]:
        print("- Difference report truncated; raise --max-differences for more.")


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    chunk_size_bytes = args.chunk_size_mb * 1024 * 1024

    root_a, label_a = resolve_source(
        args.model_a,
        revision=args.revision_a,
        subfolder=args.subfolder_a,
        token=token,
        local_files_only=args.local_files_only,
    )
    root_b, label_b = resolve_source(
        args.model_b,
        revision=args.revision_b,
        subfolder=args.subfolder_b,
        token=token,
        local_files_only=args.local_files_only,
    )
    index_a = discover_tensors(
        root_a,
        source_label=label_a,
        subfolder=args.subfolder_a,
        include_all_safetensors=args.include_all_safetensors,
    )
    index_b = discover_tensors(
        root_b,
        source_label=label_b,
        subfolder=args.subfolder_b,
        include_all_safetensors=args.include_all_safetensors,
    )
    report = compare_indexes(
        index_a,
        index_b,
        chunk_size_bytes=chunk_size_bytes,
        max_differences=args.max_differences,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if report["same"] else 1


if __name__ == "__main__":
    sys.exit(main())
