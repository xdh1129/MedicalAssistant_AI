import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from datasets import Dataset, DatasetDict, Image, IterableDataset, load_dataset


def _clean_value(value: Any) -> Any:
    """
    Ensure strings are valid UTF-8; drop/replace problematic bytes.
    """
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, bytes):
        # Encode binary payloads when present (should be rare after image handling).
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    return value


def _write_jsonl(
    split_ds: Union[Dataset, IterableDataset],
    split_path: Path,
    image_dir: Optional[Path],
) -> None:
    with split_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(split_ds):
            row = dict(row)
            if image_dir and isinstance(row.get("image"), dict):
                image_info = row["image"]
                image_name = image_info.get("path") or f"image_{idx}.jpg"
                image_out = image_dir / image_name
                image_out.parent.mkdir(parents=True, exist_ok=True)
                if image_info.get("bytes") and not image_out.exists():
                    with image_out.open("wb") as img_f:
                        img_f.write(image_info["bytes"])
                # Store relative path in JSON for portability.
                row["image"] = str(image_out.relative_to(split_path.parent))

            clean_row = {k: _clean_value(v) for k, v in row.items()}
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")


def _load_from_local_cache() -> Optional[Dict[str, Dataset]]:
    """
    Try to load the Arrow files already cached by huggingface.
    """
    cache_root = Path.home() / ".cache/huggingface/datasets/flaviagiammarino___vqa-rad/default/0.0.0"
    if not cache_root.exists():
        return None

    def find_split(name: str) -> Optional[Path]:
        for child in cache_root.iterdir():
            candidate = child / f"vqa-rad-{name}.arrow"
            if candidate.exists():
                return candidate
        return None

    train_arrow = find_split("train")
    test_arrow = find_split("test")
    if not (train_arrow and test_arrow):
        return None

    return {
        "train": Dataset.from_file(str(train_arrow)),
        "test": Dataset.from_file(str(test_arrow)),
    }


def main() -> None:
    """
    Download the VQA-RAD dataset and save each split as JSONL for easy inspection.
    """
    output_dir = Path("data/vqa-rad")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer offline cache when available to avoid repeated downloads / DNS issues.
    ds: Dict[str, Dataset] | DatasetDict | None = _load_from_local_cache()
    if ds is None:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
        ds = load_dataset("flaviagiammarino/vqa-rad", download_mode="reuse_dataset_if_exists")

    for split, split_ds in ds.items():
        if "image" in split_ds.column_names:
            # Keep reference paths instead of decoded PIL images so we can serialize to JSON.
            split_ds = split_ds.cast_column("image", Image(decode=False))
        split_path = output_dir / f"{split}.jsonl"
        image_dir = output_dir / "images" / split if "image" in split_ds.column_names else None
        _write_jsonl(split_ds, split_path, image_dir)
        print(f"Saved {split} split to {split_path}")


if __name__ == "__main__":
    main()
