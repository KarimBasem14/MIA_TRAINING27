"""Flickr8k parsing and leakage-free image-level splitting."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import random


@dataclass(frozen=True)
class ImageCaptions:
    image: str
    captions: tuple[str, ...]


def load_flickr8k_captions(captions_file: str | Path) -> list[ImageCaptions]:
    """Read Flickr8k.token.txt or a captions CSV with image/caption columns."""
    path = Path(captions_file)
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if path.suffix.lower() == ".csv" or first_line.lower().startswith("image,"):
            reader = csv.DictReader(handle)
            fields = {field.lower().strip(): field for field in reader.fieldnames or []}
            image_field = fields.get("image") or fields.get("filename") or list(fields.values())[0]
            caption_field = fields.get("caption") or fields.get("comment") or list(fields.values())[1]
            rows = ((row[image_field].strip(), row[caption_field].strip()) for row in reader)
        else:
            def text_rows():
                for line in handle:
                    parts = line.rstrip("\r\n").split("\t", maxsplit=1)
                    if len(parts) == 2:
                        image_id, caption = parts
                        yield image_id.split("#", maxsplit=1)[0].strip(), caption.strip()
            rows = text_rows()
        for image, caption in rows:
            if image and caption:
                grouped[image].append(caption)
    return [ImageCaptions(image, tuple(captions)) for image, captions in sorted(grouped.items())]


def split_images(
    records: list[ImageCaptions],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[ImageCaptions], list[ImageCaptions], list[ImageCaptions]]:
    """Split unique images before expanding captions, preventing caption leakage."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum to less than 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_count = round(len(shuffled) * test_fraction)
    validation_count = round(len(shuffled) * validation_fraction)
    test = shuffled[:test_count]
    validation = shuffled[test_count : test_count + validation_count]
    train = shuffled[test_count + validation_count :]
    return train, validation, test
