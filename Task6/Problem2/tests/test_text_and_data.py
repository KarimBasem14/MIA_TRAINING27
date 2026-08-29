from pathlib import Path

from captioning.data import ImageCaptions, load_flickr8k_captions, split_images
from captioning.text import Vocabulary, tokenize


def test_vocabulary_round_trip_and_padding():
    vocabulary = Vocabulary(min_frequency=1).fit(["A small dog", "A red dog"])
    encoded = vocabulary.encode("A small dog", max_length=7)
    assert encoded[0] == vocabulary.start_id
    assert vocabulary.end_id in encoded
    assert vocabulary.decode(encoded) == "a small dog"
    assert len(encoded) == 7


def test_flickr_parser_groups_five_captions(tmp_path: Path):
    captions_file = tmp_path / "Flickr8k.token.txt"
    captions_file.write_text("img1.jpg#0\ta cat\nimg1.jpg#1\ta small cat\nimg2.jpg#0\ta dog\n", encoding="utf-8")
    records = load_flickr8k_captions(captions_file)
    assert records == [ImageCaptions("img1.jpg", ("a cat", "a small cat")), ImageCaptions("img2.jpg", ("a dog",))]


def test_split_is_image_disjoint():
    records = [ImageCaptions(f"img-{index}.jpg", (str(index),)) for index in range(20)]
    splits = split_images(records, validation_fraction=0.2, test_fraction=0.2, seed=7)
    image_sets = [set(record.image for record in split) for split in splits]
    assert not image_sets[0] & image_sets[1]
    assert not image_sets[0] & image_sets[2]
    assert not image_sets[1] & image_sets[2]
