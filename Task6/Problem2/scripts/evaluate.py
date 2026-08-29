"""Evaluate a trained checkpoint on the image-level test split."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captioning.data import load_flickr8k_captions, split_images
from captioning.features import ImageFeatureExtractor
from captioning.metrics import corpus_scores
from captioning.model import CaptionModel
from captioning.text import Vocabulary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
    extractor = ImageFeatureExtractor()
    model = CaptionModel(checkpoint["feature_dim"], checkpoint["hidden_dim"], len(vocabulary), vocabulary.pad_id)
    model.load_state_dict(checkpoint["model"])
    model.to(extractor.device).eval()
    _, _, test_records = split_images(load_flickr8k_captions(args.captions))
    references, predictions = [], []
    for record in test_records[: args.limit or None]:
        feature = extractor.extract(args.images / record.image).unsqueeze(0).to(extractor.device)
        prediction = vocabulary.decode(model.generate(feature, vocabulary.start_id, vocabulary.end_id, checkpoint["max_length"])[0][1:])
        references.append(record.captions[0])
        predictions.append(prediction)
        print(f"{record.image}\n  generated: {prediction}\n  reference: {record.captions[0]}")
    print(corpus_scores(references, predictions))


if __name__ == "__main__":
    main()
