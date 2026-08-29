"""Gradio image captioning application."""

import argparse
from pathlib import Path

import torch

from captioning.features import ImageFeatureExtractor
from captioning.model import CaptionModel
from captioning.text import Vocabulary


def create_app(checkpoint_path: Path):
    import gradio as gr

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
    extractor = ImageFeatureExtractor()
    model = CaptionModel(checkpoint["feature_dim"], checkpoint["hidden_dim"], len(vocabulary), vocabulary.pad_id)
    model.load_state_dict(checkpoint["model"])
    model.to(extractor.device).eval()

    def caption(image):
        if image is None:
            return "Please upload an image."
        feature = extractor.extract(image).unsqueeze(0).to(extractor.device)
        ids = model.generate(feature, vocabulary.start_id, vocabulary.end_id, checkpoint["max_length"])[0]
        return vocabulary.decode(ids[1:]) or "No caption generated."

    return gr.Interface(fn=caption, inputs=gr.Image(type="filepath", label="Image"), outputs=gr.Textbox(label="Generated caption"), title="Flickr8k Image Caption Generator")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train the model first.")
    create_app(args.checkpoint).launch()
