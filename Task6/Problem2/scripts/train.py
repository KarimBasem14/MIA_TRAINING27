"""Train the Flickr8k caption model."""

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captioning.data import ImageCaptions, load_flickr8k_captions, split_images
from captioning.features import ImageFeatureExtractor
from captioning.model import CaptionModel
from captioning.text import Vocabulary


class CaptionDataset(Dataset):
    def __init__(self, records: list[ImageCaptions], image_dir: Path, vocabulary: Vocabulary, feature_cache: Path, extractor: ImageFeatureExtractor, max_length: int):
        self.samples = [(image_dir / record.image, caption) for record in records for caption in record.captions]
        self.vocabulary, self.feature_cache, self.extractor, self.max_length = vocabulary, feature_cache, extractor, max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, caption = self.samples[index]
        return self.extractor.extract_cached(image_path, self.feature_cache), torch.tensor(self.vocabulary.encode(caption, self.max_length))


def collate(batch):
    features, captions = zip(*batch)
    return torch.stack(features), torch.stack(captions)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total = 0.0
    for features, captions in tqdm(loader, leave=False):
        features, captions = features.to(device), captions.to(device)
        with torch.set_grad_enabled(train):
            loss = criterion(model(features, captions).transpose(1, 2), captions[:, 1:])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoint.pt"))
    parser.add_argument("--feature-cache", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=30)
    args = parser.parse_args()

    records = load_flickr8k_captions(args.captions)
    train_records, validation_records, test_records = split_images(records)
    vocabulary = Vocabulary(min_frequency=2, max_size=10_000).fit([caption for record in train_records for caption in record.captions])
    extractor = ImageFeatureExtractor()
    device = extractor.device
    train_set = CaptionDataset(train_records, args.images, vocabulary, args.feature_cache, extractor, args.max_length)
    validation_set = CaptionDataset(validation_records, args.images, vocabulary, args.feature_cache, extractor, args.max_length)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    validation_loader = DataLoader(validation_set, args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    model = CaptionModel(2048, 512, len(vocabulary), vocabulary.pad_id).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    best_loss, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, True)
        validation_loss = run_epoch(model, validation_loader, optimizer, criterion, device, False)
        scheduler.step(validation_loss)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")
        if validation_loss < best_loss:
            best_loss, stale = validation_loss, 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "vocabulary": vocabulary.to_dict(), "feature_dim": 2048, "hidden_dim": 512, "max_length": args.max_length}, args.output)
        else:
            stale += 1
            if stale >= 4:
                print("Early stopping")
                break
    print(f"Saved best checkpoint to {args.output} (test images held out: {len(test_records)})")


if __name__ == "__main__":
    main()
