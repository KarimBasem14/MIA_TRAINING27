# Flickr8k Image Caption Generator

Production-oriented reference implementation for Task 6. The system separates data preparation, CNN feature extraction, text processing, model training, evaluation, and serving.

## Project layout

```text
captioning/       reusable Python package
scripts/          command-line training and evaluation entry points
tests/            focused unit tests
app.py            Gradio upload interface
data/             local Flickr8k files (not committed)
artifacts/        cached features and best checkpoints (not committed)
```

## Setup

From `Task6/Problem2`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download Flickr8k and place `Flickr8k_Dataset` and `Flickr8k.token.txt` in `data/`. The split is performed on unique image names before captions are expanded, which prevents reference-caption leakage.

## Train and evaluate

```powershell
python scripts/train.py --images data/Flickr8k_Dataset --captions data/Flickr8k.token.txt --output artifacts/checkpoint.pt --epochs 20
python scripts/evaluate.py --checkpoint artifacts/checkpoint.pt --images data/Flickr8k_Dataset --captions data/Flickr8k.token.txt
```

The trainer uses a pretrained ResNet-50, caches its spatial features, trains an attention LSTM with masked cross-entropy, early stopping, a learning-rate scheduler, and best-checkpoint restoration. BLEU-1 and ROUGE-L are reported alongside qualitative examples.

## Serve

```powershell
python app.py --checkpoint artifacts/checkpoint.pt
```

Open the printed local Gradio URL and upload an image. A checkpoint is required for inference; the app gives a clear startup error when it is missing.
