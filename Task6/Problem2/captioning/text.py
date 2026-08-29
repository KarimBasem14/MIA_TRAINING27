"""Caption cleaning, vocabulary, and numericalisation utilities."""

from collections import Counter
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Normalize a caption and return lowercase word tokens."""
    return TOKEN_PATTERN.findall(str(text).lower())


class Vocabulary:
    """A fixed word vocabulary with explicit sequence control tokens."""

    PAD = "<pad>"
    UNK = "<unk>"
    START = "<start>"
    END = "<end>"

    def __init__(self, min_frequency: int = 2, max_size: int | None = 10_000):
        self.min_frequency = min_frequency
        self.max_size = max_size
        self.itos = [self.PAD, self.UNK, self.START, self.END]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def fit(self, captions: list[str]) -> "Vocabulary":
        counts = Counter(token for caption in captions for token in tokenize(caption))
        words = [word for word, count in counts.most_common() if count >= self.min_frequency]
        if self.max_size is not None:
            words = words[: max(0, self.max_size - len(self.itos))]
        for word in words:
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return self

    def encode(self, caption: str, max_length: int) -> list[int]:
        tokens = [self.START, *tokenize(caption), self.END]
        ids = [self.stoi.get(token, self.stoi[self.UNK]) for token in tokens]
        return (ids[:max_length] + [self.stoi[self.PAD]] * max_length)[:max_length]

    def decode(self, ids: list[int]) -> str:
        words = []
        for index in ids:
            token = self.itos[int(index)] if 0 <= int(index) < len(self.itos) else self.UNK
            if token == self.END:
                break
            if token not in {self.PAD, self.START}:
                words.append(token)
        return " ".join(words)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def start_id(self) -> int:
        return self.stoi[self.START]

    @property
    def end_id(self) -> int:
        return self.stoi[self.END]

    def to_dict(self) -> dict:
        return {"min_frequency": self.min_frequency, "max_size": self.max_size, "itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        vocabulary = cls(data["min_frequency"], data["max_size"])
        vocabulary.itos = list(data["itos"])
        vocabulary.stoi = {token: index for index, token in enumerate(vocabulary.itos)}
        return vocabulary

    def __len__(self) -> int:
        return len(self.itos)
