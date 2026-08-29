"""Small, dependency-light caption metrics."""

from collections import Counter


def _ngrams(tokens: list[str], size: int) -> Counter:
    return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def bleu1(reference: str, prediction: str) -> float:
    reference_tokens, prediction_tokens = reference.split(), prediction.split()
    if not prediction_tokens:
        return 0.0
    matches = sum((_ngrams(reference_tokens, 1) & _ngrams(prediction_tokens, 1)).values())
    precision = matches / len(prediction_tokens)
    brevity = min(1.0, len(prediction_tokens) / max(1, len(reference_tokens)))
    return precision * brevity


def rouge_l(reference: str, prediction: str) -> float:
    reference_tokens, prediction_tokens = reference.split(), prediction.split()
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, 1):
            current.append(previous[index - 1] + 1 if reference_token == prediction_token else max(previous[index], current[-1]))
        previous = current
    return 2 * previous[-1] / max(1, len(reference_tokens) + len(prediction_tokens))


def corpus_scores(references: list[str], predictions: list[str]) -> dict[str, float]:
    if len(references) != len(predictions) or not references:
        raise ValueError("references and predictions must be non-empty and equally sized")
    return {
        "bleu1": sum(bleu1(reference, prediction) for reference, prediction in zip(references, predictions)) / len(references),
        "rougeL": sum(rouge_l(reference, prediction) for reference, prediction in zip(references, predictions)) / len(references),
    }
