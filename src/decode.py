"""Greedy, beam and nucleus caption generation from cached image features."""

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn


def decode(
    model: nn.Module,
    features: Tensor,
    method: str = "greedy",
    beam_size: int = 3,
    *,
    vocab=None,
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    """Decode one image using a Vocabulary object or its ``itos`` list.

    Pass vocabulary explicitly: models do not store it. Special IDs are
    resolved from the supplied words, never from assumed numeric positions.
    For validation batches, pass ``features[i:i + 1]`` for each image.
    """
    if vocab is None:
        raise ValueError("decode requires vocab=Vocabulary or vocab=itos_list; models do not store vocabulary")
    words = vocab if isinstance(vocab, list) else getattr(vocab, "itos", None)
    if not isinstance(words, list) or not words or any(not isinstance(word, str) for word in words):
        raise ValueError("vocab must be a Vocabulary with an itos list, or a nonempty list of words")
    if len(set(words)) != len(words):
        raise ValueError("Vocabulary words must be unique")
    ids = {word: index for index, word in enumerate(words)}
    missing = [token for token in ("<start>", "<end>", "<pad>") if token not in ids]
    if missing:
        raise ValueError(f"Vocabulary is missing required special tokens: {', '.join(missing)}")
    return generate_caption(
        model, features, dict(enumerate(words)),
        start_id=ids["<start>"], end_id=ids["<end>"], pad_id=ids["<pad>"],
        method=method, beam_size=beam_size, top_p=top_p, seed=seed,
    )


def generate_caption(
    model: nn.Module,
    features: Tensor,
    id_to_word: Mapping[int, str],
    *,
    start_id: int,
    end_id: int,
    pad_id: int,
    method: str = "greedy",
    return_attention: bool = False,
    beam_size: int = 3,
    top_p: float | None = None,
    seed: int | None = None,
) -> str | tuple[str, Tensor | None]:
    """Generate one caption with at most 34 tokens including the start token.

    Features must be [1, 2048, 7, 7] or [1, 49, 2048], on the model's
    device. The vocabulary maps integer token IDs to words. Start and padding
    IDs are masked before selection in every method. Greedy uses argmax;
    beam ranks sequences by cumulative log probability without length
    normalization. Nucleus requires an explicit top_p in (0, 1] and uses a
    local CPU random generator (seed controls reproducibility). Special
    tokens are omitted from the returned text,
    but still count toward the sequence limit. End stops generation.

    With return_attention=True, return (caption, weights), where weights
    is a detached CPU tensor [number_of_output_words, 49]. An empty caption
    has weights shaped [0, 49]. Each row belongs to the corresponding word.
    If the model supplies no attention, weights is None instead.
    The model's existing forward interface is used by replaying the prefix.
    """
    if method not in ("greedy", "beam", "nucleus"):
        raise ValueError(f"Unsupported decoding method {method!r}; choose 'greedy', 'beam' or 'nucleus'.")
    if method == "beam" and (not isinstance(beam_size, int) or isinstance(beam_size, bool) or beam_size < 1):
        raise ValueError("beam_size must be a positive integer")
    if method == "nucleus":
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError("nucleus requires top_p in the interval (0, 1]")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or None")
        if seed is not None and not -(2**63) <= seed <= 2**64 - 1:
            raise ValueError("seed must be between -2**63 and 2**64 - 1")
    if tuple(features.shape) not in ((1, 2048, 7, 7), (1, 49, 2048)):
        raise ValueError("features must have shape [1, 2048, 7, 7] or [1, 49, 2048]")
    token_ids = (start_id, end_id, pad_id)
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids):
        raise ValueError("start_id, end_id and pad_id must be nonnegative integer IDs")
    if len(set(token_ids)) != 3:
        raise ValueError("start_id, end_id and pad_id must be distinct")
    if any(token not in id_to_word for token in token_ids):
        raise ValueError("The vocabulary must contain the start, end and padding IDs")

    prefix = torch.tensor([[start_id]], dtype=torch.long, device=features.device)
    words, attention_rows = [], []
    attention_available = True
    special_words = {"<start>", "<end>", "<pad>", "<unk>"}

    def predict(tokens):
        nonlocal attention_available
        logits, attention = model(features, tokens)
        if any(token >= logits.shape[-1] for token in token_ids):
            raise ValueError("Special token IDs exceed the model vocabulary size")
        step_logits = logits[0, -1].clone()
        step_logits[start_id] = -torch.inf
        step_logits[pad_id] = -torch.inf
        if torch.isnan(step_logits).any() or torch.isposinf(step_logits).any():
            raise ValueError("Model returned invalid logits (NaN or positive infinity)")
        if not torch.isfinite(step_logits).any():
            raise ValueError("No selectable token remains after masking start and padding")
        if attention is None:
            attention_available = False
        weights = attention[0, -1].detach().cpu() if return_attention and attention is not None else None
        return step_logits, weights

    def append_word(next_id, weights, current_words, current_rows):
        if next_id == end_id:
            return current_words, current_rows
        if next_id not in id_to_word:
            raise ValueError(f"Vocabulary has no word for generated token ID {next_id}")
        word = id_to_word[next_id]
        if next_id not in (start_id, pad_id) and word not in special_words:
            return current_words + [word], current_rows + [weights]
        return current_words, current_rows

    generator = None
    if method == "nucleus":
        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
    # Preserve even intentionally mixed training/evaluation submodule states.
    previous_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            if method == "beam" and beam_size > 1:
                # Each hypothesis carries its own prefix, score and word-aligned
                # attention. Completed hypotheses remain eligible without expansion.
                beams = [(prefix, 0.0, [], [], False)]
                completed = []
                for _ in range(33):
                    candidates = []
                    for tokens, score, beam_words, rows, ended in beams:
                        if ended:
                            candidates.append((tokens, score, beam_words, rows, True))
                            continue
                        step_logits, weights = predict(tokens)
                        log_probs = torch.log_softmax(step_logits, dim=-1)
                        scores, indices = log_probs.topk(min(beam_size, log_probs.numel()))
                        for value, index in zip(scores.tolist(), indices.tolist()):
                            if not math.isfinite(value):
                                continue
                            new_words, new_rows = append_word(index, weights, beam_words, rows)
                            candidate = (
                                torch.cat((tokens, tokens.new_tensor([[index]])), dim=1),
                                score + value, new_words, new_rows, index == end_id,
                            )
                            candidates.append(candidate)
                            if index == end_id:
                                completed.append(candidate)
                    beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]
                    if all(item[4] for item in beams):
                        break
                best = max(beams + completed, key=lambda item: item[1])
                words, attention_rows = best[2], best[3]
            else:
                # Beam width one follows precisely the existing greedy path,
                # including argmax tie handling.
                for _ in range(33):
                    step_logits, weights = predict(prefix)
                    if method == "nucleus":
                        probabilities, indices = torch.softmax(step_logits, dim=-1).cpu().sort(descending=True)
                        # Keep the smallest prefix whose mass reaches top_p,
                        # including the token that crosses the threshold.
                        cumulative = probabilities.cumsum(0)
                        remove = torch.zeros_like(probabilities, dtype=torch.bool)
                        remove[1:] = cumulative[:-1] >= top_p
                        probabilities = probabilities.masked_fill(remove, 0)
                        probabilities = probabilities / probabilities.sum()
                        choice = torch.multinomial(probabilities, 1, generator=generator)
                        next_id = int(indices[choice].item())
                    else:
                        next_id = int(step_logits.argmax().item())
                    if next_id == end_id:
                        break
                    words, attention_rows = append_word(next_id, weights, words, attention_rows)
                    prefix = torch.cat((prefix, prefix.new_tensor([[next_id]])), dim=1)
    finally:
        for module, training in previous_modes:
            module.training = training

    caption = " ".join(words)
    if return_attention:
        if not attention_available:
            return caption, None
        weights = torch.stack(attention_rows) if attention_rows else features.new_empty((0, 49)).cpu()
        return caption, weights
    return caption
