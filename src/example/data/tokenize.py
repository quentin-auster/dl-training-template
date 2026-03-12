from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token[i] for i in ids]

    @property
    def size(self) -> int:
        return len(self.id_to_token)


def build_shared_vocab(modulus: int) -> Vocab:
    """Shared vocab used by both modular arithmetic and Dyck tasks.

    Tokens:
      - Special: <PAD>, <BOS>, <EOS>
      - Digits: 0..modulus-1 (as strings)
      - Operators: "+", "=", "(", ")"
    """
    specials = ["<PAD>", "<BOS>", "<EOS>"]
    digits = [str(i) for i in range(modulus)]
    ops = ["+", "=", "(", ")"]

    id_to_token = specials + digits + ops
    token_to_id = {t: i for i, t in enumerate(id_to_token)}

    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=token_to_id["<PAD>"],
        bos_id=token_to_id["<BOS>"],
        eos_id=token_to_id["<EOS>"],
    )


def causal_lm_collate(
    batch: List[tuple[List[int], List[int]]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate for causal LM training.

    Each item is (x_ids, y_ids) of same length.
    Pads to max_len in the batch.

    Returns:
      input_ids:  (B, T)
      target_ids: (B, T) with pad positions set to -100 (ignore_index)
      attn_mask:  (B, T) 1 for real tokens, 0 for pad
    """
    max_len = max(len(x) for x, _ in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((B, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        T = len(x)
        input_ids[i, :T] = torch.tensor(x, dtype=torch.long)
        target_ids[i, :T] = torch.tensor(y, dtype=torch.long)
        attn_mask[i, :T] = 1

    return {"input_ids": input_ids, "target_ids": target_ids, "attn_mask": attn_mask}
