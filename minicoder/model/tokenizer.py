"""Byte-level BPE tokenizer — compact, lossless, dependency-free.

Vocabulary: byte ids 0..255, special tokens PAD/BOS/EOS/UNK (256..259),
learned merges from 260 on.  Deterministic training: merges are chosen by
(frequency, pair) lexicographic order so runs are reproducible.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Iterable

PAD, BOS, EOS, UNK = 256, 257, 258, 259
FIRST_MERGE = 260
SPECIALS = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", UNK: "<unk>"}


class ByteBPETokenizer:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []          # in learned order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for tok, s in SPECIALS.items():
            self.vocab[tok] = s.encode()

    # ------------------------------------------------------------- training
    def train(self, texts: Iterable[str], target_vocab: int = 800) -> None:
        seqs = [list(t.encode("utf-8")) for t in texts]
        next_id = FIRST_MERGE
        self.merges = []
        while len(self.vocab) < target_vocab:
            counts: Counter = Counter()
            for seq in seqs:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            best, n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if n < 2:
                break
            a, b = best
            new_seqs = []
            for seq in seqs:
                merged: list[int] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        merged.append(next_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                new_seqs.append(merged)
            seqs = new_seqs
            self.merges.append((a, b))
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            next_id += 1

    # ------------------------------------------------------------- encoding
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        for rank, (a, b) in enumerate(self.merges):
            new_id = FIRST_MERGE + rank
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        parts: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in SPECIALS:
                if not skip_special:
                    parts.append(self.vocab[i])
                continue
            parts.append(self.vocab.get(i, b""))
        return b"".join(parts).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE + len(self.merges)

    # ------------------------------------------------------------- storage
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"merges": self.merges}, fh)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tok.merges = [tuple(m) for m in data["merges"]]
        for rank, (a, b) in enumerate(tok.merges):
            tok.vocab[FIRST_MERGE + rank] = tok.vocab[a] + tok.vocab[b]
        return tok
