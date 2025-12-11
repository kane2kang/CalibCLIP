#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text tokenizer for CLIP.
"""

import gzip
import html
import os
import hashlib
import urllib.request
from functools import lru_cache
from typing import Union, List

import ftfy
import regex as re
import torch


@lru_cache()
def default_bpe():
    """Get default BPE vocabulary path."""
    url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    cache_dir = os.path.expanduser("~/.cache/calibclip")
    os.makedirs(cache_dir, exist_ok=True)

    filename = hashlib.md5(url.encode()).hexdigest() + ".txt.gz"
    filepath = os.path.join(cache_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading BPE vocabulary to {filepath}")
        urllib.request.urlretrieve(url, filepath)

    return filepath


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """Basic text cleaning."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Clean whitespace."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer:
    """
    Simple BPE tokenizer for CLIP.
    """

    def __init__(self, bpe_path: str = None):
        bpe_path = bpe_path or default_bpe()

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            merges = f.read().split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE
        )

        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()

        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
        text = text.replace("</w>", " ")
        return text

    def __call__(
            self,
            texts: Union[str, List[str]],
            context_length: int = 77,
            max_length: int = None,  # 添加别名支持
    ) -> torch.Tensor:
        """
        Tokenize texts.
        Args:
            texts: Input text(s)
            context_length: Maximum context length
            max_length: Alias for context_length (for compatibility)
        Returns:
            tokens: Tokenized tensor [B, context_length] or [context_length] if single text
        """
        # Support max_length as alias
        if max_length is not None:
            context_length = max_length

        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        all_tokens = []
        for text in texts:
            tokens = [self.sot_token] + self.encode(text) + [self.eot_token]
            tokens = tokens[:context_length]
            tokens = tokens + [0] * (context_length - len(tokens))
            all_tokens.append(tokens)
        result = torch.tensor(all_tokens, dtype=torch.long)

        # Return 1D tensor for single text input
        if single_text:
            return result.squeeze(0)
        return result


def tokenize(
        texts: Union[str, List[str]],
        context_length: int = 77,
        tokenizer: SimpleTokenizer = None,
) -> torch.Tensor:
    """
    Tokenize texts using default tokenizer.

    Args:
        texts: Input text(s)
        context_length: Maximum context length
        tokenizer: Tokenizer instance (creates new if None)

    Returns:
        tokens: Tokenized tensor
    """
    if tokenizer is None:
        tokenizer = SimpleTokenizer()
    return tokenizer(texts, context_length)
