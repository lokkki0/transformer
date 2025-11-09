import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Vocab:
    def __init__(self, min_freq: int = 2, specials: Optional[List[str]] = None):
        if specials is None:
            specials = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.freqs = {}
        self.itos = []
        self.stoi = {}
        self.min_freq = min_freq
        self.specials = specials

    def build(self, texts: List[List[str]]):
        for tokens in texts:
            for tok in tokens:
                self.freqs[tok] = self.freqs.get(tok, 0) + 1
        # special tokens first
        self.itos = list(self.specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        for tok, freq in sorted(self.freqs.items(), key=lambda x: (-x[1], x[0])):
            if freq >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi.get('<unk>', 1)
        return [self.stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if i < len(self.itos) else '<unk>' for i in ids]


class TranslationDataset(Dataset):
    def __init__(self, src_lines: List[List[str]], tgt_lines: List[List[str]], src_vocab: Vocab, tgt_vocab: Vocab):
        assert len(src_lines) == len(tgt_lines), "Mismatched source/target line counts"
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = self.src_lines[idx]
        tgt_tokens = self.tgt_lines[idx]
        src_ids = [self.src_vocab.stoi['<bos>']] + self.src_vocab.encode(src_tokens) + [self.src_vocab.stoi['<eos>']]
        tgt_ids = [self.tgt_vocab.stoi['<bos>']] + self.tgt_vocab.encode(tgt_tokens) + [self.tgt_vocab.stoi['<eos>']]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# --------------------
# Helpers
# --------------------

def load_text_file(path: str) -> List[str]:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # try to extract text between <seg> tags if present
            m = re.search(r'<seg[^>]*>(.*?)</seg>', line)
            if m:
                line = m.group(1)
            elif line.startswith('<'):
                # skip XML meta lines
                continue
            lines.append(line)
    return lines


def basic_tokenize(text: str) -> List[str]:
    # simple whitespace + punctuation split
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9äöüßÄÖÜ.,!?;:'\-]+", ' ', text)
    return text.strip().split()


def prepare_dataset(src_file: str,
                    tgt_file: str,
                    dev_src: Optional[str] = None,
                    dev_tgt: Optional[str] = None,
                    min_freq: int = 2) -> Tuple[TranslationDataset, Optional[TranslationDataset], Vocab, Vocab]:

    print("Loading training data...")
    src_texts = load_text_file(src_file)
    tgt_texts = load_text_file(tgt_file)

    src_tokens = [basic_tokenize(line) for line in tqdm(src_texts, desc="Tokenizing src")]
    tgt_tokens = [basic_tokenize(line) for line in tqdm(tgt_texts, desc="Tokenizing tgt")]

    print("Building vocabularies...")
    src_vocab = Vocab(min_freq=min_freq)
    tgt_vocab = Vocab(min_freq=min_freq)
    src_vocab.build(src_tokens)
    tgt_vocab.build(tgt_tokens)

    train_dataset = TranslationDataset(src_tokens, tgt_tokens, src_vocab, tgt_vocab)

    dev_dataset = None
    if dev_src and dev_tgt:
        print("Loading dev data...")
        dev_src_texts = load_text_file(dev_src)
        dev_tgt_texts = load_text_file(dev_tgt)
        dev_src_tokens = [basic_tokenize(line) for line in dev_src_texts]
        dev_tgt_tokens = [basic_tokenize(line) for line in dev_tgt_texts]
        dev_dataset = TranslationDataset(dev_src_tokens, dev_tgt_tokens, src_vocab, tgt_vocab)

    return train_dataset, dev_dataset, src_vocab, tgt_vocab


# --------------------
# Collate for DataLoader
# --------------------

def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt

    return src_padded, tgt_padded


if __name__ == "__main__":
    # Example usage
    SRC_FILE = "en-de/train.tags.en-de.en"
    TGT_FILE = "en-de/train.tags.en-de.de"
    DEV_SRC = "en-de/IWSLT17.TED.dev2010.en-de.en.xml"
    DEV_TGT = "en-de/IWSLT17.TED.dev2010.en-de.de.xml"

    train_data, dev_data, src_vocab, tgt_vocab = prepare_dataset(SRC_FILE, TGT_FILE, DEV_SRC, DEV_TGT)
    print(f"Train size: {len(train_data)}, Dev size: {len(dev_data) if dev_data else 0}")
    print(f"Src vocab: {len(src_vocab)}, Tgt vocab: {len(tgt_vocab)}")

    from torch.utils.data import DataLoader
    loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for src, tgt in loader:
        print(src.shape, tgt.shape)
        break
