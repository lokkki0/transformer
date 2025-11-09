#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import argparse
from model import Transformer
from dataset import basic_tokenize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, pad_idx):
    """Greedy decoding loop"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().to(DEVICE)
    for _ in range(max_len - 1):
        tgt_mask = (torch.triu(torch.ones((1, ys.size(1), ys.size(1)), device=DEVICE)) == 1).transpose(1, 2)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=1).item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys


def translate_sentence(model, src_vocab, tgt_vocab, sentence, pad_idx, max_len=60):
    model.eval()
    tokens = basic_tokenize(sentence)
    src_indices = [src_vocab['<bos>']] + [src_vocab.get(tok, src_vocab['<unk>']) for tok in tokens] + [src_vocab['<eos>']]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)
    src_mask = make_src_mask(src_tensor, pad_idx)

    decoded = greedy_decode(model, src_tensor, src_mask, max_len,
                            start_symbol=tgt_vocab['<bos>'],
                            end_symbol=tgt_vocab['<eos>'],
                            pad_idx=pad_idx)

    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    decoded_tokens = [inv_tgt_vocab[idx] for idx in decoded[0].tolist()]
    return " ".join(decoded_tokens[1:-1])  # 去掉 <bos> 和 <eos>


def main():
    parser = argparse.ArgumentParser(description="Translate English → German using Transformer model")
    parser.add_argument("--model", type=str, default="best_transformer.pt", help="path to model checkpoint")
    parser.add_argument("--sentence", type=str, required=True, help="English sentence to translate")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.model} ...")
    ckpt = torch.load(args.model, map_location=DEVICE)

    # restore vocab
    src_vocab = ckpt["src_vocab"]["stoi"]
    tgt_vocab = ckpt["tgt_vocab"]["stoi"]

    # rebuild model
    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=256, N=4, heads=8, d_ff=1024, dropout=0.1)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(DEVICE)
    model.eval()

    pad_idx = src_vocab["<pad>"]

    print("\nTranslating...")
    translation = translate_sentence(model, src_vocab, tgt_vocab, args.sentence, pad_idx)
    print(f"\n🟢 English: {args.sentence}")
    print(f"🔵 German:  {translation}")


if __name__ == "__main__":
    main()
