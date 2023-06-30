from tqdm import tqdm
from detoken_constants import *
from custom_data import *
from transformer import *
from data_structure import *
from torch import nn

import torch
import sys, os
import numpy as np
import argparse
import datetime
import copy
import heapq
import sentencepiece as spm
import json
import argparse
import pickle
from nltk.translate.bleu_score import corpus_bleu

from main import Manager
import wandb

def eval(manager, input_file, label_file, method):
    print("Loading sentencepiece tokenizer...")
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
    trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

    with open(input_file) as f:
        input_txt = f.readlines()
    results = []
    for input_sentence in tqdm(input_txt):
        # print("Loading sentencepiece tokenizer...")
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

        # print("Preprocessing input sentence...")
        tokenized = src_sp.EncodeAsIds(input_sentence)

        # with open("tokenized_test.pickle", "w+") as f:
        #     pickle.dump(tokenized, f)
    
        # with open("tokenized_test.pickle", "r") as f:
        #     tokenized = pickle.load(f)

        src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device) # (1, L)
        e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

        start_time = datetime.datetime.now()

        # print("Encoding input sentence...")
        src_data = manager.model.encoder.embed_tokens(src_data)
        src_data = manager.model.encoder.embed_positions(src_data)
        e_output = manager.model.encoder(src_data, e_mask) # (1, L, d_model)


        result = manager.beam_search(e_output, e_mask, trg_sp)
        results.append(result)
    # with open(input_file, "r") as f:
    #     results=f.readlines()
    with open(label_file, "r") as f:
        reference_labels = f.readlines()

    # Preprocess the reference labels
    reference_labels = [label.strip().split() for label in reference_labels]

    # Preprocess the generated results
    generated_results = [result.split() for result in results]


    # Calculate the BLEU score
    bleu_score = corpus_bleu([[label] for label in reference_labels], generated_results)

    print("BLEU Score:", bleu_score)

    with open("output_txt.txt", "w+") as f:
        for line in results:
            f.writelines(line+"\n")


if __name__ == "__main__":
    wandb.init(mode="disabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-I", default="/home2/khanhnd/transformer-mt/dev_1.vi", help="Input text file (src language).")
    parser.add_argument("--label_file", "-L", default="/home2/khanhnd/transformer-mt/dev_1.en", help="Label text file (trg language).")
    parser.add_argument("--decode", default="beam")
    parser.add_argument("--ckpt", default="checkpoint_best.pt")

    args = parser.parse_args()

    model = Manager(is_train=False, ckpt_name="checkpoint_best.pt")
    # print(model)
    eval(model, args.input_file, args.label_file, method=args.decode)


    