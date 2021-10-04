""" VecLex utils"""
import os,glob
import argparse
import threading
import subprocess as sp
from collections import Counter, deque
from cytoolz import concat, curry

import pandas as pd
import numpy as np
import path_config

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DataUtils
import torchvision.datasets as TorchDatasets
from torch.utils.data import sampler

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity


USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# Vector similarity calculation
word2vecModel = KeyedVectors.load_word2vec_format(path_config.word2vec_filepath, binary=True)

def avg_embedding (wordlist, wordmodel, size):
    s_vec = np.full(size, 0.0001)
    for word in wordlist:
        if word in wordmodel.vocab:
            s_vec = s_vec + wordmodel[word]
    if len(wordlist)==0:
        return s_vec
    else:
        return s_vec/len(wordlist)

def min_embedding (wordlist, wordmodel, size):
    s_vec = np.full(size, 100)
    for word in wordlist:
        if word in wordmodel.vocab:
            s_vec = np.minimum(s_vec, wordmodel[word])
    return s_vec

def max_embedding (wordlist, wordmodel, size):
    s_vec = np.full(size, -100)
    for word in wordlist:
        if word in wordmodel.vocab:
            s_vec = np.maximum(s_vec, wordmodel[word])
    return s_vec

def combined_embedding (wordlist, wordmodel, size):
    min_embed = min_embedding(wordlist, wordmodel, size)
    avg_embed = avg_embedding(wordlist, wordmodel, size)
    max_embed = max_embedding(wordlist, wordmodel, size)
    s_vec = np.append(min_embed, np.append(avg_embed, max_embed))
    return s_vec

def get_embedding(text):
    embed_size = 300
    text = [w.lower() for w in text]
    embedding = combined_embedding(text, word2vecModel, embed_size)
    return embedding


def compute_vector_similarity(output, reference):

    a = get_embedding(output)
    b = get_embedding(reference)

    aa = a.reshape(1,900)
    ba = b.reshape(1,900)

    cos_lib = cosine_similarity(aa, ba)
    return cos_lib[0][0]




# Lexicon matching score calculation
lex_file = open(path_config.lexicon_filepath, "r")
all_lines = lex_file.readlines()
crisis_wordlist = [row.replace('\n', '') for row in all_lines]
cw_unigram = [row for row in crisis_wordlist if row.find(' ')==-1]
cw_bigram = [row for row in crisis_wordlist if row.find(' ')!=-1]

bigram_wordlist = [word for bigram in cw_bigram for word in bigram.split()]
frequency_counts = Counter(bigram_wordlist)
for word in frequency_counts:
    if(frequency_counts[word] >= 5):    
        if(word not in cw_unigram):
            cw_unigram.append(word)

def make_n_grams(seq, n):
    """ return iterator """
    ngrams = (tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return ngrams

def _n_gram_match(summ, ref, n):
    summ_grams = Counter(make_n_grams(summ, n))
    ref_grams = Counter(make_n_grams(ref, n))
    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count


def compute_rouge_with_unigram_crisis_lexicon(output):
    """unigram lexicon matching score for a summary and crisis lexicon"""
    mode = 'p'
    assert mode in list('fpr')  # F-1, precision, recall
    reference = cw_unigram
    match = _n_gram_match(reference, output, 1)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


def _bi_gram_match(summ, ref):
    summ_grams = ( ' '.join(summ[i:i+2]) for i in range(len(summ)-2+1))
    summ_grams = Counter(summ_grams)

    ref_grams = (ref[i] for i in range(len(ref)))
    ref_grams = Counter(ref_grams)

    grams = min(summ_grams, ref_grams, key=len)
    count = sum(min(summ_grams[g], ref_grams[g]) for g in grams)
    return count

@curry
def compute_rouge_with_bigram_crisis_lexicon(output):
    """bigram lexicon matching score for a summary and crisis lexicon"""
    mode = 'p'
    assert mode in list('fpr')  # F-1, precision, recall
    reference = cw_bigram
    match = _bi_gram_match(output, reference)
    if match == 0:
        score = 0.0
    else:
        precision = match / len(output)
        recall = match / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        elif mode == 'r':
            score = recall
        else:
            score = f_score
    return score


@curry
def compute_veclex_with_unigram_lex(output, reference):
    LEX_COEFF = 40
    vector_similarity = compute_vector_similarity(output, reference)
    rouge_with_lexicon = compute_rouge_with_unigram_crisis_lexicon(output)
    score = (vector_similarity + (LEX_COEFF * rouge_with_lexicon))/2
    return score


@curry
def compute_veclex_with_bigram_lex(output, reference):
    LEX_COEFF = 40
    vector_similarity = compute_vector_similarity(output, reference)
    rouge_with_lexicon = compute_rouge_with_bigram_crisis_lexicon(output)
    score = (vector_similarity + (LEX_COEFF * rouge_with_lexicon))/2
    return score


@curry
def compute_veclex(output, reference):
    # We suggest to choose the value of lexicon mactching score co-efficient such that
    # the weight of the vector similarity score and the lexicon matching score become 
    # equal. That means one should not dominate over other.
    LEX_COEFF = 40
    vector_similarity = compute_vector_similarity(output, reference)
    rouge_with_lexicon_unigram = compute_rouge_with_unigram_crisis_lexicon(output)
    rouge_with_lexicon_bigram = compute_rouge_with_bigram_crisis_lexicon(output)
    rouge_with_lexicon = rouge_with_lexicon_unigram + rouge_with_lexicon_bigram
    
    score = (vector_similarity + (LEX_COEFF * rouge_with_lexicon))/2
    return score