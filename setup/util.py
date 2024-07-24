#!/usr/bin/env python3

import json
import os
import sys
import numpy as np

def generate_random_embeddings(d=64, num_embs=100):
    '''
    Load the embeddings from pickle files
    '''
    xb = np.random.random((num_embs, d)).astype('float32')
    xb[:, 0] += np.arange(num_embs) / 1000.
    return xb