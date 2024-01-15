# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from datasets import Dataset

import pickle
import random
import argparse
import bincomb
import os

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--testfids-file', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-testfid.pkl')
    parser.add_argument('--fundats-file', type=str, default='/nublar/datasets/jm52m/q90fundats-j1.pkl')
    parser.add_argument('--coms-file', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-raw170k.pkl')
    parser.add_argument('--data-dir', type=str, default='funcom_test/')

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    testfids_file = args.testfids_file
    fundats_file = args.fundats_file
    coms_file = args.coms_file
    data_dir = args.data_dir

    fundats = pickle.load(open(fundats_file, 'rb'))

    fundats_fids = list(fundats.keys())

    pt = int(len(fundats_fids) * .02)


    testfids = pickle.load(open(testfids_file, 'rb'))

    coms = pickle.load(open(coms_file, 'rb'))
    count = 0   
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    for fid in tqdm(testfids):
        try:
            with open(f'{data_dir}{fid}.txt', 'w') as f:
                f.write(f'TDAT:\t{fundats[fid]}\nCOM:\t{coms[fid]}' )
                count += 1
        except KeyError:
            continue
