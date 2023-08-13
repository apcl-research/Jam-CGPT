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

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--testfids-file', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-testfid.pkl')
    parser.add_argument('--valfids-file', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-valfid.pkl')
    parser.add_argument('--fundats-file', type=str, default='/nublar/datasets/jm52m/fundats-j1.pkl')
    parser.add_argument('--data-dir', type=str, default='bins/')
    parser.add_argument('--coms-file', type=str, default='/nublar/datasets/jm52m/raw_data/jam-cgpt-raw1.25m.pkl')
    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    testfids_file = args.testfids_file
    valfids_file = args.valfids_file
    fundats_file = args.fundats_file
    coms_file = args.coms_file
    data_dir = args.data_dir

    fundats = pickle.load(open(fundats_file, 'rb'))

    #fundats_fids = list(fundats.keys())


    testfids = pickle.load(open(testfids_file, 'rb'))
    valfids = pickle.load(open(valfids_file, 'rb'))

    coms = pickle.load(open(coms_file, 'rb'))
    fundats_fids = list(coms.keys())
    pt = int(len(fundats_fids) * .02)
    count = 0 
    count_val = 0
    for partnum in range(0, 50):

        print(f'starting part {partnum}')

        txtfiles = list()
        txtfiles_val = list()
        bin_file_path = data_dir + f'/val_2pt_p{partnum}.bin'

        if os.path.isfile(bin_file_path):
            continue

        start_pt = (partnum * pt)
        end_pt = ((partnum+1) * pt)

        fundats_fids_2pt_px = fundats_fids[start_pt:end_pt]

        for fid in tqdm(fundats_fids_2pt_px):

            if fid in testfids:
                continue
            elif fid in valfids:
                with open(f'tmp/{fid}', 'w') as f:
                    f.write('TDAT:\t{fundats[fid]}\nCOM:\t{coms[fid]}' )
                    count_val += 1
                txtfiles_val.append(f'tmp/{fid}')
            else:
                with open(f'tmp/{fid}', 'w') as f:
                    f.write(f'TDAT:\t{fundats[fid]}\nCOM:\t{coms[fid]}' )
                    count += 1
                txtfiles.append(f'tmp/{fid}')
        
        if( txtfiles == []):
            dataset = load_dataset('text', data_files={'val':txtfiles_val}, sample_by="document")
        elif(txtfiles_val ==[]):
            dataset = load_dataset('text', data_files={'train': txtfiles}, sample_by="document")
        elif(txtfiles_val ==[] and txtfiles ==[]):
            continue
        else: 
            dataset = load_dataset('text', data_files={'train': txtfiles, 'val':txtfiles_val}, sample_by="document")
        shmdir = 'tmp/'
        for f in os.listdir(shmdir):
            os.remove(os.path.join(shmdir, f))

        pickle.dump(dataset, open(f'pkls/dataset_funcom_2pt_p{partnum}.pkl', 'wb'))



        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(data_dir, f'{split}_2pt_p{partnum}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()
    
    bincomb.main('bins/')
