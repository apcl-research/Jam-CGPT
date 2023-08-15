# Jam-CGPT: Distilled GPT for Source Code Summarization

## Code for Distilled GPT for Source Code Summarization

Proposed by:
- [Chia-Yi Su](https://chiayisu.github.io/)
- [Collin McMillan](https://sdf.org/~cmc/)

This repository contains all the code and detailed instructions to rebuild [Jam-CGPT](https://huggingface.co/apcl/Jam-CGPT) models in our HuggingFace [Automatic Program Comprehension Lab](https://huggingface.co/apcl) hub.

## Quick link
- [To-do list](#to-do-list)
- [Finetuning](#finetuning)
- [Inference](#inference)
- [Metrics](#metrics)
- [Dataset](#dataset)


## To-do list

To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiements.
```
pip install -r requirements.txt
``` 

- **If you only want to make an inference with our models**, please see [Inference](#inference).
- If you want to finetune a model using our processed and tokenized dataset, please see [finetuning](#finetuning)

## Finetuning 
These steps will show you how to fine-tune Jam-CGPT in our paper.

### Step 1: Download the finetuning dataset

You can download all of the datasets in our paper in our [Hugginface repo](https://huggingface.co/datasets/apcl/Jam-CGPT/tree/main). Please put ``train.bin`` and  ``val.bin`` to the same ``dir`` as ``--dataset`` in ``config/finetune_model_350m_dataset_170k.py``. 

### Step 2: Download the models for finetuning

Please download the checkpoint files named ``ckpt_pretrain.pt`` in our [Hugginface repo](https://huggingface.co/apcl/Jam-CGPT/tree/main) for finetuning and put the checkpoint to the same  ``dir`` as ``--out_dir`` in ``config/finetune_model_350m_dataset_170k.py``

### Step 3: Finetuning model

```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=1 train.py config/finetune_model_350m_dataset_170k.py 
``` 
- In ``config`` dir, there are many different config files for different models and datasets. You can change ``config/finetune_model_350m_dataset_170k.py`` to the different ``config`` file in ``./config`` dir.


## Inference
After you download the test set named ``jam_cgpt_test.tar.gz`` in our [Hugginface repo](https://huggingface.co/datasets/apcl/Jam-CGPT/tree/main), you can simiply run command below for inference.

```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=1 sample_funcom_gpt.py config/finetune_model_350m_dataset_170k.py  --prediction_filename=predict_data170k_model350m.txt
```
    --outdir: directory of the model that you want to use for inference
    --prediction_filename: prediction file name 
    --outfilename: checkpoint file name
    
Note that you need to download checkpoint files our [Hugginface repo](https://huggingface.co/apcl/Jam-CGPT/tree/main) and put the checkpoint files to the same  ``dir`` as ``--out_dir`` in ``config/finetune_model_350m_dataset_170k.py`` if you just want to make an inference with our models.

## Metrics
We provide scripts for calculating the metrics that we report on the paper. The following commands are for METEOR, and USE score respectively.
```
python3 meteor.py jam_cgpt_predictions/predict_data170k_model350m.txt --coms-filename=cgptcom.test --data=./data/jam_cgpt_170k
```
```
python3 use_score_v.py jam_cgpt_predictions/predict_170k_100mparameters.txt --gpu=0 --coms-filename=cgptcom.test --data=./data/jam_cgpt_170k
```

## Dataset

We also release all of our raw datasets for the experiements in our [Hugginface repo](https://huggingface.co/datasets/apcl/Jam-CGPT/tree/main) and the scripts for compiling the raw data to ``bin`` files in this Github repo. Before running the command, please create three dir: ``pkls``, ``bins``, and ``tmp``. Then, you can simply run the following command to generate ``train.bin`` and ``val.bin``.

```
python3 data/jam_cgpt_170k/prepare_fc_raw.py
```
- Note that you will need to put ``jam-cgpt-testfid.pkl``, ``jam-cgpt-valfid.pkl``, ``fundats-j1.pkl``, ``jam-cgpt-raw1.25m.pkl``, ``jam-cgpt-raw170k.pkl``, ``jam-cgpt-raw2.15m.pkl``, and ``jam-cgpt-raw620k.pkl`` to /nublar/datasets/jm52m/raw_data or you will need to change the parameters in the script.
- Related parameters are as follows:
  
      --testfids-file: file lcation of function id on testset
      --valfids-file: file location of function id on valset
      --fundats-file: file location of function
      --coms-file: file location of comments
  

