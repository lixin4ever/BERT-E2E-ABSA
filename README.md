# BERT-E2E-ABSA
Exploiting **BERT** **E**nd-**t**o-**E**nd **A**spect-**B**ased **S**entiment **A**nalysis
<p align="center">
    <img src="architecture.jpg" height="400"/>
</p>

## Requirements
* python 3.7.3
* pytorch 1.2.0
* transformers 2.0.0
* numpy 1.16.4
* tensorboardX 1.9
* tqdm 4.32.1
* some codes are borrowed from **allennlp** ([https://github.com/allenai/allennlp](https://github.com/allenai/allennlp), an awesome open-source NLP toolkit) and **transformers** ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers), formerly known as **pytorch-pretrained-bert** or **pytorch-transformers**)

## Architecture
* Pre-trained embedding layer: BERT-Base-Uncased (12-layer, 768-hidden, 12-heads, 110M parameters)
* Task-specific layer: 
  - Linear
  - Recurrent Neural Networks (GRU)
  - Self-Attention Networks (SAN, TFM)
  - Conditional Random Fields (CRF)

## Dataset
* Restaurant: retaurant reviews from SemEval 2014 (task 4), SemEval 2015 (task 12) and SemEval 2016 (task 5)
* Laptop: laptop reviews from SemEval 2014


## Quick Start
* Reproduce the results on Restaurant and Laptop dataset:
  ```
  # train the model with 5 different seed numbers
  python fast_run.py 
  ```
* Train the model on other ABSA dataset:
  
  1. place data files in the directory `./data/[YOUR_DATASET_NAME]`.
  
  2. set `TASK_NAME` in `train.sh` as `[YOUR_DATASET_NAME]`.
  
  3. train the model:  `sh train.sh`

## Environment
* OS: REHL Server 6.4 (Santiago)
* GPU: NVIDIA GTX 1080 ti
* CUDA: 10.0
* cuDNN: v7.6.1


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@inproceedings{li-etal-2019-exploiting,
    title = "Exploiting {BERT} for End-to-End Aspect-based Sentiment Analysis",
    author = "Li, Xin  and
      Bing, Lidong  and
      Zhang, Wenxuan  and
      Lam, Wai",
    booktitle = "Proceedings of the 5th Workshop on Noisy User-generated Text (W-NUT 2019)",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-5505",
    pages = "34--41"
}
```
     
