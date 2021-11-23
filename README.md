# BERT-E2E-ABSA
Exploiting **BERT** **E**nd-**t**o-**E**nd **A**spect-**B**ased **S**entiment **A**nalysis
<p align="center">
    <img src="architecture.jpg" height="400"/>
</p>

## Requirements
* python 3.7.3
* pytorch 1.2.0 (also tested on pytorch 1.3.0)
* ~~transformers 2.0.0~~ transformers 4.1.1
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
* ~~Restaurant: retaurant reviews from SemEval 2014 (task 4), SemEval 2015 (task 12) and SemEval 2016 (task 5) (rest_total)~~
* (**IMPORTANT**) Restaurant: restaurant reviews from SemEval 2014 (rest14), restaurant reviews from SemEval 2015 (rest15), restaurant reviews from SemEval 2016 (rest16). Please refer to the newly updated files in ```./data```
* (**IMPORTANT**) **DO NOT** use the ```rest_total``` dataset built by ourselves again, more details can be found in [Updated Results](https://github.com/lixin4ever/BERT-E2E-ABSA/blob/master/README.md#updated-results-important).
* Laptop: laptop reviews from SemEval 2014 (laptop14)


## Quick Start
* The valid tagging strategies/schemes (i.e., the ways representing text or entity span) in this project are **BIEOS** (also called **BIOES** or **BMES**), **BIO** (also called **IOB2**) and **OT** (also called **IO**). If you are not familiar with these terms, I strongly recommend you to read the following materials before running the program: 

  a. [Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). 
  
  b. [Representing Text Chunks](https://www.aclweb.org/anthology/E99-1023.pdf). 
  
  c. The [paper](https://www.aclweb.org/anthology/D19-5505.pdf) associated with this project. 

* Reproduce the results on Restaurant and Laptop dataset:
  ```
  # train the model with 5 different seed numbers
  python fast_run.py 
  ```
* Train the model on other ABSA dataset:
  
  1. place data files in the directory `./data/[YOUR_DATASET_NAME]` (please note that you need to re-organize your data files so that it can be directly adapted to this project, following the input format of `./data/laptop14/train.txt` should be OK).
  
  2. set `TASK_NAME` in `train.sh` as `[YOUR_DATASET_NAME]`.
  
  3. train the model:  `sh train.sh`

* (** **New feature** **) Perform pure inference/direct transfer over test/unseen data using the trained ABSA model:

  1. place data file in the directory `./data/[YOUR_EVAL_DATASET_NAME]`.
  
  2. set `TASK_NAME` in `work.sh` as `[YOUR_EVAL_DATASET_NAME]`
  
  3. set `ABSA_HOME` in `work.sh` as `[HOME_DIRECTORY_OF_PRETRAINED_ABSA_MODEL]`
  
  4. run: `sh work.sh`

## Environment
* OS: REHL Server 6.4 (Santiago)
* GPU: NVIDIA GTX 1080 ti
* CUDA: 10.0
* cuDNN: v7.6.1

## Updated results (IMPORTANT)
* The data files of the ```rest_total``` dataset are created by concatenating the train/test counterparts from ```rest14```, ```rest15``` and ```rest16``` and our motivation is to build a larger training/testing dataset to stabilize the training/faithfully reflect the capability of the ABSA model. However, we recently found that the SemEval organizers directly treat the union set of ```rest15.train``` and ```rest15.test``` as the training set of rest16 (i.e., ```rest16.train```), and thus, there exists overlap between the ```rest_total.train``` and the ```rest_total.test```, which makes this dataset invalid. When you follow our works on this E2E-ABSA task, we hope you **DO NOT** use this ```rest_total``` dataset any more but change to the officially released ```rest14```, ```rest15``` and ```rest16```.
* To facilitate the comparison in the future, we re-run our models following the above mentioned settings and report the results (***micro-averaged F1***) on ```rest14```, ```rest15``` and ```rest16```:  

    | Model | rest14 | rest15 | rest16 |
    | --- | --- | --- | --- |
    | E2E-ABSA (OURS) | 67.10 | 57.27 | 64.31 |
    | [(He et al., 2019)](https://arxiv.org/pdf/1906.06906.pdf) | 69.54 | 59.18 | n/a |
    | [(Liu et al., 2020)](https://arxiv.org/pdf/2004.06427.pdf) | 68.91 | 58.37 | n/a |
    | BERT-Linear (OURS) | 72.61 | 60.29 | 69.67 |
    | BERT-GRU (OURS) | 73.17 | 59.60 | 70.21 |
    | BERT-SAN (OURS) | 73.68 | 59.90 | 70.51 |
    | BERT-TFM (OURS) | 73.98 | 60.24 | 70.25 |
    | BERT-CRF (OURS) | 73.17 | 60.70 | 70.37 |
    | [(Chen and Qian, 2020)](https://www.aclweb.org/anthology/2020.acl-main.340.pdf)| 75.42 | 66.05 | n/a |
    | [(Liang et al., 2020)](https://arxiv.org/pdf/2004.01951.pdf)| 72.60 | 62.37 | n/a |

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
     
