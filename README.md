# Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned Datasets
This repository contains the code of Medium-Difficulty Samples and Logit Reshaping (MDSLR) accepted at ICLR'25.

Part of the code is modified from [CRD](https://github.com/HobbitLong/RepDistiller), [PEFD](https://github.com/chenyd7/PEFD) and [InfoBatch](https://github.com/NUS-HPC-AI-Lab/InfoBatch).

## Environment
Python==3.6, pytorch==1.8.0, torchvision==0.2.1

## Datasets
You need to manually download [ImageNet](https://www.image-net.org/download.php) dataset and save it in './data'.

## Achieve the pre-trained teacher networks
sh scripts/run_pretrained_teachers.sh

## Run on CIFAR-100
sh scripts/run_cifar.sh


## Bibtex
@inproceedings{  
chen2025mediumdifficulty,  
title={Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned Datasets},  
author={Yudong Chen and Xuwei Xu and Frank de Hoog and Jiajun Liu and Sen Wang},  
booktitle={The Thirteenth International Conference on Learning Representations},  
year={2025}  
}


