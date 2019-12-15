# Simple Graph Convolution 
Python implement of [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) with tensorflow 2.0.
This repo is modified from [Tiiiger/SGC](https://github.com/Tiiiger/SGC).

# Requirements
+ Python 3.7
+ Tensorflow 2.0
+ numpy
+ scipy
+ sklearn
+ matplotlib

# Datasets
+ cora
+ citeeseer
+ pubmed

Please note that the dataset partition is rather **different** (10% for train, 10% for validation and 80% for test), here we only considering the **largest connected component (LCC)** of the graph for each dataset.
