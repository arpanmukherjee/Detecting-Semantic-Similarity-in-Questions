
# Detecting Semantic Similarity in Questions


## Quick Links
- [About](#problem-statement)
- [Dataset](#dataset)
- [Setup](#setup)
- [Algorithms](#architectures-used)
- [Results](#results)
- [Analysis](#analysis)
- [References](#references)

## Problem Statement
This project is done as part of course project of `Advanced Machine Learning`. 

In this project we try to solve if two asked questions by users on CQA(Community Question Answering) platforms are duplicate or not. How one defines two questions as duplicate? We say two questions are duplicate if the users asking the questions have expressed similar ***intent***, i.e., users expect the same kind of answers to the questions asked.

Through this project, we aimed to model this ***intent***, i.e., given two questions classify if the purpose specified by the two questions are similar or not.

## Dataset
The dataset to be used for the study has been provided by [Quora](https://www.kaggle.com/c/quora-question-pairs/data) , a Question - Answer platform not specific to any domain. We modified the dataset little bit for the ease of use and saved all the models, which can downloaded from [here](https://goo.gl/dYDK5Z).

The dataset contains 404,352 question pairs as per the format given in the following table. id represents the unique question pair identifier. qid1 and qid2 represent the question identifier for first and the second question in the pair respectively. question1 and question2 are the full text of Question 1 and Question 2 respectively.

| id | qid1 | qid2 | question1 | question2 | is_duplicate |
|--|--|--|--|--|--|
| 447 | 895 | 896 | What are natural numbers? | What is a least natural number? | 0 |
| 1518 | 3037 | 3038 | Which pizzas are the most popularly ordered pizzas on Domino's menu? | How many calories does a Dominos pizza have? | 0 |
| 3272 | 6542 | 6543 | How do you start a bakery? | How do you start a bakery business? | 1 |
| 3362 | 6722 | 6723 | Should I learn python or Java first? | If I had to choose between learning Java and Python, what should I choose to learn first? | 1 |

Following is the wordcloud of top 2000 most frquently occuring words, clearly wh-questions dominate the dataset as expected from a CQA platform dataset.
<div align="center"><img src="plots/word_cloud.png"/></div> 


## Setup

 - Install python >= 3.6 and pip
 - `pip install -r requirements.txt`
 - Download [GloVe word vectors](https://nlp.stanford.edu/projects/glove/) (glove.840B.300d)

## Architectures Used

### 1

### 1

### 1


## Results


## Analysis


## References


## Project Members

 1. [Arpan Mukherjee](https://github.com/arpanmukherjee)
 2. [Prabhat Kumar](https://github.com/prabhatkumar95)
