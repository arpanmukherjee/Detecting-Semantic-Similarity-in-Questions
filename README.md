


# Detecting Semantic Similarity in Questions


## Quick Links
- [About](#problem-statement)
- [Dataset](#dataset)
- [Setup](#setup)
- [Algorithms](#architectures-used)
	- [SVM](#support-vector-machine)
	- [Autoencoder](#autoencoder)
	- [Neural Network](#dense-neural-network)
	- [Siamese Models](#siamese-architecture)
- [Results](#results)
- [Analysis](#analysis)
- [References](#references)

## Problem Statement
We did this project as part of the course project of `Advanced Machine Learning.` 

In this project, we try to solve if two asked questions by users on CQA(Community Question Answering) platforms are duplicate or not. How one defines two questions as duplicate? We say two questions are duplicate if the users asking the questions have expressed similar ***intent***, i.e., users expect the same kind of answers to the questions asked.

Through this project, we aimed to model this ***intent***, i.e., given two questions classify if the purpose specified by the two questions are similar or not.

For more detailed report please refer to the [Report](AML_Report.pdf).

## Dataset
The dataset to be used for the study was published by [Quora](https://www.kaggle.com/c/quora-question-pairs/data), a Question-Answer platform not specific to any domain. We modified the dataset little-bit for the ease of use and saved all the models, which can be downloaded from [here](https://goo.gl/dYDK5Z).

The dataset contains 404,352 question pairs as per the format is given in the following table. `id` represents the unique question pair identifier. `qid1` and `qid2` represent the question identifier for first and the second question in the pair, respectively.  `question1` and `question2` are the full text of Question 1 and Question 2, respectively. `is_duplicate` is boolean variable, which signifies if both the questions are duplicate or not.

| id | qid1 | qid2 | question1 | question2 | is_duplicate |
|--|--|--|--|--|--|
| 447 | 895 | 896 | What are natural numbers? | What is the least natural number? | 0 |
| 1518 | 3037 | 3038 | Which pizzas are the most popularly ordered pizzas on Domino's menu? | How many calories does a Dominos pizza have? | 0 |
| 3272 | 6542 | 6543 | How do you start a bakery? | How do you start a bakery business? | 1 |
| 3362 | 6722 | 6723 | Should I learn python or Java first? | If I had to choose between learning Java and Python, what should I choose to learn first? | 1 |

Following is the wordcloud of top 2000 most frquently occuring words, clearly wh-questions dominate the dataset as expected from a CQA platform dataset.
<p align="center">
	<img src="plots/word_cloud.png" height='250px'/><br>
	<code>Fig 1: Wordcloud of top 2000 most frequently occurring words</code>
</p>

<p align="center">
	<img src="plots/cos_dist.png" height='250px'/><br>
	<code>Fig 2: Cosine distance between similar and dissimilar embedding vectors</code>
</p>

## Setup

 - Install python >= 3.6 and pip
 - `pip install -r requirements.txt`
 - Download [GloVe word vectors](https://nlp.stanford.edu/projects/glove/) (glove.840B.300d)

## Architectures Used

### Support Vector Machine
SVM with Linear kernel was applied and used as a baseline for the classification task. Results from the model helped in better understanding of the data.
	
### Autoencoder
Question similarity can also be seen as a generative task of generating questions that are similar to the input. Therefore, these tasks can be modeled using Autoencoders, where the task is to learn a representation after the encoder such that the autoencoder minimizes the loss between the representation of two similar questions. For this task, the model was trained on only similar question pairs, and no comment can be made about the representation of non-similar question pairs. The following loss function `L` was minimized, where <code>X<sub>1</sub></code> and <code>X<sub>2</sub></code> represent the two similar questions in a pair, and `M()` is the autoencoder’s output. 

<div align="center"><img src="https://latex.codecogs.com/gif.latex?L=\|M(X_1)-X_2\|^2" /></div>

Later a classification layer was added to the concatenated output of two questions after the encoder layer. This network can also be seen as a siamese network where the representations are learned by an autoencoder based upon the similarity only. 

### Dense Neural Network
A `8-layered` vanilla cone-based neural network implemented for the dataset. We used `ReLU` activation function and trained the network for `100 epochs` with `Adam` optmizer and `learning rate=0.001`.
<p align="center">
	<img src="plots/8-layered NN.png" height='250px'/><br>
	<code>Fig 3: 8-Layer NN Architecture</code>
</p>


### Siamese Architecture
In this architecture, the same neural network model is being used to encode two individual sentences which are given as input independently. Both the input sentences are now encoded into sentence vector in the same embedding space, as shown in Figure 1. Then by using some distant metric decision will be made solely based on this result.

<p align="center">
	<img src="plots/siamese.png" height='300px'/><br>
	<code>Fig 4: General architecture of Siamese Model</code>
</p>

#### Convolutional Siamese Network

<p align="center">
	<img src="plots/conv.png" height='500px' style="transform:rotate(90deg)"/><br>
	<code>Fig 5: Convolutional Siamese Network Architecture</code>
</p>

#### LSTM Siamese Network
The LSTM Siamese Architecture was trained with `learning rate=0.001` and `Mean Square Error` as the loss function with `AdaDelta` optimizer. For classification layer, we used `Cross Entropy Loss`.
<p align="center">
	<img src="plots/LSTM.png" height='500px' style="transform:rotate(90deg)"/><br>
	<code>Fig 6: LSTM Siamese Network Architecture</code>
</p>



## Results

| Algorithm | Accuracy | Embedding type |
|--|--|--|
| Support Vector Machine | 59.23% | doc2vec |
| Autoencoder | 62.85% | doc2vec |
| Neural Network | 79.28% | doc2vec |
| Convolutional Siemese | 64.33% | doc2vec |
| LSTM Siemese | 80.32% | word2vec |


## Analysis
Following graphs are only for the LSTM Siamese Network architecture. We tried considering data imbalancing and trained 2 different models according to it.

<p align="center">
	<img src="plots/balanced_lstm_acr.png" height='250px'/><br>
	<code>Fig 7: Variation of accuracy with epoch for balanced dataset for LSTM</code>
</p>

<p align="center">
	<img src="plots/balanced_lstm.png" height='250px'/><br>
	<code>Fig 8: Variation of loss with epoch for balanced dataset for LSTM</code>
</p>

<p align="center">
	<img src="plots/unbalanced_lstm_acr.png" height='250px'/><br>
	<code>Fig 9: Variation of accuracy with epoch for unbalanced dataset for LSTM</code>
</p>

<p align="center">
	<img src="plots/unbalanced_lstm.png" height='250px'/><br>
	<code>Fig 10: Variation of loss with epoch for unbalanced dataset for LSTM</code>
</p>


## References
- Zhiguo Wang, Wael Hamza, and Radu Florian. Bilateral multi- perspective matching for natural language sentences. arXiv preprint arXiv:1702.03814, 2017.
- Shuohang Wang and Jing Jiang. A compare-aggregate model for matching text sequences. arXiv preprint arXiv:1611.01747, 2016.
- MingTan,CicerodosSantos,BingXiang,andBowenZhou.Lstm-based deep learning models for non-factoid answer selection. arXiv preprint arXiv:1511.04108, 2015.
- Kuntal Dey, Ritvik Shrivastava, and Saroj Kaushik. A paraphrase and semantic similarity detection system for user generated short- text content on microblogs. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 2880–2890, 2016.
- Sepp Hochreiter and Ju ̈rgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
- Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Sa ̈ckinger, and Roopak Shah. Signature verification using a” siamese” time delay neural network. In Advances in neural information processing systems, pages 737–744, 1994.

## Project Members

 1. [Arpan Mukherjee](https://github.com/arpanmukherjee)
 2. [Prabhat Kumar](https://github.com/prabhatkumar95)
