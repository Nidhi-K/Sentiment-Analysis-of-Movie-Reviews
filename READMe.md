# Overview
Classify movie review sentences as either positive or negative. 

# Dataset and Dependencies
Sentences from Rotten Tomatoes split into 80/10/10 train/dev/test

Word Embeddings: Pre-trained 50-dimensional and 300-dimensional GloVE vectors

Python 3.6.6, Pytorch 0.4.1

# Network Architecture
**Feed-forward model:** 2-layer neural network with tanh non-linearity, softmax classification and Adam optimizer

**LSTM:** Both uni-directional and bi-directional with hidden size 100

#
To Run: <br/>
```python3 sentiment.py``` (default uses 50-D embeddings and feed forward model) <br/>
For **300-D** embeddings, add ```--word_vecs_path=data/glove.6B.300d-relativized.txt``` <br/>
For **LSTM** model, add ```--model=FANCY```
