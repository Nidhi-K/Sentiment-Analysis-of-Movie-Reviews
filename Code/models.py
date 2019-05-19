# models.py

from sentiment_data import *
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import torch.nn.functional as F

# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

# DEFINING THE COMPUTATION GRAPH
# Define the core neural network: one hidden layer, tanh nonlinearity
# Returns probabilities; in general your network can be set up to return probabilities, log probabilities,
# or (log) probabilities + loss
class FFNN(nn.Module):
    def __init__(self, inp, hid1, hid2,  out):
        super(FFNN, self).__init__()
        self.V1 = nn.Linear(inp, hid1)
        self.g1 = nn.Tanh()
        self.V2 = nn.Linear(hid1, hid2)
        self.g2 = nn.Tanh()
        self.W = nn.Linear(hid2, out)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform_(self.V1.weight)
        nn.init.xavier_uniform_(self.V2.weight)
        nn.init.xavier_uniform_(self.W.weight)

    # Forward computation. 
    def forward(self, x):
        return self.softmax(self.W(self.g2(self.V2(self.g1(self.V1(x))))))

# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    inp_size = word_vectors.vectors[0].size
    num_epochs = 15
    num_classes = 2
    hidden_1_size = 20
    hidden_2_size = 40
    
    ffnn = FFNN(inp_size, hidden_1_size, hidden_2_size, num_classes)
    initial_learning_rate = 0.001
    optimizer = optim.Adam(ffnn.parameters(), lr = initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(train_labels_arr.size)]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            sentence = train_mat[idx]
            inp = np.mean(np.asarray([word_vectors.vectors[int(i)]for i in sentence]),axis=0)
            inp = torch.from_numpy(inp).float()
            y = train_labels_arr[idx]
            # Build one-hot representation of y
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.long)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(inp)
            # Calculating Loss
            loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
        #print_evaluation(train_exs,ffnn, word_vectors, "train set")
        print_evaluation(dev_exs,ffnn, word_vectors, "dev set")
    predicted_exs = predict_test(test_exs, ffnn, word_vectors)
    return predicted_exs

class LSTM(nn.Module):
    def __init__(self, embed_size, hid_size, output_size, word_vectors): 
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_vectors.vectors).float())
        
        self.hid_size = hid_size
        self.lstm_model = nn.LSTM(embed_size, hid_size, bidirectional = False)
        self.hidden = self.initialize_hidden()
        self.fc = nn.Linear(hid_size, output_size)
        self.softmax = nn.Softmax(dim=0)
    
    def initialize_hidden(self):
        # (num_layers, minibatch_size, hidden_dim) for (h, c)
        return (torch.zeros(1, 1, self.hid_size), torch.zeros(1, 1, self.hid_size))
                 
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        output, self.hidden  = self.lstm_model(embeds.view(len(sentence), 1, -1), self.hidden)
        # output - (sentence_length,batch_size,hid_size*num_directions)
        # self.hidden[0] - (num_layers*num_directions, batch_size, hid_size)
        # self.hidden[1] - (num_layers*num_directions, batch_size, hid_size)
        #tag_space = self.fc(self.hidden[0].squeeze()) # - works for uni_directional
        tag_space = self.fc(output[-1].squeeze()) # - works for uni and bi-directional
        tag_scores = self.softmax(tag_space)
        return tag_scores

# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    train_labels_arr = np.array([ex.label for ex in train_exs])
    sentences = [ex.indexed_words for ex in train_exs]
    num_epochs = 3
    
    embed_size = word_vectors.vectors[0].size
    hid_size = 100
    output_size = 2
    num_classes = 2
    lstm = LSTM(embed_size, hid_size, output_size, word_vectors)
    initial_learning_rate = 0.001
    optimizer = optim.Adam(lstm.parameters(), lr = initial_learning_rate)

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(train_labels_arr.size)]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            inp = torch.LongTensor(sentences[idx])
            y = train_labels_arr[idx]
            # Build one-hot representation of y
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.long)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            lstm.zero_grad()
            lstm.hidden = lstm.initialize_hidden()
            scores = lstm.forward(inp)
            # Calculating Loss
            loss = torch.neg(torch.log(scores)).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
        #print_evaluation_lstm(train_exs,lstm, "lstm - train set")
        print_evaluation_lstm(dev_exs,lstm, "lstm - dev set")
    predicted_exs = predict_test_lstm(test_exs, lstm)
    return predicted_exs
            
def print_evaluation_lstm(exs, lstm, which_set):
    sentences = [ex.indexed_words for ex in exs]
    labels_arr = np.array([ex.label for ex in exs])
    correct_predictions = 0    
    for idx in range(0, labels_arr.size):
        inp = torch.LongTensor(sentences[idx])
        y = labels_arr[idx]
        lstm.hidden = lstm.initialize_hidden()
        scores = lstm.forward(inp)
        prediction = torch.argmax(scores)
        if y == prediction:
            correct_predictions += 1
    
    print(repr(correct_predictions) + "/" + repr(len(labels_arr)) + " correct in " + which_set)
    print("Accuracy on " + which_set + " = " + "{0:.2f}".format((correct_predictions*100)/len(labels_arr))+" %")
    
def predict_test_lstm(exs, lstm):
    sentences = [ex.indexed_words for ex in exs]
    predicted_exs = []
    for idx in range(len(sentences)):
        inp = torch.LongTensor(sentences[idx])
        lstm.hidden = lstm.initialize_hidden()
        scores = lstm.forward(inp)
        prediction = torch.argmax(scores)
        predicted_exs.append(SentimentExample(exs[idx].indexed_words, prediction.item()))
    print("Predicted on " + str(len(sentences)) + " test examples")
    return predicted_exs

def print_evaluation(exs, network, word_vectors, which_set):
    seq_max_len = 60
    exs_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in exs])
    labels_arr = np.array([ex.label for ex in exs])

    correct_predictions = 0
    for idx in range(0, labels_arr.size):
        sentence = exs_mat[idx]
        inp = np.mean(np.asarray([word_vectors.vectors[int(i)]for i in sentence]),axis=0)
        inp = torch.from_numpy(inp).float()
        y = labels_arr[idx]
        probs = network.forward(inp)
        prediction = torch.argmax(probs)
        if y == prediction:
            correct_predictions += 1
        #print("Example " + repr(train_xs[idx]) + "; gold = " + repr(train_ys[idx]) + "; pred = " +\
            #repr(prediction) + " with probs " + repr(probs))
    print(repr(correct_predictions) + "/" + repr(len(labels_arr)) + " correct in " + which_set)
    print("Accuracy on " + which_set + " = " + "{0:.2f}".format((correct_predictions*100)/len(labels_arr))+" %")

def predict_test(test_exs, network, word_vectors):
    seq_max_len = 60
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    num_test_exs = len(test_exs)
    predicted_exs = []
    for idx in range(num_test_exs):
        sentence = test_mat[idx]
        inp = np.mean(np.asarray([word_vectors.vectors[int(i)]for i in sentence]),axis=0)
        inp = torch.from_numpy(inp).float()
        probs = network.forward(inp)
        prediction = torch.argmax(probs)
        predicted_exs.append(SentimentExample(test_exs[idx].indexed_words, prediction.item()))
    print("Predicted on " + str(num_test_exs) + " test examples")
    return predicted_exs


