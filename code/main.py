import torch
import torch.nn as nn
import torch.optim as optim
import time

from utils.utilities import display_num_param
from utils.utilities import get_accuracy
from utils.utilities import normalize_gradient
from mods.networks import MLP1
from mods.networks import MLP3
from mods.networks import CNN
from mods.networks import RNN
from mods.networks import Combined_CNN_MLP3
from mods.networks import Combined_RNN_MLP1

# ================================================================================
# mimic labs_lecture08: Lab04: VGG architecture
def create_net1():
    # parameters
    channel_size  = 64
    input_size    = 2048
    hidden_size_1 = 4096
    hidden_size_2 = 4096
    output_size   = 10

    # modules
    cnn  = CNN(channel_size=channel_size)
    mlp3 = MLP3(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, output_size=output_size)
    net1 = Combined_CNN_MLP3(cnn=cnn, mlp3=mlp3)
    return net1

# mimic labs_lecture10: Lab02: LSTM architecture
def create_net2():
    # parameters
    vocab_size  = 10000
    hidden_size = 300
    input_size  = hidden_size
    output_size = vocab_size

    # modules
    rnn  = RNN(vocab_size=vocab_size, hidden_size=hidden_size)
    mlp1 = MLP1(input_size=input_size, output_size=output_size)
    net2 = Combined_RNN_MLP1(rnn=rnn, mlp1=mlp1)

    # initialize weights
    net2.rnn.layer1.weight.data.uniform_( -0.1, 0.1)
    net2.mlp1.layer1.weight.data.uniform_(-0.1, 0.1)
    return net2
# ================================================================================
def train_net1(net1, train_data, train_labels):
    # parameters
    epoch_count      = 100
    batch_size       = 128
    learning_rate    = 0.25
    train_data_count = train_data.shape[0]

    # objects
    criterion = nn.CrossEntropyLoss()

    # training loop
    start = time.time()
    for epoch in range(epoch_count):

        # statistics
        running_loss     = 0
        running_accuracy = 0
        num_batches      = 0

        # objects - new optimizer each epoch
        optimizer = torch.optim.SGD(net1.parameters(), lr=learning_rate)

        # sample inputs randomly
        shuffled_indices = torch.randperm(train_data_count)

        # minibatch loop
        for count in range(0, train_data_count, batch_size):

            # reset gradients
            optimizer.zero_grad()

            # create minibatch
            indices = shuffled_indices[count:count+batch_size]
            minibatch_data  = train_data[  indices]
            minibatch_label = train_labels[indices]

            # forward pass
            inputs = (minibatch_data - train_data.mean()) / train_data.std()
            inputs.requires_grad_()
            scores = net1(inputs)
            loss   = criterion(scores, minibatch_label)

            # backward pass
            loss.backward()
            optimizer.step()

            # statistics (with detach to prevent accumulation)
            running_loss += loss.detach().item()
            accuracy = get_accuracy(scores.detach(), minibatch_label)
            running_accuracy += accuracy.item()
            num_batches +=1
            print("completed={}, running_loss={}, accuracy={}, running_accuracy={}".format(count/train_data_count, running_loss, accuracy, running_accuracy))
            if count%(batch_size*2) == 0:
                break

        # statistics
        total_loss     = running_loss/num_batches
        total_accuracy = running_accuracy/num_batches
        elapsed        = (time.time()-start)/60
        print("Epoch={}, Train loss={}, Train accuracy={}".format(epoch+1, total_loss, total_accuracy))
        if epoch%2 == 0:
            break

def train_net2(net2, train_data):
    # parameters
    epoch_count      = 100
    batch_size       = 20
    learning_rate    = 5
    train_data_count = train_data.shape[0]
    seq_length       = 35

    # objects
    criterion = nn.CrossEntropyLoss()

    # training loop
    start = time.time()
    for epoch in range(epoch_count):

        # statistics
        running_loss     = 0
        running_accuracy = 0
        num_batches      = 0

        # objects - new optimizer each epoch
        optimizer = torch.optim.SGD(net2.parameters(), lr=learning_rate)

        # initialize weights
        h = torch.zeros(1, batch_size, net2.rnn.layer1.embedding_dim,dtype=torch.float)
        c = torch.zeros(1, batch_size, net2.rnn.layer1.embedding_dim,dtype=torch.float)

        # minibatch loop
        for count in range(0, train_data_count-seq_length, seq_length):

            # reset gradients
            optimizer.zero_grad()

            # create minibatch
            minibatch_data  = train_data[count  :count+seq_length]
            minibatch_label = train_data[count+1:count+seq_length+1]

            # detach to prevent backpropagating all the way to the beginning, then start tracking gradients on h and c
            h = h.detach()
            c = c.detach()
            h = h.requires_grad_()
            c = c.requires_grad_()

            # forward pass
            inputs = minibatch_data.to(torch.int64)
            scores, h, c = net2(inputs, h, c)
            scores          = scores.view(         batch_size*seq_length, net2.rnn.layer1.num_embeddings)
            minibatch_label = minibatch_label.view(batch_size*seq_length).to(torch.int64)
            loss   = criterion(scores, minibatch_label)

            # backward pass
            loss.backward()
            normalize_gradient(net2)
            optimizer.step()

            # statistics (with detach to prevent accumulation)
            running_loss += loss.detach().item()
            accuracy = get_accuracy(scores.detach(), minibatch_label)
            running_accuracy += accuracy.item()
            num_batches +=1
            print("completed={}, running_loss={}, accuracy={}, running_accuracy={}".format(count/train_data_count, running_loss, accuracy, running_accuracy))
            if count%(batch_size*2) == 0:
                break

        # statistics
        total_loss     = running_loss/num_batches
        total_accuracy = running_accuracy/num_batches
        elapsed        = (time.time()-start)/60
        print("Epoch={}, Train loss={}, Train accuracy={}".format(epoch+1, total_loss, total_accuracy))
        if epoch%2 == 0:
            break


# ================================================================================
def case_1():
    net1 = create_net1()
    train_data   = torch.rand(50000,3,32,32)
    train_labels = torch.randint(low=0, high=10, size=(50000,))
    train_net1(net1, train_data, train_labels)

def case_2():
    net2 = create_net2()
    train_data = torch.rand(46479,20)
    train_net2(net2, train_data)

# ================================================================================
case_1()
case_2()
