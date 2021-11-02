import torch
import torch.nn as nn

# ================================================================================
class MLP1(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP1, self).__init__()

        self.layer1 = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

    def forward(self,x):
        x = self.layer1(x)
        scores = x
        return scores
# ================================================================================
class MLP3(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(MLP3, self).__init__()

        self.layer1 = nn.Linear(in_features=input_size,    out_features=hidden_size_1, bias=True)
        self.layer2 = nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2, bias=True)
        self.layer3 = nn.Linear(in_features=hidden_size_2, out_features=output_size,   bias=True)

    def forward(self, x):
        x = x.view(-1,self.layer1.in_features)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        scores = x
        return scores
# ================================================================================
class CNN(nn.Module):

    def __init__(self, channel_size):
        super(CNN, self).__init__()

        # Conv2d parameters
        in_channels       = 3
        conv_kernel_size  = 3
        conv_stride       = 1
        conv_padding      = 1
        bias              = True

        # MaxPool2d parameters
        pool_kernel_size  = 2
        pool_stride       = 2
        pool_padding      = 0
        
        # block 1
        self.layer1a = nn.Conv2d(in_channels=in_channels,    out_channels=channel_size*1, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.layer1b = nn.Conv2d(in_channels=channel_size*1, out_channels=channel_size*1, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.pool1   = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

        # block 2
        self.layer2a = nn.Conv2d(in_channels=channel_size*1, out_channels=channel_size*2, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.layer2b = nn.Conv2d(in_channels=channel_size*2, out_channels=channel_size*2, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.pool2   = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

        # block 3
        self.layer3a = nn.Conv2d(in_channels=channel_size*2, out_channels=channel_size*4, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.layer3b = nn.Conv2d(in_channels=channel_size*4, out_channels=channel_size*4, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.pool3   = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

        # block 4
        self.layer4a = nn.Conv2d(in_channels=channel_size*4, out_channels=channel_size*8, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, bias=bias)
        self.pool4   = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

    def forward(self, x):
        # block 1
        x = self.layer1a(x)
        x = torch.relu(x)
        x = self.layer1b(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # block 2
        x = self.layer2a(x)
        x = torch.relu(x)
        x = self.layer2b(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # block 3
        x = self.layer3a(x)
        x = torch.relu(x)
        x = self.layer3b(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # block 4
        x = self.layer4a(x)
        x = torch.relu(x)
        x = self.pool4(x)

        scores = x
        return scores
# ================================================================================
class RNN(nn.Module):

    def __init__(self, vocab_size, hidden_size):
        super(RNN, self).__init__()

        # layers
        self.layer1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.layer2 = nn.LSTM(hidden_size, hidden_size)
        
    def forward(self, word_seq, h_init, c_init):
        g_seq                    = self.layer1(word_seq)
        h_seq, (h_final,c_final) = self.layer2(g_seq, (h_init, c_init))
        return h_seq, h_final, c_final
# ================================================================================
class Combined_CNN_MLP3(nn.Module):

    def __init__(self, cnn, mlp3):
        super(Combined_CNN_MLP3, self).__init__()

        self.cnn  = cnn
        self.mlp3 = mlp3

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp3(x)
        scores = x
        return scores
# ================================================================================
class Combined_RNN_MLP1(nn.Module):

    def __init__(self, rnn, mlp1):
        super(Combined_RNN_MLP1, self).__init__()

        self.rnn  = rnn
        self.mlp1 = mlp1

    def forward(self, word_seq, h_init, c_init):
        h_seq, h_final, c_final = self.rnn(word_seq, h_init, c_init)
        score_seq = self.mlp1(h_seq)
        return score_seq, h_final, c_final
# ================================================================================