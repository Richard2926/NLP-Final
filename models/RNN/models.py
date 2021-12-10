# BEGIN - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.
import torch
import torch.nn as nn
# END - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.

class ClassificationModel(nn.Module):
    # Instantiate layers for your model-
    # 
    # Your model architecture will be an optionally bidirectional LSTM,
    # followed by a linear + sigmoid layer.
    #
    # You'll need 4 nn.Modules
    # 1. An embeddings layer (see nn.Embedding)
    # 2. A bidirectional LSTM (see nn.LSTM)
    # 3. A Linear layer (see nn.Linear)
    # 4. A sigmoid output (see nn.Sigmoid)
    #
    # HINT: In the forward step, the BATCH_SIZE is the first dimension.
    # HINT: Think about what happens to the linear layer's hidden_dim size
    #       if bidirectional is True or False.
    # 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, \
                 num_layers=1, bidirectional=True):
        super().__init__()
        ## YOUR CODE STARTS HERE (~4 lines of code) ##
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim * 2 if bidirectional else hidden_dim, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        # self.hidden_cell = (torch.zeros(num_layers,1,hidden_dim),
        #                     torch.zeros(num_layers,1,hidden_dim))
        # self.hidden_cell = (torch.zeros(1,16,hidden_dim * 2 if bidirectional else hidden_dim),
        #                     torch.zeros(1,16,hidden_dim * 2 if bidirectional else hidden_dim))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        ## YOUR CODE ENDS HERE ##
        
    # Complete the forward pass of the model.
    #
    # Use the last hidden timestep of the LSTM as input
    # to the linear layer. When completing the forward pass,
    # concatenate the last hidden timestep for both the foward,
    # and backward LSTMs.
    # 
    # args:
    # x - 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
    #     This is the same output that comes out of the collate_fn function you completed-
    def forward(self, x):
        ## YOUR CODE STARTS HERE (~4-5 lines of code) ##
        embed = self.embedding(x)

        h0 = torch.zeros(1, x.size(0), self.hidden_dim * 2 if self.bidirectional else self.hidden_dim)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim * 2 if self.bidirectional else self.hidden_dim)

        output, (h1, c1) = self.lstm(embed, (h0, c0) )
        # 
        # print(output[len(output) - 1].shape)
        x = self.linear(h1[-1])
        x = self.sigmoid(x)
        return x
        ## YOUR CODE ENDS HERE ##
    