class EncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, cell, wordvec, class_label):
        """Set the hyper-parameters and build the layers."""
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)

        if cell == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # since this bi-directional network
        self.wordvec = wordvec
        self.init_weights()
        # self.att = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.linear = nn.Linear(self.hidden_size * 2, class_label)
        self.attention = GlobalAttention(self.hidden_size * 2)

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        # initialize with glove word embedding
        self.embed.weight.data.copy_(torch.from_numpy(self.wordvec))

    # the forward method, which compute the hidden state vector
    def forward(self, text, batch_mask, mask):
        """run the lstm to decode the text"""
        embeddings = self.embed(text)
        hiddens, _ = self.rnn(embeddings)  # b*l*f_size
        batch = embeddings.size()[0]
        # only get the last hidden states
        attvec = hiddens

        # the h_{t} at the mentions
        mask = torch.unsqueeze(mask, 2)
        mask = mask.expand(attvec.size())
        mask = mask.float()
        ht = torch.mean(mask * attvec, 1)  # get the mean vector of the mentions

        # calculate the attention
        self.attention.applyMask(batch_mask)
        output, att = self.attention(ht.view(batch, -1), hiddens)  # batch * dim

        response = self.linear(output.view(batch, -1))

        return F.log_softmax(response, dim=1)