import torch
import torch.nn as nn

class NERv1_PRE(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, num_layers, num_classes, bidirectional, device, weights = None) -> None:
        """
        NER model with pretrained embeddings, structure = EMBEDDING -> LSTM -> GRU -> LRELU -> FC

        :param voc_size: size of the vocabulary
        :param emb_size: size of the embedding
        :param hidden_size: size of the hidden layer
        :param num_layers: number of layers
        :param num_classes: number of classes
        :param bidirectional: if true the LSTM and GRU are bidirectional
        :param device: cpu or cuda
        :param weights: pretrained embedding weights
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.device = device
        if weights is None:
            self.embedding = nn.Embedding(voc_size, emb_size)
        else:
            self.weights = weights
            self.embedding = self.init_weights(self.weights, True)
        self.bidirectional = 2 if bidirectional else 1
        # with dropout in lstm no guarantee of reproducibility
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional) 
        self.gru = nn.GRU(hidden_size*self.bidirectional, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.bidirectional, num_classes)
        self.act_l1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        

    def init_weights(self, w, freeze=False):
        """
        Initialize the pretrained weights of the model.
        """
        vectors = torch.FloatTensor(w.vectors)
        return torch.nn.Embedding.from_pretrained(vectors, freeze=freeze)

    def forward(self, input) -> torch.Tensor:
        """
        :param input: input tensor of shape (batch_size, seq_len)

        :return: predictions tensor of shape (batch_size, seq_len, num_classes)
        """
        # ugly workaround for compatibility between pytorch 1.9.0 and 1.11
        not_batched = False

        # extract the embedding at the indexes of the input vector 
        embedding = self.embedding(input)

        #initialize hidden and cell states
        h0 = torch.ones(self.bidirectional*self.num_layers, 1, self.hidden_size).to(self.device)
        c0 = torch.ones(self.bidirectional*self.num_layers, 1, self.hidden_size).to(self.device)

        # ugly workaround for compatibility between pytorch 1.9.0 and 1.11
        if len(embedding.shape) < 3:
            embedding = embedding.unsqueeze(0)
            not_batched = True

        #output of lstm: batch, seqence length, hidden size
        out, (h_n, c_n) = self.lstm(embedding, (h0, c0))
        
        #output of gru: batch, seqence length, hidden size
        out, g_n = self.gru(out, h_n)

        #apply activation function 
        act_l1 = self.act_l1(out)

        # ugly workaround for compatibility between pytorch 1.9.0 and 1.11
        if not_batched:
            act_l1 = act_l1.squeeze(0)

        #apply dropout to the output of the activation function
        dropout = self.dropout(act_l1)
        # fully connected layer
        out = self.fc(dropout)
        return out
