"""
library module for providing the model architectures.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_max_pool


class GATNetwork(torch.nn.Module):
    """Graph attention neural network"""

    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_h, heads=heads)
        self.gat3 = GATv2Conv(dim_h*heads, dim_h, heads=heads)
        self.gat4 = GATv2Conv(dim_h*heads, dim_out, heads=1)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.2, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat3(h, edge_index)
        h = F.elu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat4(h, edge_index)
        return F.log_softmax(h, dim=1)


class Encoder(nn.Module):
    """Encoder for Graph2Seq model."""
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hid_dim, improved=True)
        self.conv2 = GCNConv(hid_dim, hid_dim * 2, improved=True)
        self.conv3 = GCNConv(hid_dim * 2, hid_dim * 4, improved=True)
        self.conv4 = GCNConv(hid_dim * 4, hid_dim * 2, improved=True)
        self.conv5 = GCNConv(hid_dim * 2, hid_dim, improved=True)
        self.drop1 = nn.Dropout(0.9)

    def forward(self, data, edge_index, batch):
        # batch = [num node * num graphs in batch]
        # data = [num node * num graphs in batch, num features]
        # edge_index = [2, num edges * num graph in batch]
        # Obtain node embeddings
        # data = [num node * num graphs in batch, hid dim]
        output = self.conv1(data, edge_index)
        output = output.relu()
        output = self.conv2(output, edge_index)
        output = output.relu()
        output = self.conv3(output, edge_index)
        output = output.relu()
        output = self.conv4(output, edge_index)
        output = output.relu()
        output = self.conv5(output, edge_index)

        # Readout layer
        # output = [batch size, hid dim]
        output = global_max_pool(output, batch)

        return output


class Decoder(nn.Module):
    """Decoder for Graph2Seq model."""
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data, hidden, context):
        # data = [batch size]
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]
        data = data.unsqueeze(0)
        # embedded = [1, batch size, hid dim]
        embedded = self.dropout(self.embedding(data))
        # emb_con = [1, batch size, emb dim + hid dim]
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        # output = [batch size, emb dim + hid dim * 2]
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        # prediction = [batch size, output dim]
        prediction = self.fc_out(output)

        return prediction, hidden


class Graph2Seq(nn.Module):
    """Graph2Seq model."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, data):
        # tgt = [1, output_dim * batch_size] -> [output_dim, batch_size], float -> long
        # batch_size = num of graphs in this batch
        tgt = torch.reshape(data.y, (1, -1)).t()
        # tgt_len is max target expression length.
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, 1, tgt_vocab_size).to(self.device)
        # context = [1, batch_size, hid_dim]
        context = self.encoder(data.x, data.edge_index,
                               data.batch).unsqueeze(0)

        hidden = context

        expr = tgt[0, :]
        for t in range(0, tgt_len):
            # expr = [batch_size]
            # hidden = [1, batch_size, hid_dim]
            # context = [1, batch_size, hid_dim]
            output, hidden = self.decoder(expr, hidden, context)
            outputs[t] = output

        return outputs
