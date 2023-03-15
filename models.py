import torch
import torch.nn as nn


class PCENet(nn.Module):
    def __init__(self, feature_dim, k, nhid, nclass, num_layers=3, nhead=8, dropout=0.1):
        super(PCENet, self).__init__()
        self.k = k
        self.feature_dim = feature_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim + 2 * self.k, nhead=nhead,
                                                   dim_feedforward=nhid, batch_first=True, norm_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.adjust_dim = nn.Sequential(nn.Linear(self.feature_dim + 2*self.k, self.feature_dim), nn.PReLU(),
                                        nn.Linear(self.feature_dim, self.feature_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4, 2048), nn.PReLU(),
            nn.Linear(2048, 2048), nn.PReLU(),
            nn.Linear(2048, nclass)
        )

    def forward(self, data):
        features, adj_mat = data[0], data[1]

        x = torch.cat([features, adj_mat], dim=-1)
        B, N, D = x.shape

        x = self.encoder(x)
        x = torch.cat([x[:, 0:1, :], x[:, self.k:self.k+1, :]], dim=1)
        x = x.view(-1, D)
        x = self.adjust_dim(x)
        x = x.view((B, 2, self.feature_dim))

        pred = torch.cat([features[:, 0], x[:, 0], features[:, self.k], x[:, 1]], dim=-1)
        pred = self.classifier(pred)

        return pred


class LCENet(nn.Module):
    def __init__(self, feature_dim, dim_feedforward, k, num_layers=3, nhead=8, dropout=0.1):
        super(LCENet, self).__init__()
        self.k = k
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim+self.k, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear((feature_dim+self.k)*2, feature_dim), nn.PReLU(),
                                        nn.Linear(feature_dim, 2))

    def forward(self, data):
        x, A = data
        x = torch.cat([x, A], dim=-1)
        B, N, D = x.shape

        x = self.encoder(x)

        x = torch.cat([x[:, 0, :].unsqueeze(1).repeat(1, self.k, 1), x], dim=-1)

        x = x.view(-1, 2*D)
        x = self.classifier(x)
        x = x.view((B, N, 2))

        return x
