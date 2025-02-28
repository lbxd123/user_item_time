import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Data import DataLoaderMovieLens
from torch.utils.data import DataLoader
from Data import DataLoaderKuaiRec
import torch
from torch import nn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc,roc_auc_score

class FNN(nn.Module):

    def __init__(self, n_features, user_df, item_df, dim=128):
        super(FNN, self).__init__()

        self.features = nn.Embedding(n_features, dim, max_norm=1)
        self.mlp_layer = self.__mlp(dim)

        self.user_df = user_df
        self.item_df = item_df

    def __mlp(self, dim):

        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def FMaggregator(self, feature_embs):
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        output = square_of_sum - sum_of_square

        return output

    def __getAllFeatures(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        all = torch.cat([users, items], dim=1)

        return all

    def forward(self, u, i):
        all_feature_index = self.__getAllFeatures(u, i)
        all_feature_embs = self.features(all_feature_index)
        out1 = self.FMaggregator(all_feature_embs)
        out = self.mlp_layer(out1)
        out = torch.squeeze(out)

        return out


def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u, i)

    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    auc = roc_auc_score(r, y_pred)
    precision = precision_score(r, y_pred)
    recall = recall_score(r, y_pred)
    acc = accuracy_score(r, y_pred)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, acc, f1, auc

# batchSize=512, lr=0.0005
def train(epochs=50, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1, need_eva=True):

    train_triples, test_triples, user_df, item_df, n_features = \
        DataLoaderMovieLens.read_data()

    net = FNN(n_features, user_df, item_df, dim)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.3)
    for e in range(epochs):
        all_lose = 0
        for u, i, r in DataLoader(train_triples, batch_size=batchSize, shuffle=True):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            logits = net(u, i)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train_triples) // batchSize)))

        if e % eva_per_epochs == 0 and need_eva:
            p, r, acc, f1, auc = doEva(net, train_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}, f1:{:.4f} auc:{:.4f}'.format(p, r, acc, f1, auc))

            p, r, acc, f1, auc = doEva(net, test_triples)
            print('train:p:{:.4f}, r:{:.4f}, acc:{:.4f}, f1:{:.4f} auc{:.4f}'.format(p, r, acc, f1, auc))

    return net


if __name__ == '__main__':
    train()
