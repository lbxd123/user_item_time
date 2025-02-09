import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from Data import DataLoaderKuaiRec
from Data import DataLoaderMovieLens
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import Parameter, init
from sklearn.metrics import roc_curve
from sklearn.metrics import auc,roc_auc_score


class AFM(nn.Module):
    def __init__(self, n_features, user_df, item_df, k, t):
        super(AFM, self).__init__()
        self.features = nn.Embedding(n_features, k, max_norm=1)
        self.attention_liner = nn.Linear(k, t)
        self.h = init.xavier_uniform_(Parameter(torch.empty(t, 1)))
        self.p = init.xavier_uniform_(Parameter(torch.empty(k, 1)))
        self.user_df = user_df
        self.item_df = item_df

    def FMaggregator(self, feature_embs):
        square_of_sum = torch.sum(feature_embs, dim=1) ** 2
        sum_of_square = torch.sum(feature_embs ** 2, dim=1)
        output = square_of_sum - sum_of_square
        output = 0.5 * output
        return output

    def attention(self, embs):
        embs = self.attention_liner(embs)
        embs = torch.relu(embs)
        embs = torch.matmul(embs, self.h)
        atts = torch.softmax(embs, dim=1)
        return atts

    def __getAllFeatures(self, u, i):
        users = torch.LongTensor(self.user_df.loc[u].values)
        items = torch.LongTensor(self.item_df.loc[i].values)
        all = torch.cat([users, items], dim=1)
        return all

    def forward(self, u, i):
        all_feature_index = self.__getAllFeatures(u, i)
        all_feature_embs = self.features(all_feature_index)
        embs = self.FMaggregator(all_feature_embs)

        atts = self.attention(embs)
        outs = torch.matmul(atts * embs, self.p)
        outs = torch.squeeze(outs)
        logit = torch.sigmoid(outs)
        return logit


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


def train(epochs=50, batchSize=512, lr=0.001, k=256, t=64, eva_per_epochs=1, need_eva=True):
    train_triples, test_triples, user_df, item_df, n_features = DataLoaderMovieLens.read_data()
    net = AFM(n_features, user_df, item_df, k, t)
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
