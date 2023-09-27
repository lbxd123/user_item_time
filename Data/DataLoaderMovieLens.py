import os
import numpy as np

from Data import filepath as fp
import pandas as pd
import time
from sklearn.utils import shuffle

base_path = fp.Ml_100K.ORGINAL_DIR
train_path = os.path.join(base_path, 'train.csv')
test_path = os.path.join(base_path, 'test.csv')
user_path = os.path.join(base_path, 'u.user')
item_path = os.path.join(base_path, 'u.item')
occupation_path = os.path.join(base_path, 'u.occupation')



def get1or0(r):
    return 1 if r > 4 else 0


def __read_rating_four_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('\t')
            triples.append([int(d[0]), int(d[1]), int(d[2]), int(d[3])])
    return triples


def __read_rating_three_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split('\t')
            triples.append([int(d[0]), int(d[1]), int(d[2])])
    return triples


def read_data_user_item_df():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)

    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)

    return train_triples, test_triples, user_df, item_df, max(user_df.max()) + 1, max(item_df.max()) + 1


def read_data_user_item_time_df():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.Ml_100K.TIME_DF, index_col=0)

    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)

    return train_triples, test_triples, user_df, item_df, time_df, max(user_df.max()) + 1, max(item_df.max()) + 1, max(
        time_df.max()) + 1


def read_data():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)
    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)
    return train_triples, test_triples, user_df, item_df, max(item_df.max()) + 1


def read_data_new():
    user_df = pd.read_csv(fp.Ml_100K.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.Ml_100K.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.Ml_100K.TIME_DF, index_col=0)
    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)
    return train_triples, test_triples, user_df, item_df, time_df, max(item_df.max()) + 1


if __name__ == '__main__':
    pass
