import os
from Data import filepath as fp
import pandas as pd


base_path = fp.KuaiRec.ORIGINAL_DIR
train_path = os.path.join(base_path, 'train.csv')
test_path = os.path.join(base_path, 'test.csv')


def __read_rating_four_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split(',')
            triples.append([int(d[0]), int(d[1]), int(d[3]), int(float(d[2]))])
    return triples


def read_data_user_item_time_df():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.KuaiRec.TIME_DF, index_col=0)


    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)

    return train_triples, test_triples, user_df, item_df, time_df, max(user_df.max()) + 1, \
           max(item_df.max()) + 1, max(time_df.max()) + 1


def read_data_new():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)
    time_df = pd.read_csv(fp.KuaiRec.TIME_DF, index_col=0)

    train_triples = __read_rating_four_data(train_path)
    test_triples = __read_rating_four_data(test_path)

    return train_triples, test_triples, user_df, item_df, time_df, max(item_df.max())+1


def __read_rating_three_data(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            d = line.strip().split(',')
            triples.append([int(d[0]), int(d[1]), int(d[3])])
    return triples


def read_data_user_item_df():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)

    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)

    return train_triples, test_triples, user_df, item_df, max(user_df.max()) + 1, max(item_df.max()) + 1


def read_data():
    user_df = pd.read_csv(fp.KuaiRec.USER_DF, index_col=0)
    item_df = pd.read_csv(fp.KuaiRec.ITEM_DF, index_col=0)

    train_triples = __read_rating_three_data(train_path)
    test_triples = __read_rating_three_data(test_path)

    return train_triples, test_triples, user_df, item_df, max(item_df.max()) + 1


if __name__ == '__main__':
    pass




