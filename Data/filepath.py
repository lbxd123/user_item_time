import os

ROOT = os.path.split(os.path.realpath(__file__))[0]
Model_Dir = os.path.join(ROOT, 'model')


class KuaiRec():
    ORIGINAL_DIR = os.path.join(ROOT, 'KuaiRec')
    TIME_DF = os.path.join(ORIGINAL_DIR, 'time_df')
    USER_DF = os.path.join(ORIGINAL_DIR, 'user_df')
    ITEM_DF = os.path.join(ORIGINAL_DIR, 'item_df')


class Ml_100K():
    ORGINAL_DIR = os.path.join(ROOT, 'MovieLens')
    USER_DF = os.path.join(ORGINAL_DIR, 'user_df.csv')
    ITEM_DF = os.path.join(ORGINAL_DIR, 'item_df.csv')
    TIME_DF = os.path.join(ORGINAL_DIR, 'time_df.csv')


class Ml_1M():
    ORIGINAL_DIR = os.path.join(ROOT, 'ml-1m')
    USER_DF = os.path.join(ORIGINAL_DIR, 'user_df.csv')
    ITEM_DF = os.path.join(ORIGINAL_DIR, 'item_df.csv')
    TIME_DF = os.path.join(ORIGINAL_DIR, 'time_df.csv')


if __name__ == '__main__':
    pass
