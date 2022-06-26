from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl
from features_extraction import extract_features_effnet

DATASET_PATH = '../dataset/captions.txt'
EFF_NET_CONV_TEST = '../dataset/eff_net_conv_test.pkl'


def extract_features(images: pd.Series, model='eff_net'):
    feats = []
    path = ''
    if model == 'eff_net':
        feats = extract_features_effnet('effnetb0', images.unique())
        path = EFF_NET_CONV_TEST
    save(path, feats)


def save(file_path, obj):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    df = pd.read_csv(DATASET_PATH)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    extract_features(df_test['image'], model='eff_net')
