import argparse


config_args = {
    'lr': 0.001,
    'dropout': 0.0,
    'cuda': 0,
    'epochs': 1000,
    'weight-decay': 0.0,
    'seed': 10086,
    'task': 'ea',
    'model': 'RKDEA',
    'num-layers': 2,
    'act': 'relu',
    'dim': 300,
    'n-heads': 4,
    'alpha': 0.2,
    'dataset': 'zh_en',
    'normalize_x': 0,
    'normalize_adj': 1,
    'patience': 10,
    'log-freq': 1,
    'eval-freq': 1,
    'lr-reduce-freq': 2000,
    'gamma': 0.5,
    'min-epochs': 100,
    'use_feats': 1,
    'bias': 1,
    'neg_num': 200,
    'save': 0
}

parser = argparse.ArgumentParser()
for param, val in config_args.items():
    parser.add_argument(f"--{param}", action="append", default=val)
