
SEED = 28


def default_config_fs():
    config = {
        'n_ways': [2, 3, 5],
        'k_spts': [3],
        'n_way': 5,
        'k_spt': 3,
        'm_qry': 10,

        'num_repeats': 5,
        'num_episodes': 1000,

        'show_train_interval': 10,
        'val_interval': 10,
        'num_meta_val': 20,
        'num_meta_test': 100,
        'show_test_interval': 20,
        'patience': 10,

        'num_avail': None,

        'num_layers': 2,
        'hidden_dim': 16,

        'rho': 0.05,
        'rho_min': 0.01,
        'rho_max': 1.0,
        'rho_lr': 1.0,
        'lam': 0.5,
        'lr': 0.005,
        'wd': 5e-4,
        'betas': [0.9, 0.999],

        'dropout': 0.2,

        'gpn': 'linear',

        'class_split': {
            'corafull': {'train': 40, 'val': 15, 'test': 15},
            'dblp': {'train': 80, 'val': 27, 'test': 30},
            'ogbn-arxiv': {'train': 20, 'val': 10, 'test': 10},
        }
    }

    return config


def config_fs(dataset_name='corafull'):
    config = default_config_fs()
    return config


