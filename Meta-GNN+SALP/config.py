
SEED = 28


def default_config_fs():
    config = {
        'n_ways': [2, 3, 5],
        'k_spts': [3],
        'n_way': 5,
        'k_spt': 3,
        'm_qry': 10,

        'num_repeats': 5,
        'num_episodes': 500,
        'finetune_steps': 10,

        'show_train_interval': 10,
        'val_interval': 10,
        'num_meta_val': 20,
        'num_meta_test': 100,
        'show_test_interval': 20,
        'patience': 10,

        'num_avail': None,

        'hidden_dim': 16,

        # Meta-GCN
        'rho': 0.05,
        'rho_min': 0.01,
        'rho_max': 1.0,
        'rho_lr': 1.0,
        'lr_finetune': 0.1,
        'lr_meta': 0.001,
        'wd': 5e-4,
        'dropout': 0.5,

        'class_split': {
            'cora': {'train': 3, 'val': 2, 'test': 2},
            'citeseer': {'train': 2, 'val': 2, 'test': 2},
            'corafull': {'train': 40, 'val': 15, 'test': 15},
            'coauthor-cs': {'train': 5, 'val': 5, 'test': 5},
            'amazon-clothing': {'train': 40, 'val': 17, 'test': 20},
            'amazon-electronics': {'train': 90, 'val': 37, 'test': 40},
            'amazon-computer': {'train': 4, 'val': 3, 'test': 3},
            'ogbn-arxiv': {'train': 20, 'val': 10, 'test': 10},
            'dblp': {'train': 80, 'val': 27, 'test': 30},
            'reddit': {'train': 21, 'val': 10, 'test': 10},
            'reddit2': {'train': 21, 'val': 10, 'test': 10},
        }
    }

    return config


def config_fs(dataset_name='corafull'):
    config = default_config_fs()
    return config


