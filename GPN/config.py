
SEED = 28


# TLP settings
def default_config_fs():
    config = {
        'n_ways': [2, 3, 5], 
        'k_spts': [3], 
        'n_way': 5, 
        'k_spt': 3, 
        'm_qry': 10, 

        'num_repeats': 5, 
        'num_episodes': 1000, 
        # 'finetune_steps': 20, 

        'show_train_interval': 10, 
        'val_interval': 10, 
        'num_meta_val': 20, 
        'num_meta_test': 100, 
        'show_test_interval': 20, 
        'patience': 10, 

        'num_avail': None, 

        'num_layers': 2, 
        'hidden_dim': 16, 
        'gpn': 'linear', 

        'lr': 0.005,  # 0.001
        'wd': 5e-4, 
        'betas': [0.9, 0.999], 

        # need finetune on val set
        'dropout': 0.2, 

        'class_split': {
            'corafull': {'train': 40, 'val': 15, 'test': 15}, 
            'dblp': {'train': 80, 'val': 27, 'test': 30}, 
            'ogbn-arxiv': {'train': 20, 'val': 10, 'test': 10}, 
        }
    }

    return config


def config_fs(dataset_name='corafull'):
    config = default_config_fs()

    if dataset_name == 'corafull':
        pass

    return config