import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from model import GPN_Encoder, GPN_Valuator_simple, GPN_Valuator
import utils as u
import config as c
from fgsamp import FGSAMp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def meta_learning_n_times(dataset_name:str, log_path:str, config:dict, cuda_idx=-1):
    if cuda_idx >= 0:
        global device
        device = torch.device(f'cuda:{cuda_idx}')

    u.set_seed(c.SEED)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    results = defaultdict(dict)

    n_way=config['n_way']
    k_spt=config['k_spt']
    m_qry=config['m_qry']
    num_avail = config['num_avail']
    num_repeats = config['num_repeats']

    num_spt = n_way * k_spt

    print('%s %d-way %d-shot' % (dataset_name, n_way, k_spt))

    # --- not follow MAML framework ---
    def meta_learning(results=None, idx_repreat=0):
        # load data
        # class_list: show what classes are there in each set
        # class_dict: show the origin idx of nodes in each class
        x, y, edge_index, \
            class_list_train, class_list_val, class_list_test, \
                class_dict_train, class_dict_val, class_dict_test = u.load_data(dataset_name, config['class_split'])

        num_nodes = x.shape[0]

        x = x.to(device)
        y = y.to(device)
        edge_index = edge_index.to(device)

        deg = degree(edge_index[1], num_nodes)#.to(device)
        adj = u.edge_index_to_adj_with_rw_norm(edge_index, num_nodes)#.to(device)

        if num_avail is not None:
            for i in class_dict_train:
                np.random.shuffle(class_dict_train[i])
            # for i in class_dict_val:
            #     np.random.shuffle(class_dict_val[i])
            # for i in class_dict_test:
            #     np.random.shuffle(class_dict_test[i])

        # init modules
        encoder = GPN_Encoder(
            nfeat=x.shape[1], 
            nhid=config['hidden_dim'], 
            dropout=config['dropout'], 
            num_layers=config['num_layers'], 
        ).to(device)
        if config['gpn'] == 'linear':
            scorer = GPN_Valuator_simple(
                nfeat=x.shape[1], 
                nhid=config['hidden_dim'], 
                dropout=config['dropout']
            ).to(device)
        elif config['gpn'] == 'gat':
            scorer = GPN_Valuator(
                nfeat=x.shape[1], 
                nhid=config['hidden_dim'], 
                dropout=config['dropout']
            ).to(device)

        base_optim = torch.optim.Adam
        optim = FGSAMp(
            [{'params': encoder.parameters()}, {'params': scorer.parameters()}], 
            base_optim, 
            k=config['k'], alpha=config['alpha'], rho=config['rho'], lam=config['lam'], 
            lr=config['lr'], weight_decay=config['wd'], betas=config['betas'], 
            num_pert=config['num_pert'])

        def train_task(train_index, class_list:list, class_dict:dict, train=True):
            # sample task
            idx_spt, idx_qry, class_selected = u.task_generator(
                n_way, k_spt, 5 if num_avail is not None and train else m_qry, class_list, class_dict, num_avail if train else None)

            if train:
                encoder.train()
                scorer.train()
                optim.zero_grad()
            else:
                encoder.eval()
                scorer.eval()

            def forward(use_gnn=True):
                if use_gnn:
                    embeddings = encoder(x, adj, use_gnn)
                    z_dim = embeddings.size()[1]
                    support_embeddings = embeddings[idx_spt]
                    query_embeddings = embeddings[idx_qry]
                    scores = scorer(x, adj, use_gnn)[idx_spt]
                else:
                    embeddings = encoder(x[idx_spt+idx_qry], None, use_gnn)
                    z_dim = embeddings.size()[1]
                    support_embeddings = embeddings[:num_spt]
                    query_embeddings = embeddings[num_spt:]
                    scores = scorer(x[idx_spt], None, use_gnn)

                # embedding lookup
                support_embeddings = support_embeddings.view([n_way, k_spt, z_dim])

                # node importance
                support_degrees = torch.log(deg[idx_spt].view([n_way, k_spt]))
                support_scores = scores.view([n_way, k_spt])
                support_scores = torch.sigmoid(support_degrees * support_scores)      # (N, K)
                support_scores = torch.softmax(support_scores, dim=-1).unsqueeze(-1)  # (N, K, 1)
                support_embeddings = support_embeddings * support_scores  # (N, K, hdim)

                # compute loss
                prototype_embeddings = support_embeddings.mean(dim=1)  # (N, hdim)
                logits = -torch.cdist(query_embeddings, prototype_embeddings)  # (NM, N)
                qry_labels = torch.LongTensor([class_selected.index(i) for i in y[idx_qry]]).to(device)
                loss = F.cross_entropy(logits, qry_labels)
                acc = (logits.argmax(dim=-1) == qry_labels).float().mean().item()

                return loss, acc


            # LookSAM optim
            if train:
                if train_index % config['k'] == 0:
                    params = encoder.get_params() + scorer.get_params()
                    loss, acc = forward(use_gnn=True)
                    loss_mlp, _ = forward(use_gnn=False)
                    g_mlp = torch.autograd.grad(loss_mlp, params)
                    num_encoder_params = config['num_layers'] * 2
                    g_mlp = [g_mlp[:num_encoder_params], g_mlp[num_encoder_params:]]
                    loss.backward(retain_graph=True)
                else:
                    loss, acc = forward(use_gnn=False)
                    g_mlp = None
                    loss.backward()
                optim.step(train_index, forward, g_mlp, zero_grad=True)
            else:
                _, acc = forward(use_gnn=True)


            return acc


        # meta-training
        print(os.getcwd().split(os.sep)[-1], f"K={config['k']}, num_layers={config['num_layers']}", f"hdim={config['hidden_dim']}")
        print('------------------ Meta-Train #%d %s %d-way %d-shot ------------------' % (idx_repreat, dataset_name, n_way, k_spt))
        cnt = 0
        best_val_acc = 0.
        acc_meta_train = []
        for episode in tqdm(range(config['num_episodes']), ncols=70):
            # train
            acc_train = train_task(episode, class_list_train, class_dict_train, train=True)
            acc_meta_train.append(acc_train)

            if (episode + 1) % config['show_train_interval'] == 0:
                print('Train #%d | acc: %.4f' % (episode + 1, np.mean(acc_meta_train, axis=0)))

            if (episode + 1) % config['val_interval'] == 0:
                # val
                acc_meta_val = []
                for i in range(config['num_meta_val']):
                    with torch.no_grad():
                        acc_val = train_task(i, class_list_val, class_dict_val, train=False)
                        acc_meta_val.append(acc_val)

                acc_meta_val = np.mean(acc_meta_val, axis=0)
                if acc_meta_val > best_val_acc:
                    best_val_acc = acc_meta_val
                    cnt = 0
                else:
                    cnt += 1
                print('Valid #%d | acc: %.4f, best_acc: %.4f' % (episode + 1, acc_meta_val, best_val_acc))

                if cnt >= config['patience']:  # early-stop
                    break

        # meta-testing
        print(os.getcwd().split(os.sep)[-1], f"K={config['k']}, num_layers={config['num_layers']}", f"hdim={config['hidden_dim']}")
        print('------------------ Meta-test #%d %s %d-way %d-shot ------------------' % (idx_repreat, dataset_name, n_way, k_spt))
        acc_meta_test = []
        for i in range(config['num_meta_test']):
            with torch.no_grad():
                acc_test = train_task(i, class_list_test, class_dict_test, train=False)
                acc_meta_test.append(acc_test)

            if (i + 1) % config['show_test_interval'] == 0:
                print('Test #%d | acc: %.4f' % (i + 1, np.mean(acc_meta_test, axis=0)))

        acc_meta_test = np.mean(acc_meta_test, axis=0)
        results[dataset_name]['%d-way %d-shot %s-avail #%d-repeat' % (
            n_way, k_spt, num_avail, idx_repreat)] = {'test_acc': acc_meta_test}
        write_file = {'result': results[dataset_name], 'config': config}
        with open(osp.join(log_path, 'result_%s_%d-way_%d-shot_%s-avail.json' % (dataset_name, n_way, k_spt, num_avail)), 'w') as f:
            json.dump(write_file, f, indent=4)

        return results, best_val_acc


    accs = []
    best_val_accs = []
    for n in range(num_repeats):
        results, best_val_acc = meta_learning(results, n)
        accs.append(results[dataset_name]['%d-way %d-shot %s-avail #%d-repeat' % (n_way, k_spt, num_avail, n)]['test_acc'])
        best_val_accs.append(best_val_acc)

    val_acc = np.mean(best_val_accs, axis=0)
    final_acc = np.mean(accs, axis=0)
    results[dataset_name]['%d-way %d-shot' % (n_way, k_spt)] = {'acc': final_acc}
    write_file = {'result': results[dataset_name], 'config': config}
    with open(osp.join(log_path, 'result_%s_%d-way_%d-shot_%s-avail.json' % (dataset_name, n_way, k_spt, num_avail)), 'w') as f:
        json.dump(write_file, f, indent=4)

    return final_acc, val_acc, results


if __name__ == "__main__":
    folder_name = os.getcwd().split(os.sep)[-1]
    for dataset_name in ['corafull', 'dblp', 'ogbn-arxiv']:
        for n_way in [5, 10]:
            for k_spt in [3, 5]:
                config = c.config_fs(dataset_name)
                config['n_way'] = n_way
                config['k_spt'] = k_spt
                meta_learning_n_times(dataset_name, 'log_%s_layer%s_hdim%d_k%d/' % (
                    config['gpn'], config['num_layers'], config['hidden_dim'], config['k']), config)