import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

import torch
from torch import nn

from model import MLP, GCN, Linear
import utils as u
import config as c
from sam import SAM


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
        adj_sparse = u.edge_index_to_adj_with_rw_norm(edge_index.to(device), num_nodes)

        if num_avail is not None:
            for i in class_dict_train:
                np.random.shuffle(class_dict_train[i])

        # init modules
        gnn = GCN(
            in_channels=x.shape[1], 
            hidden_channels=config['hidden_dim'], 
            out_channels=config['hidden_dim'], 
            dropout=config['dropout']).to(device)
        classifier = Linear(config['hidden_dim'], n_way).to(device)

        base_optim = torch.optim.Adam
        optim = SAM(
            [{'params': gnn.parameters()}, 
             {'params': classifier.parameters()}], base_optim, 
            rho=config['rho'], lr=config['lr_meta'], weight_decay=config['wd']
        )

        loss_f = nn.CrossEntropyLoss()

        def train_task(class_list:list, class_dict:dict, train=True):
            # sample task
            idx_spt, idx_qry, class_selected = u.task_generator(
                n_way, k_spt, 5 if num_avail is not None and train else m_qry, class_list, class_dict, num_avail if train else None)
            support_labels = torch.tensor([class_selected.index(i) for i in y[idx_spt]]).to(device)
            query_labels = torch.tensor([class_selected.index(i) for i in y[idx_qry]]).to(device)

            if train:
                gnn.train()
                # mlp.train()
                optim.zero_grad()
            else:
                gnn.eval()
                # mlp.eval()


            def forward(use_gnn=True):
                gc1_w, gc1_b, gc2_w, gc2_b, w, b = gnn.gc1.w, gnn.gc1.b, gnn.gc2.w, gnn.gc2.b, classifier.w, classifier.b
                for j in range(config['finetune_steps']):
                    if use_gnn:
                        emb_features = gnn(x, adj_sparse, [gc1_w, gc1_b, gc2_w, gc2_b])
                        ori_emb = emb_features[idx_spt]
                    else:
                        ori_emb = gnn(x[idx_spt], None, [gc1_w, gc1_b, gc2_w, gc2_b])

                    loss = loss_f(classifier(ori_emb, [w, b]), support_labels)
                    grad = torch.autograd.grad(loss, [gc1_w, gc1_b, gc2_w, gc2_b, w, b])
                    gc1_w, gc1_b, gc2_w, gc2_b, w, b = list(
                        map(lambda p: p[1] - config['lr_finetune'] * p[0], zip(grad, [gc1_w, gc1_b, gc2_w, gc2_b, w, b])))

                    # print(grad)
                    if torch.isnan(grad[0]).sum() > 0:
                        print(grad)
                        # print(1 / 0)

                gnn.eval()
                # mlp.eval()
                if use_gnn:
                    emb_features = gnn(x, adj_sparse, [gc1_w, gc1_b, gc2_w, gc2_b])
                    ori_emb = emb_features[idx_qry]
                else:
                    ori_emb = gnn(x[idx_qry], None, [gc1_w, gc1_b, gc2_w, gc2_b])

                logits = classifier(ori_emb, [w, b])
                loss = loss_f(logits, query_labels)
                acc = (logits.argmax(dim=-1) == query_labels).float().mean().item()

                return loss, acc

            loss, acc = forward(use_gnn=True)
            if train:
                loss.backward()
                optim.first_step(zero_grad=True)
                forward(use_gnn=True)[0].backward()
                optim.second_step(zero_grad=True)

            return acc


        # meta-training
        print(os.getcwd().split(os.sep)[-1])
        print('------------------ Meta-Train #%d %s %d-way %d-shot ------------------' % (idx_repreat, dataset_name, n_way, k_spt))
        cnt = 0
        best_val_acc = 0.
        acc_meta_train = []
        global t_train
        for episode in tqdm(range(config['num_episodes']), ncols=70):
            # train
            acc_train = train_task(class_list_train, class_dict_train, train=True)
            acc_meta_train.append(acc_train)

            if (episode + 1) % config['show_train_interval'] == 0:
                print('Train #%d | acc: %.4f' % (episode + 1, np.mean(acc_meta_train, axis=0)))

            if (episode + 1) % config['val_interval'] == 0:
                # val
                acc_meta_val = []
                for i in range(config['num_meta_val']):
                    acc_val = train_task(class_list_val, class_dict_val, train=False)
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
        print(os.getcwd().split(os.sep)[-1])
        print('------------------ Meta-test #%d %s %d-way %d-shot ------------------' % (idx_repreat, dataset_name, n_way, k_spt))
        acc_meta_test = []
        for i in range(config['num_meta_test']):
            acc_test = train_task(class_list_test, class_dict_test, train=False)
            acc_meta_test.append(acc_test)

            if (i + 1) % config['show_test_interval'] == 0:
                print('Test #%d | acc: %.4f' % (i + 1, np.mean(acc_meta_test, axis=0)))

        acc_meta_test = np.mean(acc_meta_test, axis=0)
        results[dataset_name]['%d-way %d-shot %s-avail #%d-repeat' % (
            n_way, k_spt, num_avail, idx_repreat)] = {'test_acc': acc_meta_test}
        write_file = {'result': results[dataset_name], 'config': config}
        with open(osp.join(log_path, 'results_%s_%d-way_%d-shot_%s-avail.json' % (dataset_name, n_way, k_spt, num_avail)), 'w') as f:
            json.dump(write_file, f, indent=4)

        return results


    accs = []
    for n in range(num_repeats):
        results = meta_learning(results, n)
        accs.append(results[dataset_name]['%d-way %d-shot %s-avail #%d-repeat' % (n_way, k_spt, num_avail, n)]['test_acc'])

    final_acc = np.mean(accs, axis=0)
    results[dataset_name]['%d-way %d-shot' % (n_way, k_spt)] = {'acc': final_acc}
    write_file = {'result': results[dataset_name], 'config': config}
    with open(osp.join(log_path, 'results_%s_%d-way_%d-shot_%s-avail.json' % (dataset_name, n_way, k_spt, num_avail)), 'w') as f:
        json.dump(write_file, f, indent=4)

    return final_acc, results


if __name__ == "__main__":
    folder_name = os.getcwd().split(os.sep)[-1]
    for dataset_name in ['corafull', 'dblp', 'ogbn-arxiv']:
        for n_way in [5, 10]:
            for k_spt in [3, 5]:
                config = c.config_fs(dataset_name)
                config['n_way'] = n_way
                config['k_spt'] = k_spt
                meta_learning_n_times(dataset_name, 'log', config)