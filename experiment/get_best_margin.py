import os
import numpy as np
from experiment.triplet_loss_baseline import triplet_loss_with_knn_exp
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run():

    # margins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 3.0, 5.0]
    margins = [1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 2.8, 3.3, 3.8, 4.1, 4.8, 5.1, 5.7, 6.1, 7.3, 9.2, 11.9]

    kwargs = {
        'ckpt_prefix': 'Run06',
        'device': '1',
        'lr': 1e-3,
        'embedding_epochs': 200,
        'classify_epochs': 3,
        'n_classes': 10,
        'n_samples': 12,
        'margin': 1.0,
        'log_interval': 80,
        'k': 3,
        'squared': False,
        'embed_dims': 64,
        'embed_net': 'vgg',
        'is_train_embedding_model': True,
        'using_pretrain': False,
        'batch_size': 128,
        'select_method': 'batch_all',
        'soft_margin': False
    }

    for idx, margin in enumerate(margins):
        ckpt_pf = 'R10_' + str(idx+13) + '_margin_' + str(margin)
        kwargs['ckpt_prefix'] = ckpt_pf
        kwargs['margin'] = margin
        triplet_loss_with_knn_exp(**kwargs)

    ckpt_root = os.path.join(ROOT_DIR, 'ckpt/batch_all_with_knn_exp')
    run_list = os.listdir(ckpt_root)
    matched = []
    for fname in run_list:
        if fname.startswith('R10') and fname.endswith('tar'):
            matched.append(fname)

    acc = []
    import re
    for s in matched:
        acc_str = re.search('acc_(.*)\.tar', s).group(1)
        acc.append(float(acc_str))

    best_idx = np.argmax(acc)
    best_fname = matched[best_idx]


    print(best_fname)


if __name__ == '__main__':
    run()