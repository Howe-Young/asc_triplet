import os
import pickle
import logging
import numpy as np
from xgboost import XGBClassifier


def xgb_cls(train_data, train_label, val_data, val_label, exp_dir, **kwargs):
    """

    :param train_data:  (nb_sample, nb_features)
    :param train_label: (nb_sample,)
    :param val_data:    (nb_sample, nb_features)
    :param val_label:   (nb_sample,)
    :param exp_dir:     experiment dir to save trained xgb classifier, string
    :param kwargs: optional args for xgboost
    :return:
    """
    clf = XGBClassifier(max_depth=kwargs.get('max_depth', 6),
                        n_estimators=kwargs.get('n_estimators', 200),
                        objective='multi:softmax',
                        n_jobs=10)

    clf.fit(train_data, train_label,
            eval_set=[(val_data, val_label)],
            verbose=True,
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 20)
            )

    train_acc = clf.score(train_data, train_label)
    val_acc = clf.score(val_data, val_label)
    logging.info("training seg_acc:{:.3f}, val seg_acc:{:.3f}".format(train_acc, val_acc))

    cls_model_file = "{}/xgb_acc{:.3f}.pkl".format(exp_dir, val_acc)
    pickle.dump(clf, open(cls_model_file, 'wb'))
