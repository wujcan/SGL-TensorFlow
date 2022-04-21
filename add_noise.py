import os
import sys
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool


np.random.seed(2018)
random.seed(2018)
tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = 'XXXXXX/PythonProjects/SGL/'
    else:
        root_folder = 'XXXXXX/PythonProjects/SGL/'
    conf = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")

    dataset = Dataset(conf)
    num_users = dataset.num_users
    num_items = dataset.num_items
    train_dict = tool.csr_to_user_dict(dataset.train_matrix)
    test_dict = tool.csr_to_user_dict(dataset.test_matrix)
    num_trainings = dataset.train_matrix.nnz
    count = 0
    while count < num_trainings * conf.ratio:
        u_id = np.random.randint(num_users)
        i_id = np.random.randint(num_items)
        if i_id not in train_dict[u_id]:
            if u_id not in test_dict:
                train_dict[u_id].append(i_id)
                count += 1
            else:
                if i_id not in test_dict[u_id]:
                    train_dict[u_id].append(i_id)
                    count += 1
    with open(root_folder + '/dataset/%s_%.2f.train' % (dataset.dataset_name, conf.ratio), 'w') as fw:
        for u in train_dict:
            for i in train_dict[u]:
                outstr = '%s,%s\n' % (str(u), str(i))
                fw.write(outstr)
