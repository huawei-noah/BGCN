'''
    Copyright (C) 2019. Huawei Technologies Co., Ltd and McGill University. All rights reserved.
    This program is free software; you can redistribute it and/or modify
    it under the terms of the MIT License.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    MIT License for more details.
    '''
import numpy as np
from src.utils_gcn import load_data


def data_partition_random(dataset_dir, dataset_name, label_n_per_class):
    # Random data partition
    text_set_n = 1000
    val_set_n = 500
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels = load_data(dataset_name, dataset_dir)

    n = len(y_train)
    k = len(y_train[0])

    labels = one_hot_labels.argmax(axis=1)

    train_index_new = np.zeros(k*label_n_per_class).astype(int)
    train_mask_new = np.zeros(n).astype(bool)
    val_mask_new = np.zeros(n).astype(bool)
    test_mask_new = np.zeros(n).astype(bool)

    y_train_new = np.zeros((n, k))
    y_val_new = np.zeros((n, k))
    y_test_new = np.zeros((n, k))

    class_index_dict = {}
    for i in range(k):
        class_index_dict[i] = np.where(labels == i)[0]

    for i in range(k):
        class_index = class_index_dict[i]
        train_index_one_class = np.random.choice(class_index, label_n_per_class, replace=False)
        train_index_new[i*label_n_per_class:i*label_n_per_class + label_n_per_class] = train_index_one_class

    train_index_new = list(train_index_new)
    test_val_potential_index = list(set([i for i in range(n)]) - set(train_index_new))
    test_index_new = np.random.choice(test_val_potential_index, text_set_n, replace=False)
    potential_val_index = list(set(test_val_potential_index) - set(test_index_new))
    val_index_new = np.random.choice(potential_val_index, val_set_n, replace=False)

    train_mask_new[train_index_new] = True
    val_mask_new[val_index_new] = True
    test_mask_new[test_index_new] = True

    for i in train_index_new:
        y_train_new[i][labels[i]] = 1

    for i in val_index_new:
        y_val_new[i][labels[i]] = 1

    for i in test_index_new:
        y_test_new[i][labels[i]] = 1

    return adj, features, y_train_new, y_val_new, y_test_new, train_mask_new, val_mask_new, test_mask_new, one_hot_labels


def data_partition_fix(dataset_dir, dataset_name, label_n_per_class):
    # Data partition using the official split from Kipf's original GCN
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels = load_data(
        dataset_name, dataset_dir)
    k = len(y_train[0])
    train_set_index = np.where(train_mask == True)[0]
    labels = one_hot_labels.argmax(axis=1)
    train_set_labels = labels[train_set_index]
    train_node_index = {}
    for i in range(k):
        train_node_index[i] = np.where(train_set_labels == i)[0]

    for i in range(k):
        hide_index = train_node_index[i][label_n_per_class:]
        print("The training set index for class {} is {}".format(i, train_node_index[i][0:label_n_per_class]))
        train_mask[hide_index] = False
        y_train[hide_index] = 0

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, one_hot_labels


