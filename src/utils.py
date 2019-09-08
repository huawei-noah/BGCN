"""
    Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
    This program is free software; you can redistribute it and/or modify
    it under the terms of the MIT License.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    MIT License for more details.
    
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import os, sys


def save_log_func(code_path, dataset, model_name, trial_index, data_partition_seed):

    # Create log Directory and save log to txt

    log_dir = code_path + '/log/'
    if not os.path.exists(os.path.dirname(log_dir)):
        os.makedirs(os.path.dirname(log_dir))
    log_file_name = dataset + '_' + model_name + '_softmax_trail_' + str(trial_index) + '_random_seed_' + str(
        data_partition_seed) + '.txt'
    sys.stdout = open(log_dir + log_file_name, 'w')


def get_one_hot_encode_labels(label, K):
    one_hot_label = np.zeros((len(label), K))
    for i in range(len(label)):
        one_hot_label[i][label[i]] = 1
    return one_hot_label

def convert_edge_list(csr_adj_matrix):
    coo_adj_matrix = coo_matrix(csr_adj_matrix)
    row = coo_adj_matrix.row
    col = coo_adj_matrix.col
    edge_list = np.zeros((2, len(row))).astype(int)
    edge_list[0,:] = row
    edge_list[1,:] = col
    edge_list = edge_list.transpose()
    return edge_list


def bernuli_distrbution(y, p):
    bernuli_dist = (p**y)*((1-p)**(1-y))
    return bernuli_dist


def graph_preparation(edges, nonedges, test_edges_n):
    nonedges_n = len(nonedges[0])
    edges_n = len(edges[0])
    # prepare test edges
    test_edges_index = list(np.random.randint(0, edges_n, size = int(test_edges_n/2)))
    test_non_edges_index = list(np.random.randint(0, nonedges_n, size = int(test_edges_n/2)))

    test_edges = [(edges[0][i], edges[1][i]) for i in test_edges_index]
    test_non_edges = [(nonedges[0][i], nonedges[1][i]) for i in test_non_edges_index]
    test_set = test_edges + test_non_edges  #
    y_test_set = np.zeros(test_edges_n)
    y_test_set[0:int(test_edges_n/2)] = 1

    return edges_n, nonedges_n, test_set, y_test_set

def edges_non_edges_index(adj, N, node_neighbors_dict):
    A_coo = coo_matrix(adj)
    A_coo_data = A_coo.data
    diagnol_element = np.array([adj[i,i] for i in range(N)])
    self_loop_index = np.where(diagnol_element == 1)[0]
    self_loop_n = len(self_loop_index)
    links_n = len(np.where(A_coo_data != 0)[0]) - self_loop_n
    links_n = int(links_n/2)
    non_links_n = int((N * N - N)/2 - links_n)

    nonedges_index_a = np.zeros(non_links_n).astype(int)
    nonedges_index_b = np.zeros(non_links_n).astype(int)

    edges_index_a = np.zeros(links_n).astype(int)
    edges_index_b = np.zeros(links_n).astype(int)

    N_list_set = np.array([i for i in range(N)])

    start_edges = 0
    start_non_edges = 0
    for i in range(N):
        # deal with links
        node_i_neighbors = node_neighbors_dict[i]
        node_i_upper_tri_index = np.arange(i+1 , N)
        node_i_neighbors_upper_tri = np.intersect1d(node_i_neighbors, node_i_upper_tri_index)
        end_edges = start_edges + len(node_i_neighbors_upper_tri)
        edges_index_a[start_edges:end_edges] = i
        edges_index_b[start_edges:end_edges] = node_i_neighbors_upper_tri

        start_edges = end_edges

        # deal with non-links
        node_i_non_neighbor = np.setdiff1d(N_list_set, node_i_neighbors)
        node_i_non_neighbor_tri = np.intersect1d(node_i_non_neighbor, node_i_upper_tri_index)
        node_i_non_neighbor_n = len(node_i_non_neighbor_tri)

        end_non_edges = start_non_edges + node_i_non_neighbor_n
        nonedges_index_a[start_non_edges:end_non_edges] = i
        nonedges_index_b[start_non_edges:end_non_edges] = node_i_non_neighbor_tri
        start_non_edges = end_non_edges

    nonedges = (nonedges_index_a, nonedges_index_b)
    edges = (edges_index_a, edges_index_b)
    return edges, nonedges


def vectorized_multinomial(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    r = r[:,np.newaxis]
    k = (s < r).sum(axis=1)
    return items[k]

def reparameterized_to_beta(theta):  # comunity strength
    theta_constant = np.sum(theta, axis = 1)
    beta = theta[:,1]/theta_constant
    return beta, theta_constant


def initialize_theta_phi_with_better_initialization(beta, pi, theta_constant, phi_constant, K):

    phi = pi*phi_constant
    theta = np.zeros((K, 2))
    theta[:, 1] = theta_constant*beta
    theta[:, 0] = theta_constant - theta[:, 1]
    return theta, phi


def reparameterized_to_pi(phi, N):  #
    row_sum_phi = (np.sum(phi, axis = 1)).reshape(N, 1)
    pi = phi/row_sum_phi
    return pi, row_sum_phi


def step_size_function(itr_index, tao, scaler):
    step_size = (tao + itr_index) ** (-0.5)
    return step_size/scaler


def z_constant(beta, node_a_membership, node_b_membership, observation_ab, delta):
    K = len(node_a_membership)
    temp = bernuli_distrbution(observation_ab, delta)
    Z_ab = temp
    for k in range(K):
        Z_ab += (bernuli_distrbution(observation_ab, beta[k]) - temp)* node_a_membership[k] * node_b_membership[k]
    return Z_ab


def metric_perp_avg(beta_samples, pi_samples, test_edge_set, y_test, delta):
    sum_edges_perplexity = 0
    for edge in range(len(test_edge_set)):
        a = test_edge_set[edge][0]
        b = test_edge_set[edge][1]
        p_edge = 0
        for i in range(min(20, len(beta_samples))):
            p_edge += z_constant(beta_samples[i], pi_samples[i][a], pi_samples[i][b], y_test[edge], delta)
        sum_edges_perplexity += np.log(1.0/min(20, len(beta_samples)) * p_edge)


    perplexity = np.exp(-sum_edges_perplexity / len(test_edge_set))

    return perplexity


def get_node_neighbor_dict(adj, N):
    node_neighbors_dict = {}
    for i in range(N):
        node = adj[i]
        node_neighbors_dict[i] = csr_matrix.nonzero(node)[1]
    return node_neighbors_dict


def get_class_node_index(labels):
    class_index_dict = {}  # dictionary for the class index
    for i in range(max(labels) + 1):
        class_index_dict[i] = np.where(labels == i)[0]
    return class_index_dict


def labels_to_one_hot_labels(labels, K):
    labels_one_hot = np.zeros((len(labels), K))
    for i in range(len(labels)):
        labels_one_hot[i][labels[i]] = 1
    return labels_one_hot


def accuracy_avg(pi_samples, initial_prediction_labels, true_label, N, K, val_set_index):
    count = np.zeros((N, K))
    for i in range(len(pi_samples)):
        predict_label = pi_samples[i].argmax(axis=1)
        labels_one_hot = np.zeros(pi_samples[i].shape)
        for j in range(len(predict_label)):
            labels_one_hot[j][predict_label[j]] = 1

        if i == 0:
            count = labels_one_hot
        else:
            count += labels_one_hot

    avg_predict_label = count.argmax(axis=1)
    ARI = adjusted_rand_score(true_label[val_set_index], avg_predict_label[val_set_index])
    acc = accuracy_score(true_label[val_set_index], avg_predict_label[val_set_index])
    change_from_the_initial_prediction_labels = len(np.where(initial_prediction_labels == avg_predict_label)[0])/float(N)
    return ARI, acc, change_from_the_initial_prediction_labels, avg_predict_label


def initialize_beta(A_real, prediction_label, K):
    class_index = {}
    initial_beta = np.zeros(K)
    for k in range(K):
        class_index[k] = np.where(prediction_label == k)[0]
    for k in range(K):
        select_nodes = A_real[class_index[k]]
        select_nodes = select_nodes[:,class_index[k]]
        block_links = np.sum(select_nodes)
        initial_beta[k] = float(block_links)/(len(class_index[k])**2 + 1)
    return initial_beta

