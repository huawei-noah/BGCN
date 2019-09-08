'''
    Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
    This program is free software; you can redistribute it and/or modify
    it under the terms of the MIT License.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    MIT License for more details.
'''

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from src.SGD_MMSBM import MMSBM_SGMCMC
from src.utils import initialize_beta, edges_non_edges_index, get_node_neighbor_dict, get_class_node_index
from src.graph_generation import MMSBM_graph_generation
from src.utils_gcn import *
from src.models import GCN
import time
from scipy.sparse import csr_matrix
import tensorflow as tf
import numpy as np


class GnnModel:
    def __init__(self, FLAGs, node_features, one_hot_labels, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask,
                 checkpt_name,
                 model_name='BGCN'):

        self.FLAGs = FLAGs
        self.model_name = model_name
        self.node_features = node_features
        self.one_hot_labels = one_hot_labels
        self.labels = one_hot_labels.argmax(axis=1)
        self.adj = adj
        self.n = len(y_train)
        self.k = len(y_train[0])
        self.checkpt_name = checkpt_name

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.test_set_index = np.where(test_mask == True)[0]
        self.val_set_index = np.where(val_mask == True)[0]
        self.upper_tri_index = np.triu_indices(self.n, k=1)

        self.sess = None
        self.placeholder = None
        self.model = None
        self.adj_sparse_tensor_orig = None
        self.features_sparse_tensor = None

        # MMSBM model parameter
        self.MMSBM_max_itr = self.FLAGs.max_itr
        self.class_index_dict = get_class_node_index(self.labels)
        self.node_neighbors_dict = get_node_neighbor_dict(self.adj, self.n)
        self.edges, self.nonedges = edges_non_edges_index(self.adj, self.n, self.node_neighbors_dict)
        self.MMSBM_membership = 0
        self.MMSBM_hard_membership = 0
        self.MMSBM_community_strength = 0
        self.MMSBM_theta_constant = 0
        self.MMSBM_phi_constant = 0
        self.MMSBM_beta_prior = 0
        self.MMSBN_membership_prior = 0
        self.beta_prior_from_gcn = 0
        self.membership_prior_from_gcn = 0

        # prediction and loss
        self.prediction_labels = 0
        self.soft_prediction_labels_OG = 0
        self.soft_prediction_labels_sample_graph = 0
        self.MC_final_prediction = 0
        self.MC_final_prediction_soft = 0
        self.train_loss = 0
        self.val_loss = 0
        self.test_loss = 0
        self.train_acc = 0
        self.val_acc = 0
        self.test_acc = 0

    def train_block_model_estimators(self, better_initialization_flag, step_size_scaler, max_iter):
        """ Given a set of parameters to generate the graph:
        :param soft_prediction_labels: the softmax output of the GCNN
        :param prediction_label: the prediction label of the GCNN
        """
        print("============================Start training the SGD-MMSBM Model========================================")

        if better_initialization_flag:
            self.membership_prior_from_gcn = self.soft_prediction_labels_OG
            acc = accuracy_score(self.labels[self.val_set_index], self.prediction_labels[self.val_set_index])
            ARI = adjusted_rand_score(self.labels[self.val_set_index], self.prediction_labels[self.val_set_index])

            print("The ARI for the initial GCN (val set) is {}".format(ARI))
            print("The acc for the initial GCN (val set) is {}".format(acc))
            self.beta_prior_from_gcn = initialize_beta(self.adj, self.prediction_labels, self.k)

        model = MMSBM_SGMCMC(self.FLAGs, self.n, self.k, self.edges, self.nonedges,
                             self.beta_prior_from_gcn,
                             self.membership_prior_from_gcn, self.MMSBM_theta_constant,
                             self.MMSBM_phi_constant,
                             self.labels, val_set_index=self.val_set_index,
                             node_neighbors_dict=self.node_neighbors_dict,
                             better_initialization_flag=better_initialization_flag, step_size_scalar=step_size_scaler,
                             mu=1, max_iter=max_iter)

        model.model_training()
        self.MMSBM_hard_membership = model.MCMC_MMSBM_prediction_labels
        self.MMSBM_theta_constant = model.theta_constant
        self.MMSBM_phi_constant = model.phi_constant

        # get the final beta value
        beta = model.beta
        self.MMSBM_community_strength = np.ones((self.k, self.k)) * self.FLAGs.delta
        for k in range(model.k):
            self.MMSBM_community_strength[k][k] = beta[k]

        # get the final pi value/hard label
        pi_soft = model.pi
        pi_hard = pi_soft.argmax(axis=1)
        labels_one_hot = np.zeros((len(pi_hard), self.k))

        for i in range(len(pi_hard)):
            labels_one_hot[i][pi_hard[i]] = 1
        self.MMSBM_membership = labels_one_hot

    def model_initialization(self):
        self.features_sparse_tensor = preprocess_features(self.node_features)
        self.adj_sparse_tensor_orig = [preprocess_adj(self.adj)]
        num_supports = 1
        model_func = GCN

        # Define placeholders
        self.placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32,
                                              shape=tf.constant(self.features_sparse_tensor[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        self.model = model_func(self.placeholders, input_dim=self.features_sparse_tensor[2][1], logging=True)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def evaluate(self, support, mask):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(self.features_sparse_tensor, support, self.one_hot_labels, mask,
                                            self.placeholders)
        outs_val = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    def get_soft_prediction_label(self, support_graph):
        feed_dict_val = construct_feed_dict(self.features_sparse_tensor, support_graph, self.y_test, self.test_mask,
                                            self.placeholders)
        feed_dict_val.update({self.placeholders['dropout']: 0})
        soft_prediction_labels = self.sess.run(tf.nn.softmax(self.model.outputs), feed_dict=feed_dict_val)
        return soft_prediction_labels

    def train_one_epoch(self, support_graph, epoch):
        # prepare feed dict for training set
        t = time.time()
        feed_dict_train = construct_feed_dict(self.features_sparse_tensor, support_graph, self.y_train, self.train_mask,
                                              self.placeholders)
        feed_dict_train.update({self.placeholders['dropout']: self.FLAGs.dropout})
        # Training step
        outs = self.sess.run([self.model.opt_op, self.model.loss, self.model.accuracy], feed_dict=feed_dict_train)

        # Validation set
        self.val_loss, self.val_acc, duration = self.evaluate(self.adj_sparse_tensor_orig,
                                                              self.val_mask)
        # get the soft max using the original graph
        soft_labels_og_graph = self.get_soft_prediction_label(self.adj_sparse_tensor_orig)

        # get the softmax using the sampled graph
        soft_labels_sample_graphs = self.get_soft_prediction_label(support_graph)

        # Print results
        if epoch % 10 == 9:
            print("===================================================================")
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

            print("val_loss=", "{:.5f}".format(self.val_loss), "val_acc=", "{:.5f}".format(self.val_acc))

        return soft_labels_og_graph, soft_labels_sample_graphs

    def train(self):
        for epoch in range(self.FLAGs.epochs):
            if self.model_name == 'BGCN':
                # =======================================GCNN pre train process=====================================
                if epoch < self.FLAGs.pretrain_n:
                    self.soft_prediction_labels_OG, self.soft_prediction_labels_sample_graph = self.train_one_epoch(
                        self.adj_sparse_tensor_orig, epoch)
                    self.prediction_labels = self.soft_prediction_labels_OG.argmax(axis=1)

                if epoch == self.FLAGs.pretrain_n:
                    acc_gcn = accuracy_score(self.labels[self.test_set_index],
                                             self.prediction_labels[self.test_set_index])

                    print("The test set accuracy from gcn is {}".format(acc_gcn))
                    print("=========================start the BGCN training===================")
                    self.soft_prediction_labels_OG, self.soft_prediction_labels_sample_graph = self.train_one_epoch(
                        self.adj_sparse_tensor_orig, epoch)
                    self.prediction_labels = self.soft_prediction_labels_OG.argmax(axis=1)

                    self.train_block_model_estimators(better_initialization_flag=False, step_size_scaler=0.1,
                                                      max_iter=1000)
                    self.train_block_model_estimators(better_initialization_flag=True, step_size_scaler=1, max_iter=200)

                if epoch > self.FLAGs.pretrain_n:
                    # ================redo the graph inference process for every 20 iterations of GCNN ===============
                    if epoch % 20 == 0:
                        # evaluate the performance by far based on the current estimators
                        self.soft_prediction_labels_OG = self.soft_prediction_labels_OG
                        self.prediction_labels = self.soft_prediction_labels_OG.argmax(axis=1)
                        self.train_block_model_estimators(better_initialization_flag=True, step_size_scaler=1,
                                                          max_iter=200)

                    generate_graph = MMSBM_graph_generation(self.n, self.k, self.MMSBM_community_strength,
                                                            self.MMSBM_membership, self.upper_tri_index)
                    generate_graph = csr_matrix(generate_graph)
                    adj_sparse_tensor_sample = [preprocess_adj(generate_graph)]

                    self.soft_prediction_labels_OG, self.soft_prediction_labels_sample_graph = self.train_one_epoch(
                        adj_sparse_tensor_sample, epoch)

                    # ===========save the softmax output from different weight samples=================
                    if epoch > self.FLAGs.epoch_to_start_collect_weights:
                        self.MC_final_prediction_soft += self.soft_prediction_labels_sample_graph
                        self.MC_final_prediction = self.MC_final_prediction_soft.argmax(axis=1)

                if epoch == self.FLAGs.epochs - 1:
                    acc_sample_graph = accuracy_score(self.labels[self.test_set_index],
                                                      self.MC_final_prediction[self.test_set_index])
                    print("==========================================================================")
                    print("========================== Final evaluation===============================")
                    print("==========================================================================")
                    print(
                        "The accuracy (test set) from the avg graph and weight samples is {}".format(acc_sample_graph))

            elif self.model_name == 'GCN':
                if epoch < self.FLAGs.epochs:
                    self.soft_prediction_labels_OG, self.soft_prediction_labels_sample_graph = self.train_one_epoch(
                        self.adj_sparse_tensor_orig, epoch)
                    self.prediction_labels = self.soft_prediction_labels_OG.argmax(axis=1)

                if epoch == self.FLAGs.epochs - 1:
                    acc_sample_graph = accuracy_score(self.labels[self.test_set_index],
                                                      self.prediction_labels[self.test_set_index])
                    print("============= test set accuracy {}==========".format(epoch + 1))
                    print("The test set accuracy is {}".format(acc_sample_graph))

            else:
                raise ValueError('Invalid argument for model: ' + str(self.FLAGs.model))

        print("Optimization Finished!")
