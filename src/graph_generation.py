"""
Created on Wed Sep 26 10:02:05 2018

@author: y84092891: Yingxue Zhang
"""
import numpy as np
from scipy.sparse import coo_matrix


def vectorized_multinomial(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    r = r[:,np.newaxis]
    k = (s < r).sum(axis=1)
    return items[k]


def MMSBM_graph_generation(N, K, B_real, membership, upper_tri_index):
    membership = membership.argmax(axis=1)
    Z_ab = (membership[upper_tri_index[0]])
    Z_ba = (membership[upper_tri_index[1]])
    B_real_flatten = B_real.flatten()
    upper_tri_value = np.random.uniform(size=int(N*(N-1)/2))
    upper_tri_value = upper_tri_value < B_real_flatten[Z_ab+Z_ba*K]

    upper_link_index = np.where(upper_tri_value == True)[0]
    upper_link_index_row = (upper_tri_index[0])[upper_link_index]
    upper_link_index_col = (upper_tri_index[1])[upper_link_index]

    link_index_row = np.concatenate((upper_link_index_row, upper_link_index_col),axis = 0)
    link_index_col = np.concatenate((upper_link_index_col, upper_link_index_row), axis = 0)
    data = np.ones(len(link_index_row))
    A_real_sparse = coo_matrix((data, (link_index_row, link_index_col)), shape=(N, N))
    return A_real_sparse
