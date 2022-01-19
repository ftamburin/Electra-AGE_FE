from __future__ import division
from __future__ import print_function
import os, sys
import argparse
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics

#------------------------------------------------------------------------------------
#from model import LinTrans, LogReg

from torch.nn.modules.module import Module

class SampleDecoder(Module):
	def __init__(self, act=torch.sigmoid):
		super(SampleDecoder, self).__init__()
		self.act = act

	def forward(self, zx, zy):
		sim = (zx * zy).sum(1)
		sim = self.act(sim)
	
		return sim


class LinTrans(Module):
	def __init__(self, layers, dims):
		super(LinTrans, self).__init__()
		self.layers = torch.nn.ModuleList()
		for i in range(layers):
			self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
		self.dcs = SampleDecoder(act=lambda x: x)

	def scale(self, z):
		
		zmax = z.max(dim=1, keepdim=True)[0]
		zmin = z.min(dim=1, keepdim=True)[0]
		z_std = (z - zmin) / (zmax - zmin)
		z_scaled = z_std
	
		return z_scaled

	def forward(self, x):
		out = x
		for layer in self.layers:
			out = layer(out)
		out = self.scale(out)
		out = F.normalize(out)
		return out


class LogReg(Module):
	def __init__(self, ft_in, nb_classes):
		super(LogReg, self).__init__()
		self.fc = torch.nn.Linear(ft_in, nb_classes)

	def weights_init(self, m):
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, seq):
		ret = self.fc(seq)
		return ret

	
#------------------------------------------------------------------------------------
#from utils import *
import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.preprocessing as preprocess

def sample_mask(idx, l):
	"""Create mask."""
	mask = np.zeros(l)
	mask[idx] = 1
	return np.array(mask, dtype=np.bool)


def load_data(path, dataset):
	# load the data: x, tx, allx, graph
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []

	for i in range(len(names)):
		'''
		fix Pickle incompatibility of numpy arrays between Python 2 and 3
		https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
		'''
		with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as rf:
			u = pkl._Unpickler(rf)
			u.encoding = 'latin1'
			cur_data = u.load()
			objects.append(cur_data)
		# objects.append(
		#	 pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file(
		"{}/ind.{}.test.index".format(path, dataset))
	test_idx_range = np.sort(test_idx_reorder)

	features = sp.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	features = torch.FloatTensor(np.array(features.todense()))
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
	
	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]
	
	idx_test = test_idx_range.tolist()
	idx_train = range(len(y))
	#idx_val = range(len(y), len(y) + 500)
	idx_val = range(len(y), len(y))

	train_mask = sample_mask(idx_train, labels.shape[0])
	val_mask = sample_mask(idx_val, labels.shape[0])
	test_mask = sample_mask(idx_test, labels.shape[0])

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)
	y_train[train_mask, :] = labels[train_mask, :]
	y_val[val_mask, :] = labels[val_mask, :]
	y_test[test_mask, :] = labels[test_mask, :]

	return adj, features, np.argmax(labels, 1), idx_train, idx_val, idx_test

def parse_index_file(filename):
	index = []
	for line in open(filename):
		index.append(int(line.strip()))
	return index


def sparse_to_tuple(sparse_mx):
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape


def mask_test_edges(adj):
	# Function to build test set with 10% positive links
	# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
	# TODO: Clean up.

	# Remove diagonal elements
	adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
	adj.eliminate_zeros()
	# Check that diag is zero:
	assert np.diag(adj.todense()).sum() == 0

	adj_triu = sp.triu(adj)
	adj_tuple = sparse_to_tuple(adj_triu)
	edges = adj_tuple[0]
	edges_all = sparse_to_tuple(adj)[0]
	#num_test = int(np.floor(edges.shape[0] / 10.))
	#num_val = int(np.floor(edges.shape[0] / 20.))
	num_test = int(np.floor(edges.shape[0]))
	num_val = int(np.floor(edges.shape[0]))

	all_edge_idx = list(range(edges.shape[0]))
	np.random.shuffle(all_edge_idx)
	val_edge_idx = all_edge_idx[:num_val]
	test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
	#test_edges = edges[test_edge_idx]
	test_edges = edges[val_edge_idx]
	val_edges = edges[val_edge_idx]
	#train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
	train_edges = edges

	def ismember(a, b, tol=5):
		rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
		return np.any(rows_close)

	test_edges_false = []
	while len(test_edges_false) < len(test_edges):
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue
		if ismember([idx_i, idx_j], edges_all):
			continue
		if test_edges_false:
			if ismember([idx_j, idx_i], np.array(test_edges_false)):
				continue
			if ismember([idx_i, idx_j], np.array(test_edges_false)):
				continue
		test_edges_false.append([idx_i, idx_j])

	val_edges_false = []
	while len(val_edges_false) < len(val_edges):
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue
		if ismember([idx_i, idx_j], train_edges):
			continue
		if ismember([idx_j, idx_i], train_edges):
			continue
		if ismember([idx_i, idx_j], val_edges):
			continue
		if ismember([idx_j, idx_i], val_edges):
			continue
		if val_edges_false:
			if ismember([idx_j, idx_i], np.array(val_edges_false)):
				continue
			if ismember([idx_i, idx_j], np.array(val_edges_false)):
				continue
		val_edges_false.append([idx_i, idx_j])

	#assert ~ismember(test_edges_false, edges_all)
	#assert ~ismember(val_edges_false, edges_all)
	#assert ~ismember(val_edges, train_edges)
	#assert ~ismember(test_edges, train_edges)
	#assert ~ismember(val_edges, test_edges)

	data = np.ones(train_edges.shape[0])

	# Re-build adj matrix
	adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
	adj_train = adj_train + adj_train.T

	# NOTE: these edge lists only contain single direction of edge!
	return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj, layer, norm='sym', renorm=True):
	adj = sp.coo_matrix(adj)
	ident = sp.eye(adj.shape[0])
	if renorm:
		adj_ = adj + ident
	else:
		adj_ = adj
	
	rowsum = np.array(adj_.sum(1))
	
	if norm == 'sym':
		degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
		adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
		laplacian = ident - adj_normalized
	elif norm == 'left':
		degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
		adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
		laplacian = ident - adj_normalized
		

	reg = [2/3] * (layer)

	adjs = []
	for i in range(len(reg)):
		adjs.append(ident-(reg[i] * laplacian))
	return adjs


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	# Predict on test set of edges
	adj_rec = np.dot(emb, emb.T)
	preds = []
	pos = []
	for e in edges_pos:
		preds.append(sigmoid(adj_rec[e[0], e[1]]))
		pos.append(adj_orig[e[0], e[1]])

	preds_neg = []
	neg = []
	for e in edges_neg:
		preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
		neg.append(adj_orig[e[0], e[1]])

	preds_all = np.hstack([preds, preds_neg])
	labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
	roc_score = roc_auc_score(labels_all, preds_all)
	ap_score = average_precision_score(labels_all, preds_all)

	return roc_score, ap_score

#------------------------------------------------------------------------------------


def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
	f_adj = np.matmul(z, np.transpose(z))
	cosine = f_adj
	cosine = cosine.reshape([-1,])
	pos_num = round(upper_threshold * len(cosine))
	neg_num = round((1-lower_treshold) * len(cosine))
	
	pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
	neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
	
	return np.array(pos_inds), np.array(neg_inds)

def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
	upth = upper_threshold + up_eta
	lowth = lower_treshold + low_eta
	return upth, lowth


'''
def gae_for():

	args_gnnlayers = 8
	args_linlayers = 1
	args_epochs = 500
	args_lr = 0.001
	args_upth_st = 0.0011
	args_lowth_st = 0.1
	args_upth_ed = 0.001
	args_lowth_ed = 0.5
	args_upd = 1
	args_dataset = 'fn1.5U'
	args_dims = [768]

	print("Using {} dataset".format(args_dataset))
 	
	adj, features, _, idx_train, idx_val, idx_test = load_data(args_dataset)
	print(idx_train, idx_val, idx_test)
	n_nodes, feat_dim = features.shape
	dims = [feat_dim] + args_dims
	layers = args_linlayers
	# Store original adjacency matrix (without diagonal entries) for later
	adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
	adj.eliminate_zeros()
	adj_orig = adj
	adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
	adj = adj_train
	n = adj.shape[0]
	adj_norm_s = preprocess_graph(adj, args_gnnlayers, norm='sym', renorm=True)
	sm_fea_s = sp.csr_matrix(features).toarray()
	print('Laplacian Smoothing...')
	for a in adj_norm_s:
		sm_fea_s = a.dot(sm_fea_s)
	adj_1st = (adj + sp.eye(n)).toarray()
	adj_label = torch.FloatTensor(adj_1st)
	sm_fea_s = torch.FloatTensor(sm_fea_s)
	adj_label = adj_label.reshape([-1,])
	inx = torch.tensor(sm_fea_s, requires_grad=True).cuda()
	adj_label = adj_label.cuda()
	pos_num = len(adj.indices)
	neg_num = n_nodes*n_nodes-pos_num
	up_eta = (args_upth_ed - args_upth_st) / (args_epochs/args_upd)
	low_eta = (args_lowth_ed - args_lowth_st) / (args_epochs/args_upd)
	pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args_upth_st, args_lowth_st, pos_num, neg_num)
	upth, lowth = update_threshold(args_upth_st, args_lowth_st, up_eta, low_eta)
	pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
	bs = 1024

	model = LinTrans(layers, dims)

	print(model)
	model.cuda()
	optimizer = optim.Adam(model.parameters(), lr=args_lr)
	for epoch in range(args_epochs):
		
		model.train()
		sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=bs)).cuda()
		sampled_pos = torch.LongTensor(np.random.choice(pos_inds_cuda.cpu(), size=bs)).cuda()
		sampled_inds = torch.cat((sampled_pos, sampled_neg), 0)
		optimizer.zero_grad()
		xind = sampled_inds // n_nodes
		yind = sampled_inds % n_nodes
		x = torch.index_select(inx, 0, xind)
		y = torch.index_select(inx, 0, yind)
		zx = model(x)
		zy = model(y)
		batch_label = torch.cat((torch.ones(bs), torch.zeros(bs))).cuda()
		batch_pred = model.dcs(zx, zy)
		# LOSS ON PREDICTED ADJ
		loss = F.binary_cross_entropy_with_logits(batch_pred, batch_label)
		loss.backward()
		cur_loss = loss.item()
		optimizer.step()
			
		model.eval()
		mu = model(inx)
		hidden_emb = mu.cpu().data.numpy()
		upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
		pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
		pos_inds_cuda = torch.LongTensor(pos_inds).cuda()

		# EVAL
		val_auc, val_ap = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
		print("AGE: {}, train_loss_gae={:.5f}, val_auc={:.5f}, val_ap={:.5f}, VAL_={:.5f}".format(
			epoch + 1, cur_loss, val_auc, val_ap, val_auc+val_ap))
			
if __name__ == '__main__':
	gae_for()
'''

