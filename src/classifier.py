import os
import sys
import math
import numpy as np
import scipy.sparse as sp
import random
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.loss import _Loss
from collections import defaultdict
import pickle
from sklearn.preprocessing import normalize

from AGE_LP import *


# Generic classifier, doesn't do much
class Classifier:
	def __init__(self, lexicon, all_unknown=False, num_components=False, max_sampled=False, num_epochs=False):
		self.clf = None
		self.lexicon = lexicon
		self.all_unknown = all_unknown
		self.num_components = num_components
		self.max_sampled = max_sampled
		self.num_epochs = num_epochs

	def train(self, X, y, lemmapos):
		raise NotImplementedError("Not implemented, use child classes")
	def predict(self, X, lemmapos):
		raise NotImplementedError("Not implemented, use child classes")



def count_parameters(model):
	tp  = sum(p.numel() for p in model.parameters() if p.requires_grad)
	ntp = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	return tp, ntp


#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
	def __init__(self, T, y, yFE, ncl):
		'Initialization'
		self.T = T
		self.y = y
		self.yFE = yFE
		self.ncl = ncl

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.T)

	def __getitem__(self, index):
		'Generates one sample of data'
		T = self.T[index]
		y = self.y[index]
		yFE = self.yFE[index]
		return T, y, yFE



#----------------------------------------------------------------------
#----------------------------------------------------------------------
# define the NN architecture
class Net(torch.nn.Module):
	def __init__(self, in_features, out_features, out_embsfea, Endx, out_FE, dev):
		super(Net, self).__init__()
		print('Net IN,OUTF,OUTFE',in_features,out_features,out_FE)
		self.inF = in_features
		self.outF = out_features
		self.outFE = out_features
		hidden_dim = 256 #int(in_features[-1]/6)

		self.AGE_model = LinTrans(1, [768, 768])
		self.Endx = torch.tensor(Endx).to(dev)
		self.prjT = torch.nn.Linear(in_features[-1], hidden_dim)
		self.prjL = torch.nn.Linear(out_embsfea, hidden_dim)
		self.outW = Parameter(torch.Tensor(out_features, hidden_dim))
		self.outb = Parameter(torch.Tensor(out_features))
		torch.nn.init.kaiming_uniform_(self.outW, a=math.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.outW)
		bound = 1 / math.sqrt(fan_in)
		torch.nn.init.uniform_(self.outb, -bound, bound)
		self.fc3 = torch.nn.Linear(hidden_dim, out_FE)

	def forward(self, t, AGE_inx, dev):
		AGE_E = self.AGE_model(AGE_inx)
		h = torch.tanh(self.prjT(t))		# Bxh
		E = torch.tanh(self.prjL(AGE_E[self.Endx]))	# Cxh
		hp = h.unsqueeze(1).repeat(1,E.shape[0],1)	# BxCxh
		Ep = E.unsqueeze(0).repeat(h.shape[0],1,1)	# BxCxh
		gjoint = torch.mul(hp, Ep)	# BxCxh
		o = torch.mul(gjoint, self.outW.unsqueeze(0).repeat(h.shape[0],1,1))  # BxCxh
		o_c = torch.sum(o, dim=-1)	#	BxC
		o_c = torch.add(o_c, self.outb.unsqueeze(0).repeat(h.shape[0],1))
		o_f = self.fc3(h)	# BxFE
		return o_c, o_f


class AdvDNNClassifier(Classifier):
	def __init__(self, lexicon, all_unknown=False, num_components=False, max_sampled=False, num_epochs=False, inF=None, outF=None, model_file=None, lexdir=None, dataset=None):
		super(AdvDNNClassifier, self).__init__(lexicon,all_unknown,num_components,max_sampled,num_epochs)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print('Working on device',self.device)
		self.model_file = model_file
		self.fr2id = lexicon.frameToId
		self.id2fr = lexicon.idToFrame
		torch.set_printoptions(profile="full")

		# HIERARCHY 
		self.lexdir = lexdir
		self.dataset = dataset
		print('LexDir:',self.lexdir)
		print('Dataset:',self.dataset)

		afile = self.lexdir+'/AGE_data/'+self.dataset+'U.dict'
		print('AGE DIctionary from',afile)
		embsNdx = pickle.load(open(afile, "rb"))
		self.embs2fr = [0]*len(embsNdx)
		for k,v in embsNdx.items():
			self.embs2fr[self.fr2id[k]] = v

		self.gamma1 = 0.5
		self.gamma2 = 0.1

		self.inF = inF
		self.outF = outF
		self.outFE = lexicon.get_number_of_FEs()
		self.outE = 768
		self.model = Net(inF, outF, self.outE, self.embs2fr, self.outFE, self.device)
		print(self.model)
		tp, ntp = count_parameters(self.model)
		print('Trainable pars:',tp)
		print('Non-Trainable pars:',ntp)
		self.model.to(self.device)

		self.loss_c = torch.nn.CrossEntropyLoss()
		self.loss_f = torch.nn.BCEWithLogitsLoss()
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-05, weight_decay=1e-4) # 3e-05 PAFIBERT



	def train(self, Tt, yt, yFEt, lpt, Tv, yv, yFEv, lpv, Ttt, ytt, yFEtt, lptt):
		args_gnnlayers = 8
		args_linlayers = 1
		args_epochs = 500
		args_lr = 0.001
		args_upth_st = 0.0011
		args_lowth_st = 0.1
		args_upth_ed = 0.001
		args_lowth_ed = 0.5
		args_upd = 1
		print("AGE: Using {} dataset from {}".format(self.dataset+'U', self.lexdir+'/AGE_data'))
		adj, features, _, idx_train, idx_val, idx_test = load_data(self.lexdir+'/AGE_data', self.dataset+'U')
		n_nodes, feat_dim = features.shape
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
		print('AGE: Laplacian Smoothing...')
		for a in adj_norm_s:
			sm_fea_s = a.dot(sm_fea_s)
		adj_1st = (adj + sp.eye(n)).toarray()
		adj_label = torch.FloatTensor(adj_1st)
		sm_fea_s = torch.FloatTensor(sm_fea_s)
		adj_label = adj_label.reshape([-1,])
		inx = sm_fea_s.clone().detach().requires_grad_(True).cuda()
		adj_label = adj_label.cuda()
		pos_num = len(adj.indices)
		neg_num = n_nodes*n_nodes-pos_num
		up_eta = (args_upth_ed - args_upth_st) / (args_epochs/args_upd)
		low_eta = (args_lowth_ed - args_lowth_st) / (args_epochs/args_upd)
		pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args_upth_st, args_lowth_st, pos_num, neg_num)
		upth, lowth = update_threshold(args_upth_st, args_lowth_st, up_eta, low_eta)
		pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
		AGE_bs = 512

		batch_size = 32
		trainset = Dataset(Tt, yt, yFEt, self.outF)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

		epoch = 1
		bestVacc = 0
		pat = 0
		avgV = [0.0]*10
		train_loss = 1.0
		while ((pat < 10) or (train_loss_c > 0.2)):
			# monitor training loss
			train_loss = train_loss_c = train_loss_e = train_loss_f = 0.0
	
			# train the model 
			self.model.train()
			for t, target, targetFE in trainloader:
				# clear the gradients of all optimized variables
				self.optimizer.zero_grad()
				sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=AGE_bs)).cuda()
				sampled_pos = torch.LongTensor(np.random.choice(pos_inds_cuda.cpu(), size=AGE_bs)).cuda()
				sampled_inds = torch.cat((sampled_pos, sampled_neg), 0)
				xind = sampled_inds // n_nodes
				yind = sampled_inds % n_nodes
				x = torch.index_select(inx, 0, xind)
				y = torch.index_select(inx, 0, yind)
				zx = self.model.AGE_model(x)
				zy = self.model.AGE_model(y)
				batch_label = torch.cat((torch.ones(AGE_bs), torch.zeros(AGE_bs))).cuda()
				batch_pred = self.model.AGE_model.dcs(zx, zy)
				loss_e = torch.nn.functional.binary_cross_entropy_with_logits(batch_pred, batch_label)

				t, target, targetFE = t.to(self.device), target.to(self.device), targetFE.to(self.device)
				# forward pass
				output_c, output_f = self.model(t, inx.to(self.device), self.device)
				# calculate the losses
				loss_c = self.loss_c(output_c, target)
				loss_f = self.loss_f(output_f, targetFE)
				loss = self.gamma2 * (self.gamma1 * loss_c + (1-self.gamma1) * loss_e) + (1-self.gamma2) * loss_f
				# backward pass
				loss.backward()
				# perform a single optimization step (parameter update)
				self.optimizer.step()
				# update running training loss
				train_loss += loss.item()*t.size(0)
				train_loss_c += loss_c.item()*t.size(0)
				train_loss_e += loss_e.item()*t.size(0)
				train_loss_f += loss_f.item()*t.size(0)
			 
			self.model.eval()
			E = self.model.AGE_model(inx)
			hidden_emb = E.cpu().data.numpy()
			upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
			pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
			pos_inds_cuda = torch.LongTensor(pos_inds).cuda()

			# calculate average loss over an epoch
			train_loss = train_loss/len(trainloader.sampler)
			train_loss_c = train_loss_c/len(trainloader.sampler)
			train_loss_e = train_loss_e/len(trainloader.sampler)
			train_loss_f = train_loss_f/len(trainloader.sampler)
		
			# EVAL
			val_auc, val_ap = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
			print("AGE: {}, train_loss_gae={:.5f}, val_auc={:.5f}, val_ap={:.5f}, VAL_={:.5f}".format(
					epoch, loss_e, val_auc, val_ap, val_auc+val_ap))
			# EVALUATE ON VALIDATION and TEST SETs
			TRacc = self.evaluate(Tt, yt, lpt, inx)
			Vacc = self.evaluate(Tv, yv, lpv, inx)
			Tacc = self.evaluate(Ttt, ytt, lptt, inx)

			avgV.pop(0)
			avgV.append(Vacc)

			# print training statistics 
			print('Epoch: %3d  Loss: %6.5f (%6.5f,%6.5f,%6.5f)  Accs: %8.7f, %8.7f(%6.5f), %8.7f  Pat: %d' % (epoch, train_loss, train_loss_c, train_loss_e, train_loss_f, TRacc, Vacc, sum(avgV)/len(avgV), Tacc, pat), end='')
			#print('Epoch: %3d  Loss: %6.5f (%6.5f,%6.5f,%6.5f)  Accs: %8.7f, %8.7f, %8.7f  Pat: %d' % (epoch, train_loss, train_loss_c, train_loss_e, train_loss_f, TRacc, Vacc, Tacc, pat), end='')

			Vacc = sum(avgV) / len(avgV) 
			if (Vacc > bestVacc):
				torch.save({'model':self.model.state_dict(), 'inx': inx, 'lexicon': self.lexicon}, self.model_file)
				print('*')
				bestVacc = Vacc
				pat = 0
			else:
				if (epoch > 10):
					pat += 1
				print()

			sys.stdout.flush()
			epoch += 1


	def predict(self, T, lemmapos, inx):
		available_frames = self.lexicon.get_available_frame_ids(lemmapos)  # get available frames from lexicon
		ambig = self.lexicon.is_ambiguous(lemmapos)
		unknown = self.lexicon.is_unknown(lemmapos)  # unknown = not in lexicon

		bestScore = None
		bestClass = None
		if unknown or self.all_unknown:  # the all_unknown setting renders all lemma.pos unknown!
			available_frames = self.lexicon.get_all_frame_ids()  # if the lemma.pos is unknown, search in all frames
		else:
			if not ambig:
				# if the LU is known and has only one frame, just return it. Even if there is no data for this LU (!)
				bestClass = available_frames[0]

		# JOIN POSSIBLE FEs FOR EACH AVAILABLE FRAME
		available_FEs = []
		for af in available_frames:
			available_FEs += self.lexicon.frameToFE[af]
		available_FEs.append(self.lexicon.FEToId['NONE'])
		available_FEs = list(set(available_FEs))

		self.model.eval()
		T = T.unsqueeze(0)
		o_c, o_f = self.model(T.to(self.device), inx.to(self.device), self.device)
		y = torch.squeeze(o_c)
		o_f = torch.squeeze(o_f)

		if (bestClass == None):
			expA_F = []
			inv_expA_F = {}
			for cl in available_frames:
				if (cl < len(y)):
					score = y[cl]
					if ((bestScore is None) or (score >= bestScore)):
						bestScore = score
						bestClass = cl
		return bestClass, (o_f > 0.0).int(), available_FEs


	def evaluate(self, T, y, lp, inx):
		total = 0
		correct = 0
		for t_, y_true, lp_ in zip(T, y, lp):
			y_predicted, _, _ = self.predict(t_, lp_, inx)
			correct += int(y_true == y_predicted)
			total += 1
		acc = correct / total if total != 0 else 0
		return acc
