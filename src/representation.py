import sys
import numpy as np
import pickle
import torch
from networkx.algorithms.dag import ancestors, descendants


# Feature mappers convert graphs into matrices given lexicon and vsm
class FeatureMapper:
	def __init__(self, HOME, vsm, lexicon, mwa=False):
		self.HOME = HOME
		self.vsm = vsm
		self.lexicon = lexicon
		self.multiword_averaging = mwa

	def get_repr(self, graph):
		raise NotImplementedError("Not implemented")

	def get_matrix(self, graph_list):
		ngraphs = len(graph_list)
		Y = []
		YFE = []
		T = []
		lemmapos = []
		gid = []
		j = 0
		print(len(graph_list),graph_list[-1].sid)
		for g in graph_list:
			t, y, yFE = self.get_repr(g, self.lexicon)
			T += [t]
			Y += [y]
			YFE += [yFE]
			lemmapos += [g.get_predicate_head()["lemmapos"]]
			gid += [g.gid]
			if (j % 100 == 0):
				print(j,'/',ngraphs)
				sys.stdout.flush()
			#if (j > 0) and (j % 100 == 0):
			#	break
			j += 1
		T = torch.stack(T)	
		Y = np.array(Y, dtype=np.int)
		YFE = np.array(YFE, dtype=np.float32)
		return T, Y, YFE, lemmapos, gid

	def save_BERT(self, graph_list, HOME, corpus, embsl):
		j = 0
		ngraphs = len(graph_list)
		bert = {}
		for g in graph_list:
			sk, s, we, se, ael, at = self.get_BERTrepr(g)
			if (j % 100 == 0):
				print(j,'/',ngraphs)
				sys.stdout.flush()
			#bert[sk] = (s, we, se, ael, at)
			bert[sk] = (s, we)
			j += 1
		with open(HOME+'/data/corpora/'+corpus+'.sentences.'+embsl+'.pkl', 'wb') as handle:
			pickle.dump(bert, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BERTMapper(FeatureMapper): # ONLY WHEN PRECOMPUTING BERT TENSORS
	def get_BERTrepr(self, graph):
		sent = graph.sent.split(" ")
		sent = [s.lower() for s in sent]
		wembs, semb, ael, attns = self.vsm.compute(sent)
		sentkey = "_".join(sent)
		return sentkey, sent, wembs, semb, ael, attns



# Dummy mapper for cases where no features are needed, e.g. for majority baselines
class DummyMapper(FeatureMapper): 
	def get_repr(self, graph):
		return np.zeros(self.vsm.dim)


def avg_embedding(wordlist, emb, sent, ndxs):
	res = []
	assert len(wordlist) == len(ndxs)
	for word, ndx in zip(wordlist,ndxs):
		word = word.lower()
		res += [emb.get(word, ndx, sent)]
	return np.mean(res, axis=0)


#--------------------------------------------------------------------
def passAll_embedding(wordlist, emb, sent, ndxs):
	if (ndxs[0] == -1):
		ndxs[0] = 0
	v0 = np.zeros(768, dtype=np.float32)
	res = [v0] * (max(ndxs)+1)
	assert len(wordlist) == len(ndxs)
	for word, ndx in zip(wordlist,ndxs):
		word = word.lower()
		res[ndx] = emb.get(word, ndx, sent)
	return res


class ElectraAGEMapper(FeatureMapper):
	def get_repr(self, graph, lexicon):
		words = graph.sent.split(" ")
		ndxs = list(range(len(words)))
		lemmapos = graph.get_predicate_head()["lemmapos"]

		x = passAll_embedding(words, self.vsm, words, ndxs)
		x = np.asarray(x)
		x = torch.from_numpy(np.pad(x,((0,512-x.shape[0]),(0,0))))

		# DYNAMIC WINDOW COMPUTATION BASED ON DEPENDENCIES
		vrb = graph.find_parent_verb(graph.predicate_head)
		if (vrb != -1):
			deps = list(graph.get_direct_dependents(vrb))
			if (deps == []):
				WIN = [-10, 10]
			else:
				deps = np.array(deps)-1
				md = min(deps)
				Md = max(deps)
				gph = graph.predicate_head - 1
				if (gph < md):
					md = gph
				if (gph > Md):
					Md = gph
				WIN = [md - gph, Md - gph] 
		else:
			WIN = [-1000, 1000] # ALL SENTENCE!
		WIN = [-1000, 1000] # ALL SENTENCE!

		# M
		predicate_head = graph.get_predicate_head()
		hd_w = [predicate_head["word"].lower(), ]
		hd_w_ndx = [graph.predicate_head - 1]
		tgt_w = graph.get_predicate_node_words()
		tgt_w_ndx = [gpn-1 for gpn in graph.predicate_nodes]
		if (self.multiword_averaging):
			# MULTI-WORD AVERAGE 
			m = torch.from_numpy(np.array([min(tgt_w_ndx), max(tgt_w_ndx), len(ndxs)],dtype=np.int))
		else:
			# NO MWA, ONLY HEAD
			m = torch.from_numpy(np.array([hd_w_ndx[0], hd_w_ndx[0], len(ndxs)],dtype=np.int))

		t = torch.sum(x[m[0]:m[1]+1,:], dim=0)

		frameId = self.lexicon.get_id(graph.get_predicate_head()["frame"])
		fesE = np.zeros((self.lexicon.get_number_of_FEs(),), dtype=np.float32)
		atleast1 = False
		for n in graph.G.nodes:
			role = graph.G.node[n].get("role", "_")
			if (role != "_"):
				fesE[self.lexicon.get_FEid(role)] = 1.0
				atleast1 = True
		if (not atleast1):
			fesE[self.lexicon.get_FEid('NONE')] = 1.0

		return t, frameId, fesE
