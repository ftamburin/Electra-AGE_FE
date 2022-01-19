import sys
import numpy as np
import codecs
from vect_features import Bert
import pickle
import torch

# Extra classes for managing external resources


class Lexicon:  # Lexicon manager. Stores information about lemma.pos -> frame mappings
	def __init__(self):
		self.frameLexicon = {}
		self.FELexicon = {}
		self.frameToId = {}
		self.idToFrame = {}
		self.FEToId = {}
		self.idToFE = {}
		self.frameToFE = {}
		self.source = "NA"

	def get_number_of_frames(self):
		return len(self.frameToId)

	def get_number_of_FEs(self):
		return len(self.FEToId)

	def get_id(self, frame):
		if frame not in self.frameToId:
			print("Unknown frame", frame, "assigning id=-1")
		return self.frameToId.get(frame, -1)

	def get_FEid(self, fe):
		if fe not in self.FEToId:
			print("Unknown FE", fe, "assigning id=-1")
		return self.FEToId.get(fe, -1)

	def get_available_frame_ids(self, lemmapos):
		return [self.frameToId[x] for x in self.frameLexicon.get(lemmapos, [])]

	def get_all_frame_ids(self):
		return list(self.idToFrame.keys())

	def get_frame(self, id):
		return self.idToFrame.get(id, "UNKNOWN_FRAME")

	# Load from pre-defined lexicon in format [frame \t lemmapos]
	def load_from_list(self, src, allFN):
		# LOAD FRAMES
		with codecs.open(src, "r", "utf-8") as f:
			frames = []
			for line in f:
				frame, lemmapos = line.strip().rstrip().split("\t")
				self.frameLexicon[lemmapos] = self.frameLexicon.get(lemmapos, []) + [frame]
				if (frame != 'Test35'):
					frames += [frame]
		with codecs.open(allFN, "r", "utf-8") as f:
			for line in f:
				line = line.split()
				if (line[0] not in frames):
					frames += [line[0]]
		frames += ['Root']
		frames = list(set(frames))
		self.frameToId = {frames[i]:i for i in range(len(frames))}
		self.idToFrame = {y:x for (x,y) in self.frameToId.items()}
		# LOAD FEs
		FEs = []
		with codecs.open(allFN.replace("Frames","FEs"), "r", "utf-8") as f:
			for line in f:
				line = line.split()
				if (line[0].upper() not in FEs):
					FEs += [line[0].upper()]
		FEs += ['NONE']
		FEs = list(set(FEs))
		self.FEToId = {FEs[i]:i for i in range(len(FEs))}
		self.idToFE = {y:x for (x,y) in self.FEToId.items()}
		self.source = src.split("/")[-1]
		# LOAD FRAME TO FEs
		fr2FE = pickle.load(open(allFN.replace("_AllFrames",".Frame-FE.pkl"), "rb"))
		for k,v in fr2FE.items():
			self.frameToFE[self.frameToId[k]] = []
			for fe in v:
				self.frameToFE[self.frameToId[k]].append(self.FEToId[fe.upper()])
		self.frameToFE[self.frameToId['Root']] = []


	def is_unknown(self, lemmapos):
		return lemmapos not in self.frameLexicon

	def is_ambiguous(self, lemmapos):
		return len(self.frameLexicon.get(lemmapos, []))>1

	# Load from training data
	def load_from_graphs(self, g_train):
		frames = []
		for g in g_train:
			predicate = g.get_predicate_head()
			lemmapos = predicate["lemmapos"]
			frame = predicate["frame"]
			self.frameLexicon[lemmapos] = self.frameLexicon.get(lemmapos, []) + [frame]
			frames += [frame]
		frames = list(set(frames))
		self.frameToId = {frames[i]: i for i in range(len(frames))}
		self.idToFrame = {y: x for (x, y) in self.frameToId.items()}
		self.source = "training_data"


class VSM:
	def __init__(self, src):
		self.map = {}
		self.dim = None
		self.source = src.split("/")[-1] if src is not None else "NA"
		# create dictionary for mapping from word to its embedding
		if src is not None:
			with open(src) as f:
				i = 0
				for line in f:
					word = line.split()[0]
					embedding = line.split()[1:]
					self.map[word] = np.array(embedding, dtype=np.float32)
					i += 1
				self.dim = len(embedding)
		else:
			self.dim = 1

	def get(self, word):
		word = word.lower()
		if word in self.map:
			return self.map[word]
		else:
			return np.zeros(self.dim, dtype=np.float32)


class BertVSM:
	def __init__(self, src, mode):
		if (mode == 'create'):
			print('Creating CONTEXTUAL VECTOR model with',src)
			self.map = {}
			self.dim = -1

			bert_batch_size=1
			bert_layers='-1,-2,-3,-4'
			#bert_layers='-1'
			bert_load_features=False
			bert_max_seq_length=512
			bert_multilingual_cased=False
			bert_model=src
			print('Loading CONTEXTUAL VECTOR model...')
			sys.stdout.flush()
			self.bert = Bert(bert_model,bert_layers, bert_max_seq_length, bert_batch_size, bert_multilingual_cased, 0)
			print('...done.')
			sys.stdout.flush()
		else:
			print('Loading precomputed CONTEXTUAL VECTOR model from',src)
			with open(src, 'rb') as handle:
				self.bert = pickle.load(handle)
			self.vdim = self.bert[next(iter(self.bert))][1][0].shape[0]

	def compute(self, sent):
		sent = [s.lower() for s in sent]
		wembs, semb, ael, attns = self.bert.extract_bert_features([sent])
		return wembs[0], semb[0], ael[0], attns[0]

	def get(self, word, ndx, sent):
		if (word != 'root'):
			sent = [s.lower() for s in sent]
			sent = "_".join(sent)

			# RETURN SIMPLE WORD EMBEDDING [w][vdim]
			(_,wembs) = self.bert[sent]
			return wembs[ndx].numpy()

			# RETURN ALL ENCODER LAYERS [w][l,vdim]
			#(_,_,_,ael,_) = self.bert[sent]
			#return ael[ndx].numpy()

			# RETURN WORD EMBEDDING [w][vdim] + WORD ATTENTIONS [w][l,h,512]
			#(_,wembs,_,_,attns) = self.bert[sent]
			#return np.concatenate((wembs[ndx].numpy(), torch.flatten(attns[ndx]).numpy()))		   #1
			#return np.concatenate((wembs[ndx].numpy(), torch.flatten(attns[ndx][-1,:,:]).numpy())) #2
		else:
			# RETURN SIMPLE WORD EMBEDDING (xp_201)
			return np.zeros((self.vdim),dtype=np.float32)

			# RETURN ALL ENCODER LAYERS
			#return np.zeros((12,vdim),dtype=np.float32)

			# RETURN WORD EMBEDDING + WORD ATTENTIONS
			#return np.zeros((vdim+12*12*512),dtype=np.float32)	#1
			#return np.zeros((vdim+12*512),dtype=np.float32)		#2
