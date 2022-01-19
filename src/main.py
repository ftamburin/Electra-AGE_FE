from __future__ import print_function
import sys
import os

#from globals import *
from data import get_graphs
from extras import Lexicon, VSM, BertVSM
from representation import ElectraAGEMapper, DummyMapper
from classifier import AdvDNNClassifier
from evaluation import Score
from reporting import ReportManager
from config import Config
from resources import ResourceManager
import time

import numpy as np
import scipy.sparse
import random as rn
import torch
import pickle
from sklearn.metrics import jaccard_score


def set_seed(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	rn.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	#tf.random.set_seed(seed)

def ExportFrame2Id(frameToId):
	pickle.dump(frameToId, open('frameToId', "wb"))



def process_all(MODE, HOME, EMBEDDINGS_NAME, DATASET):
	#set_seed(4)  # fix the random seed
	corpus_train = 'train'
	corpus_dev   = 'dev'
	corpus_test  = 'test'

	LEXICON_FULL = DATASET+"_lexicon"
	LEXICON_ONLY_TRAIN = "fnTrain_lexicon"
	LEXICON_ALLFN = DATASET+"_AllFrames"
	#LEXICON_TAXON = DATASET+".taxonomy"
	print(LEXICON_FULL,LEXICON_ALLFN)

	#vsms = [EMBEDDINGS_NAME]
	if ('electra' in EMBEDDINGS_NAME):
		embSfx = '.sentences.electra.pkl'
		vsms = ['google/electra-base-discriminator']
	else:
		embSfx = '.sentences.bert.pkl'
		vsms = ['bert-base-uncased']
	print('USING',vsms)

	#lexicons = [LEXICON_ONLY_TRAIN,LEXICON_FULL,'joined_lex']  # lexicon to use (!all_unknown setting!)
	lexicons = ['joined_lex']  # lexicon to use (mind the all_unknown setting!)
	multiword_averaging = [True]  # treatment of multiword predicates, false - use head embedding, true - use avg
	all_unknown = [False]  # makes the lexicon treat all LU as unknown, corresponds to the no-lex setting
	repeats = 10

	# CREATE ALL CONFURATIONS
	configs = []

	# ADD CONFIGURATIONS FOR ADVANCED NN CLASSIFIERS
	for lexicon in lexicons:
		for vsm in vsms:
			for mwa in multiword_averaging:
				for all_unk in all_unknown:
					for j in range(repeats):
						configs += [Config(AdvDNNClassifier, ElectraAGEMapper, lexicon, vsm, mwa, all_unk, None, None, None)]


	print("Starting resource manager")
	sources = ResourceManager(HOME)

	print("Initializing reporters")
	reports = ReportManager(sources.out)

	print("Running the experiments!")
	runs = len(configs)
	print (runs, "configurations runs")

	current_config = 0

	g_train = get_graphs(*sources.get_corpus(corpus_train))
	reports.conll_reporter_train.report(g_train)

	for conf in configs:
		current_config += 1
		start_time = time.time()

		model_file = HOME+'/model'+str(current_config)+'.bin'
		print('Model file:',model_file)

		lexicon = Lexicon()
		# go to configuration, check which lexicon is needed, locate the lexicon in FS, load the lexicon
		if (MODE == 'train'):
			lexicon.load_from_list(sources.get_lexicon(conf.get_lexicon()),sources.get_lexicon(LEXICON_ALLFN))
		else:
			checkpoint = torch.load(model_file)
			lexicon = checkpoint['lexicon']
		reports.lexicon_reporter.report(lexicon)
		print(lexicon.frameToId) 
		#ExportFrame2Id(lexicon.frameToId)
		print(lexicon.get_number_of_frames())
		print(lexicon.FEToId) 
		print(lexicon.get_number_of_FEs())
		#print(lexicon.frameToFE)


		# PREPARE TRAINING SET
		print('###TRAIN###')
		vsm = BertVSM(HOME+'/data/corpora/'+corpus_train+embSfx,'get')
		mapper = conf.get_feat_extractor()(HOME, vsm, lexicon, mwa=conf.get_multiword_averaging())
		T_train, y_train, yFE_train, lemmapos_train, gid_train = mapper.get_matrix(g_train)
		print('T_train',len(T_train))

		# PREPARE DEV SET
		print('###DEV###')
		g_dev = get_graphs(*sources.get_corpus(corpus_dev))
		vsm = BertVSM(HOME+'/data/corpora/'+corpus_dev+embSfx,'get')
		mapper = conf.get_feat_extractor()(HOME, vsm, lexicon, mwa=conf.get_multiword_averaging())
		T_dev, y_dev, yFE_dev, lemmapos_dev, gid_dev = mapper.get_matrix(g_dev)

		# PREPARE TEST SET
		print('###TEST###')
		g_test = get_graphs(*sources.get_corpus(corpus_test))
		vsm = BertVSM(HOME+'/data/corpora/'+corpus_test+embSfx,'get')
		mapper = conf.get_feat_extractor()(HOME, vsm, lexicon, mwa=conf.get_multiword_averaging())
		T_test, y_test, yFE_test, lemmapos_test, gid_test = mapper.get_matrix(g_test)

		print(T_train.shape,T_train[0].shape)

		# CREATE THE MODEL
		clf = conf.get_clf()(lexicon, conf.get_all_unknown(), conf.get_num_components(), 
						 	conf.get_max_sampled(), conf.get_num_epochs(), 
						 	T_train[0].shape,
						 	lexicon.get_number_of_frames(), model_file, 
						 	os.path.split(sources.get_lexicon(LEXICON_ALLFN))[0], DATASET)

		if (MODE == 'train'):
			# TRAIN THE MODEL
			clf.train(T_train, y_train, yFE_train, lemmapos_train, 
						T_dev, y_dev, yFE_dev, lemmapos_dev,
						T_test, y_test, yFE_test, lemmapos_test)

		# EVALUATE ON TEST
		score = Score()  # storage for scores
		score_v = Score()  # storage for verb-only scores
		score_known = Score()  # storage for known lemma-only scores
		start_time = time.time()
		reports.set_config(conf, corpus_train, corpus_test)
		reports.conll_reporter_test.report(g_test)

		# predict and compare
		print('Loading model',model_file)
		checkpoint = torch.load(model_file)
		clf.model.load_state_dict(checkpoint['model'])
		inx = checkpoint['inx']
		js_num = js_den = 0
		for t, y_true, fe_true, lemmapos, gid, g in zip(T_test, y_test, yFE_test, lemmapos_test, gid_test, g_test):
			y_predicted, fe_predicted, fe_indexes = clf.predict(t, lemmapos, inx)
			fe_predicted = fe_predicted.detach().cpu().numpy()
			fe_predicted = fe_predicted[fe_indexes]
			fe_true = fe_true[fe_indexes].astype(int)
			fe_pl = [idx for idx, val in enumerate(fe_predicted) if val != 0]
			fe_tl = [idx for idx, val in enumerate(fe_true) if val != 0]
			js_num += jaccard_score(fe_true, fe_predicted)
			js_den += 1
			y_true = y_true.item()
			correct = y_true == y_predicted

			score.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)
			if lemmapos.endswith(".v"):
				score_v.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)
			if not lexicon.is_unknown(lemmapos):
				score_known.consume(correct, lexicon.is_ambiguous(lemmapos), lexicon.is_unknown(lemmapos), y_true)

			reports.result_reporter.report(gid, g, lemmapos, y_predicted, y_true, lexicon)
		#print('HL -----------------------')

		fe_js = js_num / js_den
		reports.summary_reporter.report(corpus_train, corpus_test, conf, score, fe_js, time.time() - start_time)
		reports.summary_reporter_v.report(corpus_train, corpus_test, conf, score_v, -1, time.time() - start_time)
		reports.summary_reporter_known.report(corpus_train, corpus_test, conf, score_known, -1, time.time() - start_time)

		print ("============ STATUS: - conf", current_config, "/", len(configs))


if __name__ == "__main__":
	print('PyTorch Version:',torch.__version__)
	MODE = sys.argv[1]
	HOME = sys.argv[2]
	EMBEDDINGS_NAME = sys.argv[3]
	DATASET = sys.argv[4]
	process_all(MODE, HOME, EMBEDDINGS_NAME, DATASET)
