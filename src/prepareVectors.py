from __future__ import print_function
import sys
import os

from data import get_graphs
from extras import Lexicon, VSM, BertVSM
from representation import BERTMapper, DummyMapper
from classifier import AdvDNNClassifier
from evaluation import Score
from reporting import ReportManager
from config import Config
from resources import ResourceManager
import time

import numpy as np
#import tensorflow as tf
import random as rn
import torch



def prepare(HOME, EMBEDDINGS_NAME):
	corpus_train = 'train'
	corpus_dev   = 'dev'
	corpus_test  = 'test'

	print("Building vectors using",EMBEDDINGS_NAME)
	if ('electra' in EMBEDDINGS_NAME):
		embsl = 'electra'
	else:
		embsl = 'bert'

	print("Starting resource manager")
	sources = ResourceManager(HOME)

	lexicon = None

	vsm = BertVSM(EMBEDDINGS_NAME,'create')
	mapper = BERTMapper(HOME, vsm, lexicon)

	print('Vectorizing',corpus_dev)
	g_test = get_graphs(*sources.get_corpus(corpus_dev))
	mapper.save_BERT(g_test, HOME, corpus_dev, embsl)

	print('Vectorizing',corpus_train)
	g_train = get_graphs(*sources.get_corpus(corpus_train))
	mapper.save_BERT(g_train, HOME, corpus_train, embsl)

	print('Vectorizing',corpus_test)
	g_test = get_graphs(*sources.get_corpus(corpus_test))
	mapper.save_BERT(g_test, HOME, corpus_test, embsl)



if __name__ == "__main__":
	print('PyTorch Version:',torch.__version__)
	HOME = sys.argv[1]
	EMBEDDINGS_NAME = sys.argv[2]
	prepare(HOME, EMBEDDINGS_NAME)
