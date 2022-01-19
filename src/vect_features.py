import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig

from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from torch.nn import functional as F

class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, tokens, input_ids, input_mask, map_to_original_tokens):
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.map_to_original_tokens = map_to_original_tokens

class Bert(object):
	"""A facade to Bert model that extracts features for sets of tokens"""

	def __init__(self, pretrained_model, layer_indexes, max_seq_length, batch_size, multi_lingual=False, which_cuda = 0):
		print('Init transformers with',pretrained_model)
		self.max_seq_length = max_seq_length
		self.batch_size = batch_size
		self.pretrained_model = pretrained_model
		self.layer_indexes = layer_indexes = [int(x) for x in layer_indexes.split(",")]
		config = AutoConfig.from_pretrained(pretrained_model, output_hidden_states=True, output_attentions=True)
		#self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=not multi_lingual) # lower case for english, keep case for multi lingual
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case=not multi_lingual) # lower case for english, keep case for multi lingual
		
		self.device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else 'cpu')
		#self.model = AutoModel.from_pretrained(pretrained_model, config=config).to(self.device)
		self.model = AutoModel.from_pretrained(pretrained_model, config=config).to(self.device)
		
		# tells pytorch to run in evaluation mode instead of training
		self.model.eval()
		
	def get_bert_features(self, sentence):
		## sentence is in the format ['tok1', 'tok2']
		bert_tokens, map_to_original_tokens = self.convert_to_bert_tokenization(sentence)
		feature = self.from_bert_tokens_to_features(bert_tokens, map_to_original_tokens)
		features = [feature]
		
		# get ids
		all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
		# mask with 0's for placeholders
		all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
		# tensor with 1...n where n is the number of examples
		all_encoder_layers, _ = self.model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
		last_layer = all_encoder_layers[-1]
		
		return bert_tokens, map_to_original_tokens, last_layer

	def extract_bert_features(self, sentences):
		# data loading
		features = []
		for sentence in sentences:
			bert_tokens, map_to_original_tokens = self.convert_to_bert_tokenization(sentence)
			feature = self.from_bert_tokens_to_features(bert_tokens, map_to_original_tokens)
			features.append(feature)
		
		all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
		# mask with 0's for placeholders
		all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
		# tensor with 1...n where n is the number of examples
		all_token_maps = torch.tensor([f.map_to_original_tokens for f in features], dtype=torch.long)
		# indexes that map back dataset
		all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
		
		# create a dataset the resources needed
		eval_data = TensorDataset(all_input_ids, all_input_mask, all_token_maps, all_example_index)
		# create a sampler which will be used to create the batches
		eval_sampler = SequentialSampler(eval_data)
		eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)
			
		ssbert = []
		ssael = []
		ssattn = []
		for input_ids, input_mask, token_maps, example_indices in eval_dataloader:
			input_ids = input_ids.to(self.device)
			input_mask = input_mask.to(self.device)
			### RUN MODEL ###
			# run model to get all 12 layers of bert
			all_encoder_layers = self.model(input_ids, token_type_ids=None, attention_mask=input_mask)
			#cls = all_encoder_layers[1].clone().detach().cpu()
			#attns = all_encoder_layers[3]
			if (self.pretrained_model.find('bert') != -1):
				all_encoder_layers = all_encoder_layers[2][1:]	# BERT
			else:
				all_encoder_layers = all_encoder_layers[1][1:]	# ELECTRA

			averaged_output = torch.stack([all_encoder_layers[idx] for idx in self.layer_indexes]).mean(0) / len(self.layer_indexes)

			sbert = []
			for i, idx in enumerate(example_indices):
				for j in range(len(sentences[idx])):
					if token_maps[i,j] < 511:
						sbert.append(averaged_output[i,token_maps[i,j]].clone().detach().cpu())
					else:
						sbert.append(averaged_output[i,token_maps[i,511]].clone().detach().cpu())
			ssbert.append(sbert)

			'''
			# RESTRUCTURING ENCODER LAYERS
			all_encoder_layers = torch.stack(all_encoder_layers)
			sael = []
			for i, idx in enumerate(example_indices):
				for j in range(len(sentences[idx])):
					if token_maps[i,j] < 511:
						jj = j
					else:
						jj = 511
					xx = all_encoder_layers[:,i,token_maps[i,jj],:]
					sael.append(xx.clone().detach().cpu())
			ssael.append(sael)	# #word list of [12,768] tensors

			# RESTRUCTURING ATTNS
			attns = torch.stack(attns)
			sattn = []
			for i, idx in enumerate(example_indices):
				for j in range(len(sentences[idx])):
					if token_maps[i,j] < 511:
						jj = j
					else:
						jj = 511
					xx = attns[:,i,:,token_maps[i,jj]]
					padS = 512-xx.shape[2]
					xx = F.pad(xx, (0,padS), "constant", 0)
					sattn.append(xx.clone().detach().cpu())
			ssattn.append(sattn)	# #word list of [12,12,512] tensors
			'''
		#return ssbert, cls, ssael, ssattn
		return ssbert, [[]], [[]], [[]]

	
	def from_bert_tokens_to_features(self, tokens, map_to_original_tokens):
		# from word to id
		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		# mask
		input_mask = [1] * len(input_ids)
		# add padding
		while len(input_ids) < self.max_seq_length:
			input_ids.append(0)
			input_mask.append(0)

		while len(map_to_original_tokens) < self.max_seq_length:
			map_to_original_tokens.append(0)

		return InputFeatures(
				tokens=tokens,
				input_ids=input_ids,
				input_mask=input_mask,
				map_to_original_tokens = map_to_original_tokens
		)
	
	# modified from https://github.com/google-research/bert#tokenization
	def convert_to_bert_tokenization(self, orig_tokens):
		### Output
		bert_tokens = []

		# Token map will be an int -> int mapping between the `orig_tokens` index and
		# the `bert_tokens` index.
		orig_to_tok_map = []

		bert_tokens.append("[CLS]")
		for orig_token in orig_tokens:
		  orig_to_tok_map.append(len(bert_tokens))
		  bert_tokens.extend(self.tokenizer.tokenize(orig_token))

		# truncate
		# account for [SEP] with "- 1"
		if len(bert_tokens) > self.max_seq_length - 1:
			bert_tokens = bert_tokens[0:(self.max_seq_length - 1)]	

		# Add [SEP] after truncate
		bert_tokens.append("[SEP]")

		return bert_tokens, orig_to_tok_map

# convert the following format:
# [[torch(data), torch(data), torch(data)],[torch(data), torch(data)]]
# to torch([[data,data,data], [data,data,placeholder]])
def from_tensor_list_to_one_tensor(tensor_list, bert_hidden_size):
	# calculate max length
	longest_length = max([len(t) for t in tensor_list])
	# pad tensors smaller than the max length
	padded_tensor_list = [t +(longest_length - len(t)) * [torch.zeros(bert_hidden_size)] for t in tensor_list]
	# first stack all tensors in the first dimentions creating n stacked tensors
	stacked_tensor_list = [torch.stack(sentence_tensors) for sentence_tensors in padded_tensor_list]
	# finally stack the n stacked tensors into a single tensor
	one_tensor = torch.stack(stacked_tensor_list)
	return one_tensor


