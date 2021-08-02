from ..dataset import DataSet
from dataset.babi.data_preprocess.preprocess import parse
from torch.utils.data import TensorDataset, DataLoader
import json
import torch
import numpy as np
		
class BABI():
	"""Pre-processing and loading of the babi dataset"""
	def __init__(self, args):
		# DataSet().__init__(self, args.path)
		self.args = args
		self.path = args.path
		
		with open(args.data_config_file, "r") as fp:
			self.config = json.load(fp)
		print(self.config)
		self.max_seq = 0 # load max sequence from the args parameter
			# Load data
		if self.config["task-ids"]=="all":
			self.task_ids = range(1,21)
		else:
			self.task_ids = [self.config["task-ids"]]

		# train_raw_data, valid_raw_data, test_raw_data, word2id = parse_all(data_config["data_path"],list(range(1,21)))
		self.word2id = None
		self.train_data_loaders = {}
		self.valid_data_loaders = {}
		self.test_data_loaders = {}

		self.num_train_batches = self.num_valid_batches = self.num_test_batches = 0
		self.train_batch_size = self.config["train_batch_size"]
		self.valid_batch_size = self.config["valid_batch_size"]
		self.test_batch_size = self.config["test_batch_size"]

		for i in self.task_ids:
			train_raw_data, valid_raw_data, test_raw_data, self.word2id = parse(self.path, str(i), word2id=self.word2id, use_cache=True, cache_dir_ext="")
			train_epoch_size = train_raw_data[0].shape[0]
			valid_epoch_size = valid_raw_data[0].shape[0]
			test_epoch_size = test_raw_data[0].shape[0]

			max_story_length = np.max(train_raw_data[1])
			max_sentences = train_raw_data[0].shape[1]
			self.max_seq = max(self.max_seq, train_raw_data[0].shape[2])
			max_q = train_raw_data[0].shape[1]
			valid_batch_size = valid_epoch_size // 73  # like in the original implementation
			test_batch_size = test_epoch_size // 73

			train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
			valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])
			test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])

			train_data_loader = DataLoader(train_dataset, batch_size=self.config["train_batch_size"], shuffle=True)
			valid_data_loader = DataLoader(valid_dataset, batch_size=self.config["valid_batch_size"])
			test_data_loader = DataLoader(test_dataset, batch_size=self.config["test_batch_size"])

			self.train_data_loaders[i] = [iter(train_data_loader), train_data_loader]
			self.valid_data_loaders[i] = valid_data_loader
			self.test_data_loaders[i] = test_data_loader

			self.num_train_batches += len(train_data_loader)
			self.num_valid_batches += len(valid_data_loader)
			self.num_test_batches += len(test_data_loader)

		print(f"total train data: {self.num_train_batches*self.config['train_batch_size']}")
		print(f"total valid data: {self.num_valid_batches*self.config['valid_batch_size']}")
		print(f"total test data: {self.num_test_batches*self.config['test_batch_size']}")
		print(f"voca size {len(self.word2id)}")

	def vocabSize(self):
		return len(self.word2id)

	def getTrainData(self):
		return self.train_data_loaders

	def getValidData(self):
		return self.valid_data_loaders

	def getTestData(self):
		return self.test_data_loaders




			