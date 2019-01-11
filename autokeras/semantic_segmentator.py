# ----------------------------------
# The idea of the classes and functions in this file is largely borrowed from
# https://github.com/amdegroot/ssd.pytorch
# A huge thank you to the authors: Max deGroot and Ellis Brown
# ----------------------------------
from __future__ import print_function
from autokeras.semantic_segmentation.dataset import TestDataset
from autokeras.semantic_segmentation.models import ModelBuilder, SegmentationModule
from autokeras.semantic_segmentation.utils import colorEncode
from autokeras.semantic_segmentation.lib.nn import user_scattered_collate, async_copy_to
from autokeras.semantic_segmentation.lib.utils import as_numpy, mark_volatile
import autokeras.semantic_segmentation.lib.utils.data as torchdata
from autokeras.utils import download_file

from scipy.io import loadmat

import os
import datetime
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt

class SemanticSegmentator():

	def __init__(self, pretrain=, cuda=True):

		self.cuda = cuda
		self.arch_encoder = 'resnet50_dilated8'
		self.arch_decoder = 'ppm_bilinear_deepsup'
		self.fc_dim = 2048
		self.num_class = 150
		self.base_dir = 'semantic_segmentation'
		self.test_img = os.path.join(self.base_dir, 'data/ADE_val_00001519.jpg')
		self.model_path = os.path.join(self.base_dir, 'baseline-resnet50_dilated8-ppm_bilinear_deepsup')
		self.suffix = '_epoch_20.pth'
		self.num_val = -1
		self.batch_size = 1
		self.gpu_id = 0

		self.imgSize = [300, 400, 500, 600]
		self.imgMaxSize = 1000
		self.padding_constant = 8
		self.segm_downsampling_rate = 8
		self.result = os.path.join(self.base_dir, 'result')

		if not os.path.isdir(self.result):
			os.makedirs(self.result)

		if torch.cuda.is_available():
			if self.cuda:
				torch.set_default_tensor_type('torch.cuda.FloatTensor')
			if not self.cuda:
				print("WARNING: It looks like you have a CUDA device, but aren't " +
					  "using CUDA.\nRun with --cuda for optimal training speed.")
				torch.set_default_tensor_type('torch.FloatTensor')
		else:
			torch.set_default_tensor_type('torch.FloatTensor')

	def load(self):

		self.weight_path = os.path.join(self.base_dir, 'baseline-resnet50_dilated8-ppm_bilinear_deepsup')
		if not os.path.isdir(self.weight_path):
			os.makedirs(self.weight_path)

		self.weights_encoder = os.path.join(self.weight_path, 'encoder_epoch_20.pth')
		self.weights_decoder = os.path.join(self.weight_path, 'decoder_epoch_20.pth')

		if not os.path.isfile(self.weights_encoder):
			download_file('http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet50_dilated8-ppm_bilinear_deepsup/encoder_epoch_20.pth', self.weights_encoder)
		if not os.path.isfile(self.weights_decoder):
			download_file('http://sceneparsing.csail.mit.edu/model/pytorch/baseline-resnet50_dilated8-ppm_bilinear_deepsup/decoder_epoch_20.pth', self.weights_decoder)

		# self.weights_encoder = os.path.join(self.model_path, 'encoder' + self.suffix)
		# self.weights_decoder = os.path.join(self.model_path, 'decoder' + self.suffix)

		# Network Builders
		builder = ModelBuilder()
		net_encoder = builder.build_encoder(arch=self.arch_encoder,
											fc_dim=self.fc_dim,
											weights=self.weights_encoder)

		net_decoder = builder.build_decoder(arch=self.arch_decoder,
											fc_dim=self.fc_dim,
											num_class=self.num_class,
											weights=self.weights_decoder,
											use_softmax=True)

		crit = nn.NLLLoss(ignore_index=-1)

		self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

		self.segmentation_module.cuda()

	def visualize_result(self, preds, output_path):

		colors = loadmat(os.path.join(self.base_dir, 'data/color150.mat'))['colors']
		pred_color = colorEncode(preds, colors).astype(np.uint8)
		cv2.imwrite(output_path, pred_color)

	def predict(self, input_path=os.path.join(os.getcwd(), 'semantic_segmentation/data/ADE_val_00001519.jpg'), output_path=None):

		self.segmentation_module.eval()
		# Dataset and Loader
		list_test = [{'fpath_img': input_path}]
		dataset_val = TestDataset(list_test, self, max_sample=self.num_val)

		loader = torchdata.DataLoader(dataset_val,
									  batch_size=self.batch_size,
									  shuffle=False,
									  collate_fn=user_scattered_collate,
									  num_workers=5,
									  drop_last=True)

		for i, batch_data in enumerate(loader):
			# process data
			batch_data = batch_data[0]
			segSize = (batch_data['img_ori'].shape[0],
					   batch_data['img_ori'].shape[1])
			img_resized_list = batch_data['img_data']

			with torch.no_grad():
				pred = torch.zeros(1, self.num_class, segSize[0], segSize[1])

				for img in img_resized_list:
					feed_dict = batch_data.copy()
					feed_dict['img_data'] = img
					del feed_dict['img_ori']
					del feed_dict['info']
					feed_dict = async_copy_to(feed_dict, self.gpu_id)

					pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
					pred = pred + pred_tmp / len(self.imgSize)

				_, preds = torch.max(pred, dim=1)
				preds = as_numpy(preds.squeeze(0))

			if output_path:
				self.visualize_result(preds, output_path)
			else:
				return preds

if __name__ == "__main__":

	clf = SemanticSegmentator()
	clf.load()
	clf.predict(output_path='./semantic_segmentation/result/ADE_val_00001519.png')