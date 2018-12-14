import sys
sys.path.insert(0, r'C:\Users\Brooks\github\splitr')# access library code from outside /models

# library functions:
import torch
import time
import pandas as pd

# Splitr modules:
import model_utils

# this is really a constructor for a bidirectional LSTM but i figured
# BD_LSTM  was only 2 letters off of BDSM so why not
class BDSM(torch.nn.Module):
	def __init__(self, num_inputs, num_hidden_layers,layer_count=1):
		super(BDSM, self).__init__()

		self.rnn = torch.nn.LSTM(num_inputs, num_hidden_layers, bidirectional=True, batch_first=True, num_layers=layer_count)

	def forward(self, x):
		rnn_output, _ = self.rnn(x)
		return rnn_output

# Convolution cell with adjustable activation / maxpool size / batchnorm
class CNN_cell(torch.nn.Module):
	def __init__(self,in_channels=False,out_channels=False,kernel_size=False,activation=False, pool_shape=False, pool_stride=False, batchnorm=False):
		super(CNN_cell, self).__init__()

		_layers = []

		if in_channels and out_channels:
			_layers.append(torch.nn.Conv2d(in_channels, out_channels,kernel_size))
		if activation:
			_layers.append(self.find_activation(activation))
		if batchnorm:
			_layers.append(torch.nn.BatchNorm2d(batchnorm))
		if pool_shape and pool_stride:
			_layers.append(torch.nn.MaxPool2d(pool_shape, pool_stride))

		self.cnn = torch.nn.Sequential(*_layers)

	def find_activation(self, activation):
		if activation == 'relu':
			return torch.nn.ReLU()
		elif activation == 'tanh':
			return torch.nn.Tanh()
		elif activation == 'leaky':
			return torch.nn.LeakyReLU()
		else:
			print('activation function call |%s| is not configured' % activation )
	def forward(self, input_tensor):
		output = self.cnn(input_tensor)
		return output

# https://arxiv.org/pdf/1507.05717.pdf
class model(torch.nn.Module):
	def __init__(self, channel_count=1,num_hidden= 256, unique_char_count=57,rnn_layer_stack=1):
		super(model, self).__init__()

		# add batch norm later
		# add dropout to cnn layers
		self.softmax = torch.nn.LogSoftmax(dim=2)

		# CONVOLUTIONS
		_cnn_layer = []
		_cnn_layer.append(CNN_cell(in_channels=channel_count, out_channels=64, kernel_size=3, activation='relu', pool_shape=(2,2), pool_stride=2))

		_cnn_layer.append(CNN_cell(in_channels=64, out_channels=128, kernel_size=3,activation='relu', pool_shape=(2,2), pool_stride=2))

		_cnn_layer.append(CNN_cell(in_channels=128, out_channels=256, kernel_size=3, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=256, out_channels=512, kernel_size=3, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=3, activation='relu', pool_shape=(1,2), pool_stride=2))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu', batchnorm=512))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2))

		# RNN LAYERS
		_bdsm_layer = []
		_bdsm_layer.append(BDSM(num_inputs=2048, num_hidden_layers=num_hidden, layer_count=rnn_layer_stack))
		_bdsm_layer.append(BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden, layer_count=1))

		# CHAR activations (transcription)
		self.linear = torch.nn.Sequential(
		torch.nn.Linear(in_features=num_hidden*2, out_features=unique_char_count),torch.nn.ReLU())

		self.cnn = torch.nn.Sequential(*_cnn_layer)
		self.rnn = torch.nn.Sequential(*_bdsm_layer)

	def forward(self, x):
		t = self.cnn(x)
		batch, depth, base, height = t.shape

		cnn_output = t.view(batch, height, depth*base)

		rnn_output = self.rnn(cnn_output)
		batch, char_len, depth = rnn_output.shape
		rnn_output = rnn_output.contiguous().view(batch*char_len, depth)

		output = self.linear(rnn_output).view(batch, char_len, -1)
		output = self.softmax(output)

		return output
