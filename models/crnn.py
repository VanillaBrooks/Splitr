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
	def __init__(self, num_inputs, num_hidden_layers,char_in, char_out, layer_count=1):
		super(BDSM, self).__init__()
		self.char_out = char_out

		# make the last layer not have a linear layer inside

		self.rnn = torch.nn.LSTM(num_inputs, num_hidden_layers, num_layers=layer_count, bidirectional=True, batch_first=True)
		self.linear = torch.nn.Linear(char_in, char_out)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		# print('>>starting rnn that has output chars of', x.shape)
		rnn_output, _ = self.rnn(x)
		# print('raw rnn out', rnn_output.shape)
		batch, char_count, depth = rnn_output.shape
		rnn_output = rnn_output.contiguous().view(batch*depth, char_count)
		# print('reshaped rnn out', rnn_output.shape)
		linear = self.linear(rnn_output)
		output = linear.view(batch, self.char_out, depth)
		# print('after linear shape', output.shape)
		output =self.relu(output)

		return output

# Convolution cell with adjustable activation / maxpool size / batchnorm
class CNN_cell(torch.nn.Module):
	def __init__(self,in_channels=False,out_channels=False,kernel_size=3,activation=False, pool_shape=False, pool_stride=False, batchnorm=False):
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

		# add dropout to cnn layers
		self.softmax = torch.nn.LogSoftmax(dim=2)

		# CONVOLUTIONS
		_cnn_layer = []
		_cnn_layer.append(CNN_cell(in_channels=1,   out_channels=64,  kernel_size=3, activation='relu', pool_shape=False, pool_stride=False))

		_cnn_layer.append(CNN_cell(in_channels=64 , out_channels=128, kernel_size=3, activation='relu', pool_shape=(2,2), pool_stride=2))

		_cnn_layer.append(CNN_cell(in_channels=128, out_channels=256, kernel_size=3, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=256, out_channels=512, kernel_size=3, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=3, activation='relu', pool_shape=(1,2), pool_stride=2))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu', batchnorm=512))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu'))

		_cnn_layer.append(CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu'))


		# RNN LAYERS
		_bdsm_layer = []# 2048
		# _bdsm_layer.append(BDSM(num_inputs=512, num_hidden_layers=num_hidden, char_in=56,char_out=56, layer_count=rnn_layer_stack))
		# _bdsm_layer.append(BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden, char_in=85,char_out=140, layer_count=1))
		# _bdsm_layer.append(BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden,char_in= 140, char_out=190, layer_count=1))
		# _bdsm_layer.append(BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden,char_in= 190, char_out=250, layer_count=1))
		# _bdsm_layer.append(BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden,char_in= 250, char_out=350, layer_count=1))

		inc = 1.8
		max_len = 600
		current = 53
		p = 0

		while current < max_len:
			p+=1
			prev = current
			current = int(inc * prev)
			_bdsm_layer.append(BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden,char_in= prev, char_out=current, layer_count=1))


		print('number of rnns stacked %s' % p)

		# l1 = BDSM(num_inputs=2048, num_hidden_layers=num_hidden, char_in=56,char_out=65, layer_count=rnn_layer_stack)
		# l2 = []
		# max_1, cur = 56, 75
		# while cur < max_1:
		# 	old = cur
		# 	cur = int(cur * 1.05)
		# 	l2.append(torch.nn.Linear(old, cur))
		# 	l2.append(torch.nn.ReLU())
		#
		# l2 = torch.nn.Sequential(*l2)
		#
		# l3 = BDSM(num_inputs=num_hidden*2, num_hidden_layers=num_hidden, char_in=56,char_out=65, layer_count=rnn_layer_stack)
		# l4 = []
		#
		# max_1, cur = 75, 90
		# while cur < max_1:
		# 	old = cur
		# 	cur = int(cur * 1.05)
		# 	l4.append(torch.nn.Linear(old, cur))
		# 	l4.append(torch.nn.ReLU())
		#
		# l4 = torch.nn.Sequential(*l4)


		# CHAR activations (transcription)
		self.linear = torch.nn.Sequential(
		torch.nn.Linear(in_features=num_hidden*2, out_features=unique_char_count),torch.nn.ReLU())

		self.cnn = torch.nn.Sequential(*_cnn_layer)
		self.rnn = torch.nn.Sequential(*_bdsm_layer)

	def forward(self, x):
		t = self.cnn(x)
		batch, depth, height, base = t.shape
		# print('raw cnn shape: ', t.shape)
		# import sys


		# cnn_output = t.view(batch, height, depth*base)
		cnn_output = t.view(batch, base, height*depth)


		# print(' NEW after reshape', cnn_output.shape, type(cnn_output))
		# sys.exit('exits')

		rnn_output = self.rnn(cnn_output)
		batch, char_len, depth = rnn_output.shape
		rnn_output = rnn_output.contiguous().view(batch*char_len, depth)

		# print('rnn output ', rnn_output.shape)

		output = self.linear(rnn_output).view(batch, char_len, -1)
		output = self.softmax(output)

		return output
