import sys
# add the modules outside the model folder to the environment
sys.path.insert(0, r'C:\Users\Brooks\github\splitr')

# library functions
import torch
import numpy as np
import os
import time
import pandas as pd

import model_utils

# this is really a constructor for a bidirectional LSTM but i figured
# BD_LSTM  was only 2 letters off of BDSM so why not
class BDSM(torch.nn.Module):
	def __init__(self, num_inputs, num_hidden_layers, num_output_layers):
		super(BDSM, self).__init__()

		self.rnn = torch.nn.LSTM(num_inputs, num_hidden_layers, bidirectional=True)
		self.linear = torch.nn.Linear(num_hidden_layers*2, num_output_layers)

	def forward(self, x):
		# print('bdsm input size:', x.shape)
		rnn_output, _ = self.rnn(x)
		# print('rnn output size', rnn_output.shape)
		batch_num, base,height = rnn_output.size()

		# print('bdsm: rnn shape out is ', rnn_output.shape)
		x = rnn_output.view(batch_num*base, height)
		x = self.linear(x)
		x = x.view(batch_num, base, -1)

		# print('bdsm: final shape is ', x.shape)

		return x

# https://arxiv.org/pdf/1507.05717.pdf
class model(torch.nn.Module):
	def __init__(self, channel_count=1):
		super(model, self).__init__()

		# add batch norm later
		# add dropout to cnn layers
		self.softmax = torch.nn.LogSoftmax(dim=1)

		# CONVOLUTIONS
		self.cnn1 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=channel_count,out_channels=64,kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		) # out: torch.Size([3, 64, 39, 249])

		self.cnn2 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		) # out: torch.Size([3, 128, 18, 123])

		self.cnn3 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
		torch.nn.ReLU(),
		) # out: torch.Size([3, 256, 8, 60])

		self.cnn4 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((1,2), stride = 2)
		) # out: torch.Size([3, 512, 3, 29])

		self.cnn5 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		torch.nn.ReLU(),
		torch.nn.BatchNorm2d(512),
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		torch.nn.MaxPool2d((3,3), stride= 2)
		) # out:torch.Size([x, 512, 2, 28])

		num_hidden = 200
		num_output = 200
		unique_char_count = 57
		self.rnn = torch.nn.Sequential(
		BDSM(num_inputs=512, num_hidden_layers=num_hidden,num_output_layers=num_output),
		BDSM(num_inputs=num_output, num_hidden_layers=num_hidden, num_output_layers=unique_char_count )
		)
	def forward(self, x):
		# q = lambda x: print(x.shape)
		t = self.cnn1(x)
		t = self.cnn2(t)
		t = self.cnn3(t)
		t = self.cnn4(t)
		t = self.cnn5(t)

		# rearrange the tensor
		# batch x 512 x 0 x 27 ==== > batch x 27 x 512
		# t = t.view(t.shape[0], 27, 512)
		t = t.view(27, t.shape[0], 512)

		t = self.rnn(t) # batch x 27 x 57

		predicted = self.softmax(t)
		return predicted


def train(epochs=10000):
	batch_size = 2
	workers = 8
	shuffle = True

	a1 = time.time()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# device = torch.device('cpu')
	print('device being used: %s' % device)

	a2 = time.time()
	OCR = model().to(device)

	a3 = time.time()
	criterion = torch.nn.CTCLoss().to(device)
	optimizer = torch.optim.SGD(OCR.parameters(), lr=1e-1)

	a4 = time.time()
	training_set = model_utils.OCR_dataset_loader(r'C:\Users\Brooks\github\Splitr\data\final.csv', r'C:\Users\Brooks\Desktop\OCR_data', encode='vector',crop_dataset=False)
	training_data = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=workers, shuffle=shuffle)

	a5 = time.time()
	for i in range(epochs):
		epoch_loss = 0
		count = 0

		for training_img_batch, training_label_batch in training_data:
			count += 1
			training_img_batch = training_img_batch[:,None,:,:].to(device)
			training_label_batch = training_label_batch.to(device)
			training_label_batch = training_label_batch.squeeze()

			optimizer.zero_grad()

			predicted_labels = OCR.forward(training_img_batch)

			dim1, dim2, dim3 = predicted_labels.shape
			predicted_size = torch.full((dim2,),dim1, dtype=torch.long)
			length = torch.randint(1,100,(16,), dtype=torch.long)

			# debug code for sizes
			print('pred label', predicted_labels.size())
			print('training_label_batch', training_label_batch.size())
			print('predicted_size', predicted_size.size())
			print('length', length.size())

			########################################################
			#### soemthing is not correct with the loss function####
			########################################################
			
			loss = criterion(predicted_labels, training_label_batch, predicted_size, length)


			loss.backward()
			optimizer.step()

			epoch_loss += loss.item() / dim2
			print(count, loss.item()/dim2)

			break
		break

		outstr = 'epoch: %s loss: %s loss decrease:%s'
		if i > 0:
			outstr = outstr % (i+1, epoch_loss, prev_epoch_loss- epoch_loss)
		else:
			outstr =  outstr % (i+1, epoch_loss, epoch_loss)

		print(outstr)
		prev_epoch_loss = epoch_loss

	a6 = time.time()

	print('device: %s\nmodel: %s\ncrit + optim: %s\ndataset: %s\npass: %s \n' %(a2-a1, a3-a2, a4-a3, a5-a4, a6-a5))

if __name__ == '__main__':
	train()
