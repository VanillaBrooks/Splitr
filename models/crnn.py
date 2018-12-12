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

		return x

# CNN cell creates convolutional layers w/ optional activation functions
# this is done to save lines of code as well as fast prototyping
class CNN_cell(torch.nn.Module):
	def __init__(self,in_channels=False,out_channels=False,kernel_size=False,activation=False, pool_shape=False, pool_stride=False, batchnorm=False):
		super(model, self).__init__()

		self.cnn = torch.nn.Sequential()

		if in_channels and out_channels:
			self.cnn.add_module(torch.nn.Conv2d(in_channels, out_channels,kernel_size))
		if activation:
			self.cnn.add_module(self.find_activation(activation))
		if batchnorm:
			self.cnn.add_module(torch.nn.BatchNorm2d(batchnorm))
		if pool_shape and pool_stride:
			self.cnn.add_module(torch.nn.MaxPool2d(pool_shape, pool_stride))

	def find_activation(self, activation):
		if activation == 'relu':
			return torch.nn.ReLU()
		else:
			print('activation function call %s is not configured' % activation )
	def forward(self, input_tensor):
		output = self.cnn(input_tensor)
		return output


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
		# self.cnn1 = CNN_cell(in_channels=channel_count, out_channels=64, kernel_size=3, activation='relu', pool_shape=(2,2), pool_stride=2)


		self.cnn2 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		) # out: torch.Size([3, 128, 18, 123])
		# self.cnn2 = CNN_cell(in_channels=64, out_channels=128, kernel_size=3, pool_shape=(2,2), pool_stride=2)


		self.cnn3 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
		torch.nn.ReLU(),
		) # out: torch.Size([3, 256, 8, 60])
		# self.cnn3 = CNN_cell(in_channels=128, out_channels=256, kernel_size=3, activation='relu')



		self.cnn4 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((1,2), stride = 2)
		) # out: torch.Size([3, 512, 3, 29])
		# self.cnn4 = CNN_cell(in_channels=256, out_channels=512, kernel_size=3, activation='relu')
		# self.cnn5 = CNN_cell(in_channels=512, out_channels=512, kernel_size=3, activation='relu', pool_shape=(1,2), pool_stride=2)



		self.cnn5 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		torch.nn.ReLU(),
		torch.nn.BatchNorm2d(512),
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		torch.nn.MaxPool2d((3,3), stride= 2)
		) # out:torch.Size([x, 512, 2, 28])
		# self.cnn6 = CNN_cell(in_channels=512, out_channels=512, kernel_size=2, activation='relu', batchnorm=512)
		# self.cnn7 = CNN_cell(in_channels=512, out_channels=512, kernel_size=2, pool_shape=(3,3), pool_stride=2)


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
	batch_size = 90
	workers = 8
	shuffle = True

	model_save_path = r'C:\Users\Brooks\github\Splitr\models\%s_%s.model'
	start_time = time.time()

	# use gpu if it is available
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device being used: %s' % device)

	# initialize the model
	OCR = model().to(device)

	# option to resume training from last checkpoint
	if LOAD_PATH:
		OCR.load_state_dict(torch.load(LOAD_PATH))

	# optimizer for stepping and CTC loss function for backprop
	criterion = torch.nn.CTCLoss().to(device)
	optimizer = torch.optim.SGD(OCR.parameters(), lr=1e-1)

	# initialize the Dataset. This is done so that we can work with more data
	# than what is loadable into RAM
	training_set = model_utils.OCR_dataset_loader(
		csv_file_path = r'C:\Users\Brooks\github\Splitr\data\final.csv',
		path_to_data =r'C:\Users\Brooks\Desktop\OCR_data',
		crop_dataset=False,
		transform = False)

	# initialize the training data into a dataloader for batch pulling
	# and shuffling the data
	training_data = torch.utils.data.DataLoader(
		training_set,
		batch_size=batch_size,
		num_workers=workers,
		shuffle=shuffle)

	# iterate through all the designated epochs for training
	for i in range(epochs):
		# runing variables to keep track of data between batches
		epoch_loss = 0	# accumulated loss for the epoch
		count = 0		# what batch number is currently being handled

		# iterate through the Dataset to pull batch data
		for training_img_batch, training_label_batch in training_data:
			count += 1

			# convert the image batch to a 4D tensor (avoid error in forward call)
			training_img_batch = training_img_batch[:,None,:,:].to(device)
			# construct a list of all the lengths of strings in the data
			target_length_list = [len(word) for word in training_label_batch]

			# convert all the strings pulled from target_length_list to
			# tensors so that they can be fed to loss function
			training_label_batch = model_utils.encode_single_vector(
				training_label_batch,
				training_set.max_str_len,
				training_set.unique_chars
			).squeeze().to(device)

			# get the predicted optical characters from the model
			predicted_labels = OCR.forward(training_img_batch)

			# find the dimentions of the return tensor and create a vector
			# of the word sizes being used
			#     Note: the target_length does not need to hold actual lengths becuase
			#     CTC loss will evaluate 0 as space
			dim1, dim2, dim3 = predicted_labels.shape
			predicted_size = torch.full((dim2,),dim1, dtype=torch.long)	# this is the size of the word that came from the predictor
			target_length = torch.tensor(target_length_list)

			# find the loss of the batch
			loss = criterion(predicted_labels, training_label_batch, predicted_size, target_length)

			# zero the accumulated gradients / backpropagate / optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# add losses to the running total
			epoch_loss += loss.item()
			print(count, 100*loss.item()/dim2)

		# output the epoch loss and the change from last loss
		outstr = 'epoch: %s loss: %s loss decrease:%s'
		if i > 0:
			outstr = outstr % (i+1, epoch_loss, prev_epoch_loss- epoch_loss)
		else:
			outstr =  outstr % (i+1, epoch_loss, epoch_loss)

		# save the model every 5 epochs
		if i % 5 == 0:
			torch.save(OCR.state_dict(), model_save_path % (int(start_time), i))

		print(outstr)
		prev_epoch_loss = epoch_loss

if __name__ == '__main__':
	train()
