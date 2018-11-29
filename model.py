import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os
import time
from build_tensor import build_tensor_stack

# https://arxiv.org/pdf/1306.2795v1.pdf
class model(torch.nn.Module):
	def __init__(self, channel_count=1):
		super(model, self).__init__()

		# add batch norm later
		# add dropout to cnn layers
		self.a = torch.nn.Conv2d(1,64,2)

		# CONVOLUTIONS
		self.cnn1 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=channel_count,out_channels=64,kernel_size=2),
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
		torch.nn.MaxPool2d((2,2), stride = 2)
		) # out: torch.Size([3, 256, 8, 60])

		self.cnn4 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride = 2)
		) # out: torch.Size([3, 512, 3, 29])

		self.cnn5 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		) # out:torch.Size([x, 512, 2, 28])

		# RECURRENT
		self.lstm1 = torch.nn.Sequential(
		torch.nn.LSTM(input_size=56, hidden_size=56, num_layers=2,batch_first=False, bidirectional=False)
		)

		self.lstm2 = torch.nn.Sequential(
		torch.nn.LSTM(input_size=56, hidden_size=56, num_layers=2)
		)

	def forward(self, x):
		# q = lambda x: print(x.shape)
		t = self.cnn1(x)
		t = self.cnn2(t)
		t = self.cnn3(t)
		t = self.cnn4(t)
		t = self.cnn5(t)


		# new_tensor = t.view(t.shape[0], 512, -1)
		new_tensor = t.view(t.shape[0], 512, 56)
		# new_tensor = t.view(t.shape[0], 56, 512)
		print(new_tensor.size())


		t= self.lstm1(new_tensor)
		# returns output, hidden
		# hidden is fed into the next network

		new_t = t[0]
		# print(type(new_t))
		# print(new_t.shape)
		t = self.lstm2(new_t)
		print(t[0].shape)

		return t[0]


def train(epochs=1000):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device being used: %s' % device)

	OCR = model().to(device)

	criterion = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.SGD(OCR.parameters(), lr=1e-2, momentum=.9)#.to(device)

	with open('finaltraining_data_y.json') as labels:
		import json
		data = json.load(labels)

	ct = 10000

	labels, filenames = [], []
	index = 1
	n = len(data)

	for key in data.keys():
		val = data[key]	# the label
		labels.append(key)
		filenames.append(os.path.join(r'D:\OCR_data',key + '.jpg'))

		if index % 10000 == 0:
			print('percent done: %s' % (index * 100 / n))
		index +=1

		# this is debug code
		# dont gen images we wont use
		if index > ct:
			print('::::::::::breaking now %s %s' % (index, ct))
			break

	training_images = []
	s = time.time()
	i = 1
	ct = 10000
	print('\n the length of filenames ' , len(filenames))
	for path in filenames[:ct]:
		print(path)
		im = cv.imread(path)
		im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
		training_images.append(np.array(im))

		if i % 10 ==0:
			print('percent done images: %s' % (i*100 / ct))
		i += 1
	print('total time to import %s images: %s' % (ct,time.time() - s))
	from build_tensor import build_tensor_stack

	print('datatype: %s length %s' % (training_images[0].dtype, len(training_images)))
	s2 = time.time()

	training_data = build_tensor_stack(training_images)
	print('time to convert to tensors: %s' % (time.time() - s2))
	print('total time: %s' % (time.time() - s))

	print('len of returned data')
	print(training_data.shape)


	# training_data = torch.Tensor(training_#d_results).to(device)







if __name__ == '__main__':
	train()

	# x = np.zeros((80,500)).reshape(80,500)
	#
	# gray = lambda x: cv.cvtColor(x, cv.COLOR_BGR2GRAY)
	#
	# # this image is generated (it will error)
	# im = np.array(gray(cv.imread(r'D:\OCR_data2\i.jpg')), dtype=np.uint8)
	#
	# # this image is not (it wont error)
	# new_im = gray(cv.imread('receipt2.jpg'))
	#
	# print('the image that errors: ')
	# print(im.shape)
	# print(im.dtype)
	#
	# print('the image that does not error')
	# print(new_im.shape)
	# print(new_im.dtype)
	#
	#
	# x = build_tensor_stack([x])
