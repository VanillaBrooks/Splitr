import torch
import torch.nn as nn
import numpy as np
import cv2 as cv

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

def init_model():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	return model().to(device)

def load_data():
	pass

if __name__ == '__main__':
	m = init_model()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	img =cv.imread('example.jpg')
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


	from build_tensor import build_tensor_stack

	tensor = build_tensor_stack([img for i in range(1)]).to(device)

	m.forward(tensor)
