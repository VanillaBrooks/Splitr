import torch
import torch.nn as nn
import numpy as np
import cv2 as cv

# https://arxiv.org/pdf/1306.2795v1.pdf
class model(torch.nn.Module):
	def __init__(self, channel_count):
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
		torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2,batch_first=False, bidirectional=False)
		)

		self.lstm2 = torch.nn.Sequential(
		torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2)
		)

	def forward(self, x):
		q = lambda x: print(x.shape)
		t = self.cnn1(x)
		# q(t)
		t = self.cnn2(t)
		# q(t)
		t = self.cnn3(t)
		# q(t)
		t = self.cnn4(t)
		# q(t)
		t = self.cnn5(t)
		# print('ending shape')
		# print(t.shape)

		# new_tensor = t.view(t.shape[0], 512, -1)
		new_tensor = t.view(t.shape[0], -1, 512)


		t= self.lstm1(new_tensor)
		# returns output, hidden
		# hidden is fed into the next network

		new_t = t[0]
		# print(type(new_t))
		# print(new_t.shape)
		t = self.lstm2(new_t)
		
		return t[0]


batch_size =1
channels = 1 # 3 for RGB, 1 for greyscale TBD
width = 80	# im width
height = 500 # im height


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)


img =cv.imread('example.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


from build_tensor import build_tensor_stack

tensor = build_tensor_stack([img for i  in range(1)]).to(device)

print(tensor.shape, len(tensor.shape), len(tensor))

x = model(channels).to(device)

x.forward(tensor)


# a = torch.Tensor(batch_size,channels,width,height)


# x.forward(torch.Tensor(batch_size,channels,width,height))#.to(device))
# output size : torch.Size([3, 512, 2, 28])


# if there is an error with the tensor it is probably in the wrong dimention
# convert tensor from 3d to 4d (aka one batch)
# tensor = tensor[None, :, :, :]
