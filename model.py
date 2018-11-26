import torch
import torch.nn as nn

class model(torch.nn.Module):
	def __init__(self, batch_size):
		super(model, self).__init__()

		# add batch norm later
		self.l1_1 = torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2) # (batch_size, )
		self.l1_2= torch.nn.ReLU(self.l1_1)
		self.l1_3 = torch.nn.MaxPool2d((2,2), stride=2)

		# CONVOLUTIONS
		self.cnn1 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=2),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		)

		self.cnn2 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		)

		self.cnn3 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride = 2)
		)
		
		self.cnn4 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride = 2)
		)

		self.cnn5 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		# torch.nn.ReLU(),
		# torch.nn.MaxPool2d((2,2), stride = 2)
		)

		# RECURRENT

		
	def forward(self, x):
		print('insize is ', x.size())

		# manually by layers
		a = self.l1_1(x)
		print('outsize of cnn is', a.size())

		a = self.l1_2(a)
		print('outsize of relu is', a.size())

		a = self.l1_3(a)
		print('outsize of pool is ', a.size())

		# container for all the first layers
		b = self.cnn1(x)
		print('\noutsize of cnn1 is' ,b.size())

		b = self.cnn2(b)
		print('outsize of cnn2 is', b.size())

		b= self.cnn3(b)
		print('outside of cnn3 is', b.size())

		b= self.cnn4(b)
		print('outside cnn4 is ', b.size())

		b= self.cnn5(b)
		print('outside cnn5 is', b.size())


batch_size =1
channels = 3 # 3 for RGB, 1 for greyscale TBD
width = 100	# im width
height = 100 # im height

x = model(b_size)
x.forward(torch.Tensor(batch_size,channels,width,height))
# (batch size (i think), in_channels, width, height)


# insize is  torch.Size([64, 1, 2, 2])
# outsize is  torch.Size([64, 64, 1, 1])
