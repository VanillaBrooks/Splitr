import torch
import cv2 as cv
import sys
import os


sys.path.insert(0, r'C:\Users\Brooks\github\splitr')
from build_training_data import OCR_DATA_PATH

class Attention_OCR_Model(torch.nn.Module):
	def __init__(self):
		super(Attention_OCR_Model, self).__init__()

		self.relu = torch.nn.ReLU()
		self.softmax = torch.nn.Softmax()


		# CONVOLUTIONS
		self.cnn1 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3),
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
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
		) # out:torch.Size([x, 512, 2, 28])

		# Attention Layer ?



		# RECURRENT LAYERS

		self.lstm1 = torch.nn.LSTM(input_size=224, hidden_size=112, num_layers=1, bidirectional=True)

		self.lstm2 = torch.nn.LSTM(input_size=224, hidden_size=112, num_layers=1, bidirectional=True)


	def forward(self,x):
		t = self.cnn1(x)
		t = self.cnn2(t)
		t = self.cnn3(t)
		t = self.cnn4(t)
		t = self.cnn5(t)

		t = t.view(t.shape[0], 512, -1)

		t, _ = self.lstm1(t)
		t = self.relu(t)

		t, _ = self.lstm2(t)
		t = self.relu(t)

		

		print(t.shape)
if __name__ == '__main__':
	model = Attention_OCR_Model()

	imagefiles = os.listdir(OCR_DATA_PATH)
	imfile = imagefiles[0]
	image_path = os.path.join(OCR_DATA_PATH, imfile)
	image = cv.imread(image_path)

	model.forward(torch.from_numpy(image).float().view(1,3,80,500))

	# cv.imshow('ImageWindow', image)
	# cv.waitKey()
