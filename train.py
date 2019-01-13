import torch
import time
import model_utils
from models import crnn
import os
import torchvision
from multiprocessing import Pool

class average():
	def __init__(self, n=100):

		self.max = n
		self.total_change = 0
		self.current = []
		self.previous = []

	def new_number(self, num):
		self.current.append(num)
		if len(self.current) > self.max:
			self.previous.append(self.current.pop(0))

		if len(self.previous) > self.max:
			self.previous.pop(0)

		ret = self._avg_diff()

		return ret

	def _average(self, var=False):
		if var:
			return float(sum(var)) / max(len(var), 1)

		else:
			print('var was false')
			return float(sum(self.current)) / max(len(self.current), 1)


	def _avg_diff(self):
		long =  self._average(self.previous)
		short = self._average(self.current)

		diff =  long - short
		if len(self.previous) >= self.max:
			self.total_change += diff

		return diff

def train(epochs=10000,batch_size=2, workers=8, shuffle=True,channel_count=1,
	num_hidden= 256, unique_char_count=57,rnn_layer_stack=1, LOAD_MODEL=False,
	LOAD_MODEL_PATH=False, learning_rate = 1e-3, model_name='model'):

	# MODEL_SAVE_PATH = r'C:\Users\Brooks\github\Splitr\models\%s_%s_%s.model'
	MODEL_SAVE_PATH = r'E:\models\%s.model' % model_name
	TXT_SAVE_PATH = r'C:\Users\Brooks\github\Splitr\models\%s.txt'

	START_TIME = int(time.time())
	print(MODEL_SAVE_PATH)

	# use gpu if it is available
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device being used: %s' % device)

	# initialize the model
	OCR = crnn.model(channel_count,num_hidden, unique_char_count,rnn_layer_stack).to(device)

	# option to resume training from last checkpoint
	if LOAD_MODEL:
		print('model loaded')
		OCR.load_state_dict(torch.load(LOAD_MODEL_PATH))

	# optimizer for stepping and CTC loss function for backprop
	criterion = torch.nn.CTCLoss().to(device)
	# optimizer = torch.optim.SGD(OCR.parameters(), lr=learning_rate, momentum=.9, dampening=0, weight_decay=0, nesterov=False)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
	# optimizer = torch.optim.Adam(OCR.parameters(), lr=learning_rate, amsgrad=True)
	# optimizer = torch.optim.Adadelta(OCR.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
	optimizer = torch.optim.RMSprop(OCR.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

	# initialize the Dataset. This is done so that we can work with more data
	# than what is loadable into RAM
	training_set = model_utils.OCR_dataset_loader(
		csv_file_path = r'C:\Users\Brooks\github\Splitr\data\training_data.csv',
		path_to_data =r'C:\Users\Brooks\Desktop\OCR_data',
		# transform = torchvision.transforms.Compose([model_utils.Rotate(5), model_utils.Pad()])
		transform = torchvision.transforms.Compose([model_utils.Pad()]))

	# initialize the training data into a dataloader for batch pulling
	# and shuffling the data
	training_data = torch.utils.data.DataLoader(
		training_set,
		batch_size=batch_size,
		num_workers=workers,
		shuffle=shuffle)

	epoch_loss = 0
	previous_save_path = False
	avg = average()

	# iterate through all the designated epochs for training
	for i in range(1,epochs+1):
		# runing variables to keep track of data between batches
		prev_epoch_loss = epoch_loss
		run_loss, prev_run_loss,epoch_loss,count = 0, 0, 0, 0

		# iterate through the Dataset to pull batch data
		for training_img_batch, training_label_batch in training_data:
			count += 1

			# convert the image batch to a 4D tensor (avoid error in forward call)
			training_img_batch = training_img_batch.to(device)

			# construct a list of all the lengths of strings in the data
			target_length_list = [len(word) for word in training_label_batch]
			max_str_len = max(target_length_list)

			# convert all the strings pulled from target_length_list to
			# tensors so that they can be fed to loss function
			training_label_batch = model_utils.encode_single_vector(
				training_label_batch,
				max_str_len,
				training_set.unique_chars).squeeze().to(device)

			# get the predicted optical characters from the model
			predicted_labels = OCR.forward(training_img_batch)

			# find the dimentions of the return tensor and create a vector
			# of the word sizes being used
			#     Note: the target_length does not need to hold actual lengths becuase
			#     CTC loss will evaluate 0 as space
			batch, _pred_len, _char_activation = predicted_labels.shape

			predicted_size = torch.full((batch,),_pred_len, dtype=torch.long)	# this is the size of the word that came from the predictor
			target_length = torch.tensor(target_length_list)

			# find the loss of the batch
			loss = criterion(predicted_labels.permute(1,0,2), training_label_batch, predicted_size, target_length)

			# zero the accumulated gradients / backpropagate / optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# scheduler.step(loss)

			# add losses to the running total
			_loss = 100* loss.item()/batch
			run_loss += (_loss)
			print(count, max_str_len, _loss)

			if count % 100 == 0:
				running_average = avg.new_number(run_loss)
				batch_out_str = '%s : %s run: %s avg: %s total: %s dec: %s' % (i, count/100, run_loss, running_average,  avg.total_change, prev_run_loss-run_loss)
				print(batch_out_str)

				with open(TXT_SAVE_PATH % (START_TIME), 'a') as file:
					file.write(batch_out_str + '\n')
				if count % (100 * 5) == 0:
					try:
						torch.save(OCR.state_dict(), MODEL_SAVE_PATH)
					except Exception as e:
						print('|||| EXCEPTION : ', e)

				epoch_loss += run_loss
				prev_run_loss = run_loss
				run_loss = 0

		# output the epoch loss and the change from last loss
		outstr = 'epoch: %s loss: %s loss decrease:%s'% (i, epoch_loss, prev_epoch_loss- epoch_loss)
		with open(TXT_SAVE_PATH % START_TIME, 'a') as file:
			file.write(outstr + '// END OF EPOCH' + '\n')

if __name__ == '__main__':

	LOAD_MODEL =True
	LOAD_MODEL_PATH = r'E:\models\model_sgd_3.model'
	while True:
		try:
			train(
				epochs=10000,
				batch_size=32,
				workers=8,
				shuffle=True,
				channel_count=1,
				num_hidden= 256,
				unique_char_count=80,
				rnn_layer_stack=1,
				LOAD_MODEL= LOAD_MODEL,
				LOAD_MODEL_PATH=LOAD_MODEL_PATH,
				learning_rate = 1e-2,
				model_name='model_rms')
		except Exception as e:
			print(':::::main exception: ', e)
