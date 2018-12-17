import torch
import time
import model_utils
from models import crnn
import os

def train(epochs=10000,batch_size=2, workers=8, shuffle=True,channel_count=1,num_hidden= 256, unique_char_count=57,rnn_layer_stack=1, LOAD_MODEL=False, LOAD_MODEL_PATH=False):
	MODEL_SAVE_PATH = r'C:\Users\Brooks\github\Splitr\models\%s_%s_%s.model'
	TXT_SAVE_PATH = r'C:\Users\Brooks\github\Splitr\models\%s.txt'

	START_TIME = int(time.time())

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
	# optimizer = torch.optim.SGD(OCR.parameters(), lr=.001, momentum=.90)
	optimizer = torch.optim.Adam(OCR.parameters(), lr=5e-5, )

	# initialize the Dataset. This is done so that we can work with more data
	# than what is loadable into RAM
	training_set = model_utils.OCR_dataset_loader(
		csv_file_path = r'C:\Users\Brooks\github\Splitr\data\training_data.csv',
		path_to_data =r'C:\Users\Brooks\Desktop\OCR_data',
		transform = torchvision.transforms.Compose([Rotate(20), Pad()]))

	# initialize the training data into a dataloader for batch pulling
	# and shuffling the data
	training_data = torch.utils.data.DataLoader(
		training_set,
		batch_size=batch_size,
		num_workers=workers,
		shuffle=shuffle)

	epoch_loss = 0
	previous_save_path = False
	# iterate through all the designated epochs for training
	for i in range(1,epochs+1):
		# runing variables to keep track of data between batches
		prev_epoch_loss = epoch_loss
		run_loss, prev_run_loss,epoch_loss,count = 0, 0, 0, 0

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

			# add losses to the running total
			_loss = 100* loss.item()/batch
			run_loss += (_loss)
			print(count, _loss)

			if count % 100 == 0:
				batch_out_str = 'epoch: %s iter: %s running loss: %s RL decrease:%s' % (i, count, run_loss, prev_run_loss-run_loss)
				print(batch_out_str)

				with open(TXT_SAVE_PATH % (START_TIME), 'a') as file:
					file.write(batch_out_str + '\n')
				if count % 100 *1 == 0:
					save_path = MODEL_SAVE_PATH % (START_TIME, i, count)
					torch.save(OCR.state_dict(), save_path)
					if previous_save_path:
						os.remove(previous_save_path)
					previous_save_path = save_path

				epoch_loss += run_loss
				prev_run_loss = run_loss
				run_loss = 0

		# output the epoch loss and the change from last loss
		outstr = 'epoch: %s loss: %s loss decrease:%s'% (i, epoch_loss, prev_epoch_loss- epoch_loss)
		with open(TXT_SAVE_PATH % START_TIME, 'a') as file:
			file.write(outstr + '// END OF EPOCH' + '\n')

if __name__ == '__main__':
	# model parameters
	channel_count=1
	num_hidden= 256
	unique_char_count=57
	rnn_layer_stack=2

	LOAD_MODEL =True
	LOAD_MODEL_PATH = r'models\CRNN_2rnn_256hidden_57char_1channel.model'


	train(
		epochs=10000,
		batch_size=90,
		workers=8,
		shuffle=True,
		channel_count=channel_count,
		num_hidden= num_hidden,
		unique_char_count=unique_char_count,
		rnn_layer_stack=rnn_layer_stack,
		LOAD_MODEL= LOAD_MODEL,
		LOAD_MODEL_PATH=LOAD_MODEL_PATH)
