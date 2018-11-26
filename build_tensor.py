import torch
import cv2
import numpy as np

# lambda helper function to calculate the amount of padding that should be added to the tensor
calculate_padding = lambda desired, actual: desired-actual

def build_tensor_stack(list_of_numpy_array):
	DESIRED_HEIGHT = 80
	DESIRED_WIDTH = 500

	# the input was not a list, prevents the error in the for loop
	if type(list_of_numpy_array) != list:
		raise TypeError('build_tensor_stack expects the argument to be a list of numpy arrays you input a variable that was not a list')

	tensor_list = []

	for i in range(len(list_of_numpy_array)):
		image = list_of_numpy_array[i]
		# height, width, channels = image.shape
		# height, width, channels = float(height), float(width), float(channels)
		height, width = image.shape
		height, width = float(height), float(width)

		# the current 'image' is not a numpy arr
		if type(image) != np.ndarray:
			raise TypeError('the input list contains an element that is not a numpy array')
		# if they have not been made into black and white
		# elif channels != 1:
		# 	raise ValueError('build_tensor expects 1 channel images (greyscale)')


		#################################################
		#	actual function code starts here basically	#
		#################################################

		height_ratio, width_ratio = 0,0

		# find the current conditions of the height
		if height <= DESIRED_HEIGHT:

			padding_height = calculate_padding(DESIRED_HEIGHT, height)
		else:
			height_ratio = DESIRED_HEIGHT/ height

		# find the current conditions of the width
		if width <= DESIRED_WIDTH:
			padding_width = calculate_padding(DESIRED_WIDTH, width)
		else:
			width_ratio = DESIRED_WIDTH / width

		# this block finds out which ratio is bigger
		# since they both need to be adjusted to by the same ratio
		# to preserve the aspect ratio
		if width_ratio < height_ratio:									# if theres a bug its prolly in here
			height_ratio = width_ratio
		if height_ratio < width_ratio:
			width_ratio = height_ratio

		# if the height or widh ratio is not 0 (as they were originall assigned)
		# then we need to calculate a new image size
		if height_ratio or width_ratio:
			new_height = int(height_ratio * height)
			new_width  = int(width_ratio * width)

			image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)

			# recalculate the padding that the image needs
			padding_height = calculate_padding(DESIRED_HEIGHT, new_height)
			padding_width = calculate_padding(DESIRED_WIDTH, new_width)

			# just to make sure that it was resized correctly
			# will throw an error if the image is still bigger than it
			# is supposed to be
			if padding_height <0 or padding_width <0:
				raise ValueError('ooooh shit the images were not resized correctly you better hit up brooks about this one')

		# finally we add the padding to the array
		# left right top down
		pad_function = torch.nn.ConstantPad2d((padding_width, padding_height),255)
		resulting_image = pad_function(torch.from_numpy(image))

		tensor_list.append(resulting_image)

	final_stack = torch.stack(tensor_list)

	return final_stack
