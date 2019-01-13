from cv2 import imwrite
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import string
import pandas as pd
import random
import torchvision
import time


from multiprocessing import Process

class FontChoice():
	def __init__(self, font_directory):
		self.original_fonts = os.listdir(font_directory)
		self.root = font_directory
		self._init_fonts()

	def pick_font(self):
		if len(self.fonts) == 0:
			self._init_fonts()
		font = self.fonts.pop(0)
		return os.path.join(self.root, font)
	def _init_fonts(self):
		self.fonts = self.original_fonts[:]

class WordChoice():
	def __init__ (self, list_of_words):
		self.original_words = list_of_words
		self._init_words()
		self.len = len(self.words)

	def pick_word(self, count):
		words_to_return = []
		for i in range(count):
			if len(self.words) == 0:
				self._init_words()

			words_to_return.append(self.words.pop(0))
			self.len -= 1
		return words_to_return

	def _init_words(self):
		self.words = self.original_words[:]
		random.shuffle(self.words)

def make_string(list_of_words, number=True):
	def make_number(length, include_decimal):
		outstr = ''
		for i in range(length):
			outstr += str(random.randint(0,9))
		if include_decimal:
			outstr = outstr[:-2] + '.' + outstr[-2:]
		return outstr

	if number:
		number = make_number(random.randint(2,5), random.randint(0,1))
		insertion_location = random.randint(0,len(list_of_words)-1)
		list_of_words = list_of_words[:insertion_location] + [number] + list_of_words[insertion_location:]

	return(''.join([i + ' ' for i in list_of_words]))

def ascii_only(file_to_filter):
	with open(file_to_filter, 'r', encoding='utf-8') as f:
		d = f.readlines()
		inlist = [i.strip('\n') for i in d]
	new = []

	good_characters = set(string.ascii_letters + "@#$%^&1234567890.,!/_()[]-'\\")

	is_ascii = lambda x: x not in good_characters

	for word in inlist:
		good = True
		for letter in word:
			if is_ascii(letter):
				good = False
				break
		if good:
			new.append(word)

	with open(WORDLIST_FILE, 'a', encoding='utf-8') as f:
		for i in new:
			f.write(i)
			f.write('\n')
	return new

def load_data(WORDLIST_FILE):
	with open(WORDLIST_FILE, 'r', encoding='utf-8') as f:
		k = f.readlines()
		data = [i.strip('\n') for i in k]
		# print('data is %s' % data)
	return data

def construct_image(font_path, word_string, fontsize = False, scale=1):
	if not fontsize:
		lower_bound, upper_bound = 10,50
		fontsize = random.randint(lower_bound, upper_bound) * scale

	img = Image.new('L', (scale*1000,scale*80), color=255)


	d = ImageDraw.Draw(img)
	fnt = ImageFont.truetype(font_path, fontsize)
	d.text((0,0), word_string, fill=0, font=fnt)

	if scale != 1:
		img = img.resize((600,20), Image.LANCZOS)
	img = trim(img)

	return np.array(img)

def gen_pillow_small(process_name, GENERATION_COUNT, PROCESS_COUNT,path, wordlist):
	GENERATION_COUNT = int(GENERATION_COUNT/ PROCESS_COUNT)

	def dump_file(names, labels, i):
		df = pd.DataFrame({'labels':labels, 'names':names})
		df.to_csv(os.path.join(r'C:\Users\Brooks\github\Splitr\data', i+'.csv'), index=False)

	FontPicker = FontChoice(r'C:\Users\Brooks\github\Splitr\fonts')
	WordPicker = WordChoice(wordlist)

	names, labels = [] , []

	total_iter_track = 0

	FILE_EXTENSION = '%s_' + str(process_name) + '.jpg'

	while total_iter_track <= GENERATION_COUNT:
		current_file_name = FILE_EXTENSION % total_iter_track

		current_font = FontPicker.pick_font()
		word_string = make_string(WordPicker.pick_word(random.randint(2,5)))
		L = len(word_string)

		img = construct_image(current_font, word_string)

		img_save_path = os.path.join(path, current_file_name)
		imwrite(img_save_path, img)

		names.append(current_file_name)
		labels.append(word_string)

		total_iter_track += 1

		if total_iter_track % 10000 == 0:
			print('done percent: {}'.format(100*total_iter_track / GENERATION_COUNT))

	dump_file(names, labels, 'training_data' + str(process_name))


def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)


def unique_characters(wordlist, path):
	t = len(wordlist)
	iterator = 0
	holder_set = set()
	for word in wordlist:
		iterator +=1
		holder_set = holder_set | set(word)

		if iterator % 10000 == 0:
			print('percent completed: %s' % (100 * iterator / t))

	new_string = ''.join(str(i) for i in holder_set)
	print('new string generated: ')
	print(new_string)
	print('total len of string: %s' % len(new_string))

	with open(path, 'w') as f:
		f.write(new_string)

def __del_data(files, fpath):
	files_copy = files[:]
	redo_files = []

	while True:
		try:
			if len(files) == 0:
				if len(redo_files) == 0:
					return False
				else:
					files = redo_files[:]
					redo_files = []

			for f in files:
				file_path =os.path.join(fpath, f)
				files_copy.remove(f)
				os.remove(file_path)
			files = files_copy[:]
		except Exception as e:
			print('deletion exception %s'% e)
			files = files_copy[:]
			redo_files.append(f)
			sleep(.01)

def delete_old_data(path=False):
	def chunks(l, n):
		ret = [l[i:i+n] for i in range(0, len(l), n)]
		return ret

	if not path:
		from build_training_data import OCR_DATA_PATH as path

	num_process = 10
	files  = os.listdir(path)
	chunk_size = int(len(files) / num_process)

	files = chunks(files,chunk_size)

	for i in files:
		print('here')
		p = Process(target=__del_data, args=(i, path))
		p.start()

def merge_csv(path):
	files = os.listdir(path)
	csvs = [i for i in files if 'training_data' in i]
	frames = [pd.read_csv(os.path.join(path,fname)) for fname in csvs]
	stacked = pd.concat(frames)
	stacked.to_csv(os.path.join(path, 'training_data.csv'), index=False)

# config
WORDLIST_FILE = r'data\trainwords.txt'
OCR_DATA_PATH = r'C:\Users\Brooks\Desktop\OCR_data'	# where to dump the generated images
# OCR_DATA_PATH = r'C:\Users\Brooks\Desktop\test'
UNIQUE_CHARS_PATH = r'data\unique_characters.txt'
gen_unique= False							# create a file of unique characters that are being trained
crop_dataset = False
GENERATION_COUNT = 3e6
PROCESS_COUNT = 6

# 318628
if __name__ == '__main__':
	# ascii_only(r'C:\Users\Brooks\github\Splitr\data\trainwords_old.txt')
	words = load_data(WORDLIST_FILE)
	delete_old_data(r'C:\Users\Brooks\Desktop\test')

	# t= time.time()
	# for i in range(PROCESS_COUNT):
	# 	p = Process(target=gen_pillow_small, args=(i,GENERATION_COUNT,PROCESS_COUNT,OCR_DATA_PATH,words))
	# 	p.start()
	# 	print(i)

	# merge_csv(r'C:\Users\Brooks\github\Splitr\data')

	# print('total runtime: %s' %(time.time()-t))
