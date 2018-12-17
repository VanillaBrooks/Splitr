from cv2 import imwrite
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import string
import pandas as pd
import random
import torchvision

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
			if self.len == 0:
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

def gen_pillow(wordlist, path, GENERATION_COUNT):

	def dump_file(names, labels, i):
		df = pd.DataFrame({'labels':labels, 'names':names})
		df.to_csv(os.path.join(r'C:\Users\Brooks\github\Splitr\data', i+'.csv'), index=False)

	FontPicker = FontChoice(r'C:\Users\Brooks\github\Splitr\fonts')
	WordPicker = WordChoice(wordlist)

	names, labels = [] , []

	total_iter_track = 0
	scale = 1

	while total_iter_track <= GENERATION_COUNT:
		current_font = FontPicker.pick_font()
		word_string = make_string(WordPicker.pick_word(random.randint(2,5)))
		L = len(word_string)

		minx, maxx = int(scale*500 /20), int(scale* 500 *10/20)
		miny, maxy = int(scale*80 /10), int(scale* 80 *14 /20)


		if L < 15:
			lower_bound = 30
			upper_bound = 40
			miny, maxy = int(scale*80 /10), int(scale* 80 *8 /20)
		elif L < 20:
			lower_bound = 20
			upper_bound = 30
			minx, maxx = int(scale*500 /20), int(scale* 500 *6/20)
			miny, maxy = int(scale*80 /10), int(scale* 80 *11 /20)
		elif L < 30:
			lower_bound = 10
			upper_bound = 20
		elif L > 50:
			minx, maxx = int(scale*500 /20), int(scale* 500 *2/20)
			miny, maxy = int(scale*80 /10), int(scale* 80 *11 /20)
			lower_bound = 10
			upper_bound = 13
		else:
			lower_bound = 10
			upper_bound = 15
			minx, maxx = int(scale*500 /20), int(scale* 500 *8/20)


		fontsize = random.randint(lower_bound,upper_bound) * scale

		img = Image.new('L', (500*scale,80*scale), color=255)

		x_offset = random.randint(minx, maxx)
		y_offset = random.randint(miny, maxy)

		d = ImageDraw.Draw(img)
		fnt = ImageFont.truetype(current_font, fontsize)
		d.text((x_offset,y_offset), word_string, fill=0, font=fnt)

		if scale != 1:
			img = img.resize((500,80), Image.LANCZOS)

		img_save_path = os.path.join(path, str(total_iter_track) + '.jpg')
		imwrite(img_save_path, np.array(img))

		names.append(str(total_iter_track) +'.jpg')
		labels.append(word_string)

		total_iter_track += 1

		if total_iter_track % 10000 == 0:
			print('done percent: {}'.format(100*total_iter_track / GENERATION_COUNT))

	dump_file(names, labels, 'training_data')

def gen_pillow_small(wordlist, path, GENERATION_COUNT):

	def dump_file(names, labels, i):
		df = pd.DataFrame({'labels':labels, 'names':names})
		df.to_csv(os.path.join(r'C:\Users\Brooks\github\Splitr\data', i+'.csv'), index=False)

	FontPicker = FontChoice(r'C:\Users\Brooks\github\Splitr\fonts')
	WordPicker = WordChoice(wordlist)

	names, labels = [] , []

	total_iter_track = 0
	scale = 1

	while total_iter_track <= GENERATION_COUNT:
		current_font = FontPicker.pick_font()
		word_string = make_string(WordPicker.pick_word(random.randint(2,5)))
		L = len(word_string)

		xpos, ypos = 0,0
		lower_bound, upper_bound = 10,50
		fontsize = random.randint(lower_bound, upper_bound) * scale

		img = Image.new('L', (scale*1000,scale*80), color=255)


		d = ImageDraw.Draw(img)
		fnt = ImageFont.truetype(current_font, fontsize)
		d.text((xpos,ypos), word_string, fill=0, font=fnt)

		if scale != 1:
			img = img.resize((600,20), Image.LANCZOS)
		img = trim(img)

		img_save_path = os.path.join(path, str(total_iter_track) + '.jpg')
		imwrite(img_save_path, np.array(img))

		names.append(str(total_iter_track) +'.jpg')
		labels.append(word_string)

		total_iter_track += 1

		if total_iter_track % 10000 == 0:
			print('done percent: {}'.format(100*total_iter_track / GENERATION_COUNT))

	dump_file(names, labels, 'training_data')

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

def delete_old_files(fpath):
	files = os.listdir(fpath)
	print(files)

	for f in files:
		file_path =os.path.join(fpath, f)
		os.remove(file_path)

# config
WORDLIST_FILE = r'data\trainwords.txt'
OCR_DATA_PATH = r'C:\Users\Brooks\Desktop\OCR_data'	# where to dump the generated images
# OCR_DATA_PATH = r'C:\Users\Brooks\Desktop\test'
UNIQUE_CHARS_PATH = r'data\unique_characters.txt'
delete_old_data = False						# remove older files in folder. faster than windows delete
gen_unique= False							# create a file of unique characters that are being trained
crop_dataset = False
GENERATION_COUNT = 15e6

# 318628
if __name__ == '__main__':
	# ascii_only(r'C:\Users\Brooks\github\Splitr\data\trainwords_old.txt')
	if delete_old_data:
		delete_old_files(OCR_DATA_PATH)
	words = load_data(WORDLIST_FILE)
	# if crop_dataset:
	# 	words = words[:crop_dataset]
	# gen_pillow(words, OCR_DATA_PATH, GENERATION_COUNT)
	# if gen_unique:
	# 	unique_characters(words,UNIQUE_CHARS_PATH)

	gen_pillow_small(words,OCR_DATA_PATH, GENERATION_COUNT)
