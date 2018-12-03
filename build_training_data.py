import cv2
import torch
import pprint
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json
import string
import pandas as pd


def ascii_only(filename):
	with open(filename, 'r', encoding='utf-8') as f:
		d = f.readlines()
		inlist = [i.strip('\n') for i in d]
	new = []

	good_characters = set(string.ascii_letters + " .,;'")

	is_ascii = lambda x: x in good_characters

	for word in inlist:
		good = True
		for letter in word:
			if is_ascii(letter):
				continue
			else:
				good = False
				break
		if good:
			new.append(word)

	with open(WORDLIST_FILE, 'a', encoding='utf-8') as f:
		for i in new:
			f.write(i)
			f.write('\n')

def load_data(WORDLIST_FILE):
	with open(WORDLIST_FILE, 'r', encoding='utf-8') as f:
		k = f.readlines()
		data = [i.strip('\n') for i in k]
		# print('data is %s' % data)
	return data

def gen_pillow(wordlist, path):
	fp = r'C:\Users\Brooks\github\Splitr\fonts'
	fonts = os.listdir(fp)
	L = len(fonts)
	# print('sorting')
	wordlist.sort(key=len, reverse=True)
	# wordlist = wordlist[:]

	names, labels = [] , []

	# [print(i, len(i)) for i in wordlist[:100]]

	total_iter_track = 0
	scale = 5
	try:
		for k in range(len(fonts)):
			current_font = fonts[k]

			for i in range(len(wordlist)):
				word = wordlist[i]

				if len(word) > 14:
					fontsize = 35*scale
				else:
					fontsize = 50 *scale

				img = Image.new('L', (500*scale,80*scale), color=255)
				# img = Image.new('RGB', (500*scale, 80*scale), color=(255,255,255))
				d = ImageDraw.Draw(img)
				fnt = ImageFont.truetype(os.path.join(fp, current_font), fontsize)
				d.text((10,-10), word, fill=0, font=fnt)

				if scale != 1:
					img = img.resize((500,80), Image.LANCZOS)

				img_save_path = os.path.join(path, str(total_iter_track) + '.jpg')
				# img.save(os.path.join(path, str(total_iter_track) + '.jpg'))
				cv2.imwrite(img_save_path, np.array(img))

				names.append(str(total_iter_track) +'.jpg')
				labels.append(word)

				total_iter_track += 1

				if total_iter_track % 100000 == 0:
					print('done percent: %s' % (100*(1/len(fonts))*(1/len(wordlist))*total_iter_track))

	finally:
		dump_file(names, labels, 'final')

def dump_file(names, labels, i):
	df = pd.DataFrame({'labels':labels, 'names':names})
	df.to_csv(i+'.csv', index=False)

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

	for f in files:
		file_path =os.path.join(fpath, f)
		os.remove(file_path)



# config
WORDLIST_FILE = r'data\trainwords.txt'
OCR_DATA_PATH = r'C:\Users\Brooks\Desktop\OCR_data'	# where to dump the generated images
UNIQUE_CHARS_PATH = r'data\unique_characters.txt'
delete_old_data = False						# remove older files in folder. faster than windows delete
gen_unique= False							# create a file of unique characters that are being trained

if __name__ == '__main__':
	# ascii_only('trainwords_old.txt')
	if delete_old_data:
		delete_old_files(OCR_DATA_PATH)
	words = load_data(WORDLIST_FILE)[:10000]
	# gen_pillow(words, OCR_DATA_PATH)
	if gen_unique:
		unique_characters(words,UNIQUE_CHARS_PATH)
