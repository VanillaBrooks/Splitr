import cv2
import torch
import pprint
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import json

WORDLIST_FILE = 'trainwords.txt'
def unicode_only(filename):
	with open(filename, 'r', encoding='utf-8') as f:
		d = f.readlines()
		inlist = [i.strip('\n') for i in d]
	new = []
	for word in inlist:
		try:
			print(word)	# hella ghetto way to trigger unicode error
			new.append(word)
		except Exception as e:
			print(e)
			pass
	with open(WORDLIST_FILE, 'a', encoding='utf-8') as f:
		for i in new:
			f.write(i)
			f.write('\n')

def load_data(WORDLIST_FILE):
	with open(WORDLIST_FILE, 'r', encoding='utf-8') as f:
		k = f.readlines()
		data = [i.strip('\n') for i in k]
	return data

def gen_pillow(wordlist, path):
	fp = r'C:\Users\Brooks\github\Splitr\fonts'
	fonts = os.listdir(fp)
	L = len(fonts)
	# print('sorting')
	wordlist.sort(key=len, reverse=True)
	# wordlist = wordlist[:]
	track_dict = {}
	# [print(i, len(i)) for i in wordlist[:100]]

	total_iter_track = 0
	try:
		for k in range(len(fonts)):
			current_font = fonts[k]

			for i in range(len(wordlist)):
				word = wordlist[i]

				if len(word) > 14:
					fontsize = 35*3
				else:
					fontsize = 50 *3

				img = Image.new('L', (500*3,80*3), color=255)
				d = ImageDraw.Draw(img)
				fnt = ImageFont.truetype(os.path.join(fp, current_font), fontsize)
				d.text((10,-10), word, fill=0, font=fnt)
				img = img.resize((500,80), Image.BILINEAR)
				img.save(os.path.join(path, str(total_iter_track) + '.jpg'))

				track_dict[total_iter_track] = [current_font, word]
				total_iter_track += 1

				if total_iter_track % 100000 == 0:
					print('done percent: %s' % (100*(1/len(fonts))*(1/len(wordlist))*total_iter_track))

	finally:
		with open('training_data.json', 'w') as f:
			json.dump(track_dict, f)

if __name__ == '__main__':
	WORDLIST_FILE = 'trainwords.txt'
	PATH = r'D:\OCR_data'
	words = load_data(WORDLIST_FILE)
	# generate_images(words, PATH)
	gen_pillow(words, PATH)
