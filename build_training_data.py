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

def generate_images(wordlist, PATH):
	fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC]
	fontscale = 2
	font_color = 0
	linetype = 2


	for i in range(len(fonts)):
		font = fonts[i]
		img = np.ones((80,500), np.uint8) * 255
		a = cv2.putText(img,'Hello World!',
		(10,60),
		font,
		fontscale,
		font_color,
		linetype)

		cv2.imwrite(os.path.join(PATH, str(i) + '.jpg'), a)
def gen_pillow(wordlist, path):
	fp = r'C:\Users\Brooks\github\Splitr\fonts'
	fonts = os.listdir(fp)
	L = len(fonts)
	track_dict = {}

	f = 0
	try:
		for i in range(len(wordlist)):
			word = wordlist[i]

			img = Image.new('L', (500,80), color=255)
			d = ImageDraw.Draw(img)
			fnt = ImageFont.truetype(os.path.join(fp, fonts[f]), 70)
			d.text((10,-10), word, fill=0, font=fnt)
			img.save(os.path.join(path, str(i)+ '.jpg'))

			track_dict[i] = word
			f += 1
			if f >= L:
				f =0
	except ValueError as e:
		print(e)
		with open('training_data.json', 'w') as f:
			json.dump(track_dict, f)

if __name__ == '__main__':
	WORDLIST_FILE = 'trainwords.txt'
	PATH = r'D:\OCR'
	words = load_data(WORDLIST_FILE)
	# generate_images(words, PATH)
	gen_pillow(words, PATH)
