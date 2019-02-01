import time
import os
import cv2
from threading import Thread
import numpy as np
from mss import mss
from PIL import Image
from direct_keys import PressKey, ReleaseKey, W, A, S, D, KeyDown, KeyUp
from img_processing import process_img, bbox, scale_factor
from img_functions import get_gray
from get_keys import key_check, listen_keys


def process_feed():
	sct = mss()
	last_time = time.time()
	while 1:

		screen = np.array(sct.grab(bbox))
		new_screen = process_img(screen)

		print('Loop took {0:.4f} seconds with {1:.4f} fps'.format(time.time() - last_time, (time.time() - last_time)**-1))
		last_time = time.time()

		cv2.imshow('filtered_screen',  new_screen)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

def key_test():
	listen_keys()
	while 1:
		key = key_check()
		output.append(keys_output(key))
		print(keys_output(key))

		# out = np.array(output)
		# np.save(file_name, out)
		# np.savetxt('test.csv', out)

file_name = 'training_data.npy'

if os.path.isfile(file_name):
	print('file exists, loading previous data')
	training_data = list(np.load(file_name))
else:
	print('file does not exist, writing new file')
	training_data = []

def keys_output(keys):
	output = [0, 0, 0]
	if 'A' in keys:
		output[0] = 1
	elif 'D' in keys:
		output[2] = 1
	else:
		output[1] = 1
	return output

output = []

width = int(bbox['width']*scale_factor/100)
height = int(bbox['height']*scale_factor/100)
dim = (width, height)

def main():

	sct = mss()

	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)

	last_time = time.time()

	t = Thread(target=listen_keys)
	t.daemon = True
	t.start()

	while 1:
		screen = np.array(sct.grab(bbox))
		screen = get_gray(screen)
		screen = cv2.resize(screen, dim, interpolation=cv2.INTER_AREA)

		print('Loop took {0:.4f} seconds with {1:.4f} fps'.format(time.time() - last_time, (time.time() - last_time)**-1))
		last_time = time.time()
		key = key_check()
		output = keys_output(key)
		training_data.append([screen, output])

		if len(training_data) % 800 == 0:
			print(len(training_data))
			np.save(file_name, training_data)
		
# key_test()
# process_feed()
main()
