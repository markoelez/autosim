import time
import os
import cv2
from threading import Thread
import numpy as np
from mss import mss
from direct_keys import PressKey, ReleaseKey, W, A, S, D, KeyDown, KeyUp
from img_processing import process_img, bbox, scale_factor
from img_functions import get_gray
from get_keys import key_check, listen_keys
from models import alexnet
from keras.models import load_model


WIDTH = int(bbox['width']*scale_factor/100)
HEIGHT = int(bbox['height']*scale_factor/100)
dim = (WIDTH, HEIGHT)

LR = 1e-5
EPOCHS = 50
BATCH_SIZE = 5
MODEL_NAME = 'pyAutoSim-{}-{}-{}-epochs.model'.format(LR, 'Alexnet', EPOCHS)

def straight():
	PressKey(W)
	ReleaseKey(A)
	ReleaseKey(D)
	ReleaseKey(S)

def left():
	PressKey(A)
	PressKey(W)
	ReleaseKey(D)
	ReleaseKey(S)

def right():
	PressKey(D)
	PressKey(W)
	ReleaseKey(A)
	ReleaseKey(S)

# model = alexnet(image_size=dim, learning_rate=LR)
model = load_model(MODEL_NAME)

def main():

	sct = mss()

	for i in list(range(4))[::-1]:
		print(i+1)
		time.sleep(1)
	
	last_time = time.time()

	t = Thread(target=listen_keys)
	t.daemon = True
	t.start()

	paused = False

	while 1:
		if not paused:
			screen = np.array(sct.grab(bbox))
			screen = get_gray(screen)
			screen = cv2.resize(screen, dim, interpolation=cv2.INTER_AREA)

			print('Loop took {0:.4f} seconds with {1:.4f} fps'.format(time.time() - last_time, (time.time() - last_time)**-1))
			last_time = time.time()
			
			prediction = model.predict([screen.reshape(-1, WIDTH, HEIGHT, 1)])[0]
			moves = list(np.around(prediction))
			print(moves)

			if moves == [0, 0, 1]:
				right()
			elif moves == [1, 0, 0]:
				left()
			elif moves == [0, 1, 0]:
				straight()

		keys = key_check()
		print(keys)
		if 'T' in keys:
			if paused:
				paused = False
				time.sleep(1)
			else:
				paused = True
				ReleaseKey(A)
				ReleaseKey(W)
				ReleaseKey(D)
				time.sleep(1)


main()
