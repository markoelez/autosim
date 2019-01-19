import numpy as np
import cv2


def get_gray(screen):
	return cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

def draw_car_rect(screen):
	new_screen = cv2.rectangle(screen, (300, 160), (570, 360), 255, 2)
	return new_screen

def draw_car_poly(screen, vertices):
	poly = np.array(vertices, np.int32)
	draw_poly = cv2.polylines(screen, [poly], True, 255, 2)
	fill_in_poly = cv2.fillPoly(screen, [poly], 0)
	return draw_poly

def subtract_bottom(screen, vertices):
	poly = np.array(vertices, np.int32)
	fill_in_poly = cv2.fillPoly(screen, [poly], 0)
	return fill_in_poly

def roi(screen, vertices):
	mask = np.zeros_like(screen)
	cv2.fillPoly(mask, [vertices], 255)
	masked = cv2.bitwise_and(screen, mask)
	return masked

def blur_screen(screen, kernel_size):
	return cv2.GaussianBlur(screen, (kernel_size, kernel_size), 0)

def get_edges(screen, threshold1, threshold2):
	return cv2.Canny(screen, threshold1, threshold2)

def get_lines(screen_edges, min_length, max_gap):
	return cv2.HoughLinesP(screen_edges, 1, np.pi/180, 180, min_length, max_gap)
	
def filter_colors(screen, white_threshold, gray_threshold):

	hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
	lower_white = np.array([0, 0, white_threshold])
	upper_white = np.array([0, 0, 255])
	white_mask = cv2.inRange(hsv, lower_white, upper_white)
	white_screen = cv2.bitwise_and(screen, screen, mask=white_mask)

	lower_gray = np.array([0, 0, 0])
	upper_gray = np.array([0, 0, gray_threshold])
	gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
	gray_screen = cv2.bitwise_and(screen, screen, mask=gray_mask)

	filtered_screen = cv2.addWeighted(white_screen, 1., gray_screen, 1., 0.)

	return filtered_screen
