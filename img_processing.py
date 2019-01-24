import cv2
from lanes import draw_lanes, output_lanes_to_screen
from img_functions import *


# window frame
bbox = {'top': 150, 'left': 100, 'width': 400, 'height': 150}
# car frame
car_poly_vertices = [(330, 300), (540, 300), (500, 260), (370, 260), (330, 300)]
# vertices to subtract from bottom of frame
lower_subtraction_vertices = [[300, 300], [500, 300], [450, 220], [350, 220]]
# region of interest mask vertices
roi_vertices = np.array([[10, 300], [120, 220], [170, 120], [630, 120], [680, 220], [790, 300]])

# color filter parameters in hsv -- min white, max gray
white_threshold = 100
gray_threshold = 40

# edge detection
low_threshold = 200
high_threshold = 300

# Gaussian blur
kernel_size = 5

# Hough line detection
min_length = 20
max_gap = 15

def process_img(original_image):
	# processed_img = filter_colors(original_image, white_threshold, gray_threshold)
	# processed_img = get_gray(original_image)
	processed_img = get_edges(original_image, low_threshold, high_threshold)
	processed_img = roi(processed_img, roi_vertices)
	# processed_img = subtract_bottom(processed_img)
	processed_img = blur_screen(processed_img, kernel_size)

	lines = get_lines(processed_img, min_length, max_gap)
	output_lanes_to_screen(processed_img, lines)
	
	processed_img = draw_car_poly(processed_img, car_poly_vertices)
	return processed_img
