import cv2
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean


def output_lines_to_screen(screen, lines):
	try:
		for coords in lines:
			coords = coords[0]
			try:
				cv2.line(screen, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
			except Exception as e:
				print(str(e))
	except Exception as e:
		pass

# from https://github.com/georgesung/road_lane_line_detection/blob/master/lane_lines.py
def draw_lines(img, lines):

	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - 0.4)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Draw the right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), 255, 12)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), 255, 12)

# from https://github.com/Sentdex/pygta5/blob/master/Tutorial%20Codes/Part%201-7/part-6-lane-finder.py
def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

	# if this fails, go with some default line
	try:

		# finds the maximum y value for a lane marker 
		# (since we cannot assume the horizon will always be at the same point.)

		ys = []  
		for i in lines:
			for ii in i:
				ys += [ii[1],ii[3]]
		min_y = min(ys)
		max_y = 600
		new_lines = []
		line_dict = {}

		for idx,i in enumerate(lines):
			for xyxy in i:
				# These four lines:
				# modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
				# Used to calculate the definition of a line, given two sets of coords.
				x_coords = (xyxy[0],xyxy[2])
				y_coords = (xyxy[1],xyxy[3])
				A = vstack([x_coords,ones(len(x_coords))]).T
				m, b = lstsq(A, y_coords)[0]

				# Calculating our new, and improved, xs
				x1 = (min_y-b) / m
				x2 = (max_y-b) / m

				line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
				new_lines.append([int(x1), min_y, int(x2), max_y])

		final_lanes = {}

		for idx in line_dict:
			final_lanes_copy = final_lanes.copy()
			m = line_dict[idx][0]
			b = line_dict[idx][1]
			line = line_dict[idx][2]
			
			if len(final_lanes) == 0:
				final_lanes[m] = [ [m,b,line] ]
				
			else:
				found_copy = False

				for other_ms in final_lanes_copy:

					if not found_copy:
						if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
							if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
								final_lanes[other_ms].append([m,b,line])
								found_copy = True
								break
						else:
							final_lanes[m] = [ [m,b,line] ]

		line_counter = {}

		for lanes in final_lanes:
			line_counter[lanes] = len(final_lanes[lanes])

		top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

		lane1_id = top_lanes[0][0]
		lane2_id = top_lanes[1][0]

		def average_lane(lane_data):
			x1s = []
			y1s = []
			x2s = []
			y2s = []
			for data in lane_data:
				x1s.append(data[2][0])
				y1s.append(data[2][1])
				x2s.append(data[2][2])
				y2s.append(data[2][3])
			return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

		l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
		l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

		return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
	except Exception as e:
		print(str(e))