'''
*****************************************************************************************
*
*        		===============================================
*           		Pharma Bot (PB) Theme (eYRC 2022-23)
*        		===============================================
*
*  This script is to implement Task 1A of Pharma Bot (PB) Theme (eYRC 2022-23).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*************************************PB_9999_Task_1A.zip****************************************************
'''

# Team ID:			[ PB_2302 ]
# Author List:		[ Atharv Hardikar, Prachit Gupta, Sahil Sudhakar, Tanmay Nitul Jain ]
# Filename:			task_1a.py
# Functions:		detect_traffic_signals, detect_horizontal_roads_under_construction, detect_vertical_roads_under_construction,
#					detect_medicine_packages, detect_arena_parameters
# 					[ Comma separated list of functions in this file ]


####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the three available ##
## modules for this task (numpy, opencv)                    ##
##############################################################
import cv2
import numpy as np


##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
def detect_orange(image):
	boundaries = [([0, 50,50], [10, 255, 255])] #orange
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output_orange = cv2.bitwise_and(image, image, mask = mask)
		return output_orange

def detect_pink(image):
	boundaries = [([175, 0,250], [185, 5, 255])]  #pink
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output_pink = cv2.bitwise_and(image, image, mask = mask)
		return output_pink

def detect_red(image):
	boundaries = [([0, 0, 250], [5, 5, 255])]   # red
	image_og = image
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output_red = cv2.bitwise_and(image, image, mask = mask)
		return output_red
		

def detect_green(image):
	boundaries = [([0, 250,0], [5, 255, 5])] #green
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output_green = cv2.bitwise_and(image, image, mask = mask)
		return output_green

def detect_blue_node(image):
	boundaries = [([254, 0, 0], [255, 1, 1])] # blue node on road
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output_blue_node = cv2.bitwise_and(image, image, mask = mask)
		return output_blue_node

def detect_sky_blue(image):
	boundaries = [([250, 250, 0], [255, 255, 10])] # sky blue
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(image, lower, upper)
		output_sky_blue = cv2.bitwise_and(image, image, mask = mask)
		return output_sky_blue

def detect_black(image):
	img = np.copy(image)	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	(thresh1, blackAndWhiteImage_B) = cv2.threshold(cv2.cvtColor(detect_blue_node(image),cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
	(thresh2, blackAndWhiteImage_G) = cv2.threshold(cv2.cvtColor(detect_green(image),cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
	(thresh3, blackAndWhiteImage_P) = cv2.threshold(cv2.cvtColor(detect_pink(image),cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
	(thresh4, blackAndWhiteImage_O) = cv2.threshold(cv2.cvtColor(detect_orange(image),cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
	(thresh5, blackAndWhiteImage_R) = cv2.threshold(cv2.cvtColor(detect_red(image), cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
	(thresh6, blackAndWhiteImage_SB) = cv2.threshold(cv2.cvtColor(detect_sky_blue(image),cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

	img = blackAndWhiteImage + blackAndWhiteImage_G + blackAndWhiteImage_P + blackAndWhiteImage_B + blackAndWhiteImage_O + blackAndWhiteImage_R + blackAndWhiteImage_SB
	return img
	'''
	imgGry = img

	ret, thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
	contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
		cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
		x = approx.ravel()[0]
		y = approx.ravel()[1] - 5
		
		if len(approx) == 4 :
			x, y , w, h = cv2.boundingRect(approx)
			aspectRatio = float(w)/h
			print(aspectRatio)
			if aspectRatio >= 0.95 and aspectRatio < 1.05:
				cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

			else:
				cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
	#cv2.imshow("shpa",img)

	#cv2.waitKey(0)
	'''

def get_med_col(image, pos):
	col = image[pos[1]][pos[0]]
	col_str = ""
	if col[0] == 0 and col[1] == 255 and col[2] == 0 :
		col_str = "Green"
	elif col[0] == 0 and col[1] == 127 and col[2] == 255 :
		col_str = "Orange"
	elif col[0] == 180 and col[1] == 0 and col[2] == 255 :
		col_str = "Pink"
	elif col[0] == 255 and col[1] == 255 and col[2] == 0 :
		col_str = "Skyblue"
	return col_str

def detect_circle(image):
	
	circle_Array = []
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Blur using 3 * 3 kernel.
	gray_blurred = cv2.blur(gray, (3, 3))
	
	# Apply Hough transform on the blurred image.
	detected_circles = cv2.HoughCircles(gray_blurred, 
					cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
				param2 = 30, minRadius = 10, maxRadius = 15)
	
	# Draw circles that are detected.
	if detected_circles is not None:
	
		# Convert the circle parameters a, b and r to integers.
		detected_circles = np.uint16(np.around(detected_circles))
	
		for pt in detected_circles[0, :]:
			a, b, r = pt[0], pt[1], pt[2]
	
			# Draw the circumference of the circle.
			#cv2.circle(image, (a, b), r, (0, 255, 0), 2)
			
			# Draw a small circle (of radius 1) to show the center.
			#cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

			circle_Array.append([get_shop_no_from_loc((a,b)),get_med_col(image,(a,b)),"Circle",[a,b]])
		return circle_Array


def detect_triangle(image):                        
	triangleArray = []
	img = np.copy(image)
	imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
	contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
		if len(approx) == 3:
			cv2.drawContours(img, [approx], 0, (0, 0, 0), 2)
			x = approx.ravel()[0]
			y = approx.ravel()[1] - 5	
			M = cv2.moments(contour)
			if M['m00'] != 0.0:
				a = int(M['m10']/M['m00'])
				b = int(M['m01']/M['m00'])
			
			triangleArray.append([get_shop_no_from_loc((a,b)),get_med_col(image,(a,b)),"Triangle",[a,b]])
			#cv2.putText( img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )

	return triangleArray

def detect_square(image):  
	img = np.copy(image)
	squareArray = []
	img = detect_green(image) + detect_orange(image) + detect_pink(image) + detect_sky_blue(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# setting threshold of gray image
	_, threshold = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
	
	# using a findContours() function
	contours, _ = cv2.findContours(
		threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	i = 0
	
	# list for storing names of shapes
	for contour in contours:
	
		'''
		# here we are ignoring first counter because 
		# findcontour function detects whole image as shape
		if i == 0:
			i = 1
			continue
		'''

		# cv2.approxPloyDP() function to approximate the shape
		approx = cv2.approxPolyDP(
			contour, 0.01 * cv2.arcLength(contour, True), True)
		
		# using drawContours() function
		cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
	
		# finding center point of shape
		M = cv2.moments(contour)
		if M['m00'] != 0.0:
			a = int(M['m10']/M['m00'])
			b = int(M['m01']/M['m00'])
	
		if len(approx) == 4 and cv2.contourArea(contour)>200 :
			#print(a,b)
			#cv2.putText(img, 'Quadrilateral', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 2)
			squareArray.append([get_shop_no_from_loc((a,b)),get_med_col(image,(a,b)),"Square",[a,b]])

	return squareArray


def node_location(pos):
	y = pos[1]/100
	x = chr(int(64+ pos[0]/100))
	a = str(x)+str(y)
	a = a[:2]
	return a

def get_h_section(pos):
	y= pos[1]/100
	x = pos[0]/100
	x1 = int(x)
	x2 = x1+1
	locL = node_location((x1*100,y*100))
	locR = node_location((x2*100,y*100))
	return locL, locR

def get_v_section(pos):
	x= pos[0]/100
	y = pos[1]/100
	y1 = int(y)
	y2 = y1+1
	locU = node_location((x*100,y1*100))
	locD = node_location((x*100,y2*100))
	return locU, locD

def get_shop_no_from_loc(pos):
	x = int(pos[0]/100)
	shop_str = str("Shop_"+str(x))
	return shop_str[:6]

def alphabetical_colors(list):
	
	tmplist = []
	tmpalph = []
	for element in list:
		tmpalph.append(element[1][0])
	tmpalph.sort()
	alph = tmpalph
	for i in range(0,len(list)):
		for j in range(0,len(list)):
			if list[j][1][0] == alph[i] :
				tmplist.append(list[j])
	return tmplist
	

def arrangeList(image):
	medicine_packages = []

	circleArray = alphabetical_colors( detect_circle(image) )
	triangleArray = alphabetical_colors( detect_triangle(image) )
	squareArray = alphabetical_colors( detect_square(image) )

	c = int(circleArray[0][0][5])
	t = int(triangleArray[0][0][5]) 
	s = int(squareArray[0][0][5])

	if c>t and c>s:
		if t>s:
			for i in range(0, len(squareArray)):
				medicine_packages.append(squareArray[i])
			for i in range(0, len(triangleArray)):
				medicine_packages.append(triangleArray[i])
		
		else:
			for i in range(0, len(triangleArray)):
				medicine_packages.append(triangleArray[i])
			for i in range(0, len(squareArray)):
				medicine_packages.append(squareArray[i])

		for i in range(0, len(circleArray)):
				medicine_packages.append(circleArray[i])

	elif t>c and t>s:
		if c>s:
			for i in range(0, len(squareArray)):
				medicine_packages.append(squareArray[i])
			for i in range(0, len(circleArray)):
				medicine_packages.append(circleArray[i])
		else:
			for i in range(0, len(circleArray)):
				medicine_packages.append(circleArray[i])
			for i in range(0, len(squareArray)):
				medicine_packages.append(squareArray[i])
		for i in range(0, len(triangleArray)):
			medicine_packages.append(triangleArray[i])

	else:
		if c>t:
			for i in range(0, len(triangleArray)):
				medicine_packages.append(triangleArray[i])
			for i in range(0, len(circleArray)):
				medicine_packages.append(circleArray[i])
		else:
			for i in range(0, len(circleArray)):
				medicine_packages.append(circleArray[i])
			for i in range(0, len(triangleArray)):
				medicine_packages.append(triangleArray[i])
		for i in range(0, len(squareArray)):
			medicine_packages.append(squareArray[i])

	return medicine_packages


##############################################################


def detect_traffic_signals(maze_image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list of
	nodes in which traffic signals are present in the image

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`traffic_signals` : [ list ]
			list containing nodes in which traffic signals are present
	
	Example call:
	---
	traffic_signals = detect_traffic_signals(maze_image)
	"""
	traffic_signals = []

	##############	ADD YOUR CODE HERE	##############
	C_X = []
	C_Y = []

	red = detect_red(maze_image)
	image = red

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#blur = cv2.GaussianBlur(gray, (5, 5),cv2.BORDER_DEFAULT)

	ret, thresh = cv2.threshold(gray, 5, 255,cv2.THRESH_BINARY_INV)
	#thresh = cv2.bitwise_not(thresh)
	contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	blank = np.zeros(thresh.shape[:2],dtype='uint8')

	cv2.drawContours(blank, contours, -1,(255, 0, 0), 1)

	#cv2.imwrite("Contours.png", blank)
	for i in contours:
		#c=max(contours, key = cv2.contourArea)
		M = cv2.moments(i)
		((x,y), radius) = cv2.minEnclosingCircle(i)
		if M['m00'] != 0 and radius < 30:
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
			cv2.circle(image, (cx, cy), 7, (0, 0, 255), -1)
			cv2.putText(image, "center", (cx - 20, cy - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
			#print(f"x: {cx} y: {cy} r: {radius}")
			traffic_signals.append(node_location((cx, cy)))

	##################################################

	return traffic_signals


def detect_horizontal_roads_under_construction(maze_image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list
	containing the missing horizontal links

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`horizontal_roads_under_construction` : [ list ]
			list containing missing horizontal links
	
	Example call:
	---
	horizontal_roads_under_construction = detect_horizontal_roads_under_construction(maze_image)
	"""
	horizontal_roads_under_construction = []

	##############	ADD YOUR CODE HERE	##############

	dt = 100
	black = detect_black(maze_image)

	for i in range(150, 750, dt):
		for j in range(100,800, dt):
			if(black[j][i]) == 255:
				L,R = get_h_section((i,j))
				horizontal_roads_under_construction.append(str(L+"-"+R))
			
	##################################################

	return horizontal_roads_under_construction


def detect_vertical_roads_under_construction(maze_image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list
	containing the missing vertical links

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`vertical_roads_under_construction` : [ list ]
			list containing missing vertical links
	
	Example call:
	---
	vertical_roads_under_construction = detect_vertical_roads_under_construction(maze_image)
	"""
	vertical_roads_under_construction = []

	##############	ADD YOUR CODE HERE	##############

	dt = 100
	black = detect_black(maze_image)

	for i in range(100, 800, dt):
		for j in range(150, 750, dt):
			if(black[j][i]) == 255:
				U,D = get_v_section((i,j))
				vertical_roads_under_construction.append(str(U+"-"+D))
	
	##################################################

	return vertical_roads_under_construction


def detect_medicine_packages(maze_image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a nested list of
	details of the medicine packages placed in different shops

	** Please note that the shop packages should be sorted in the ASCENDING order of shop numbers 
	   as well as in the alphabetical order of colors.
	   For example, the list should first have the packages of shop_1 listed. 
	   For the shop_1 packages, the packages should be sorted in the alphabetical order of color ie Green, Orange, Pink and Skyblue.

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`medicine_packages` : [ list ]
			nested list containing details of the medicine packages present.
			Each element of this list will contain 
			- Shop number as Shop_n
			- Color of the package as a string
			- Shape of the package as a string
			- Centroid co-ordinates of the package
	Example call:
	---
	medicine_packages = detect_medicine_packages(maze_image)
	"""
	medicine_packages = []

	##############	ADD YOUR CODE HERE	##############

	medicine_packages = arrangeList(maze_image)

	
	##################################################

	return medicine_packages


def detect_arena_parameters(maze_image):
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a dictionary
	containing the details of the different arena parameters in that image

	The arena parameters are of four categories:
	i) traffic_signals : list of nodes having a traffic signal
	ii) horizontal_roads_under_construction : list of missing horizontal links
	iii) vertical_roads_under_construction : list of missing vertical links
	iv) medicine_packages : list containing details of medicine packages

	These four categories constitute the four keys of the dictionary

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`arena_parameters` : { dictionary }
			dictionary containing details of the arena parameters
	
	Example call:
	---
	arena_parameters = detect_arena_parameters(maze_image)
	"""
	arena_parameters = {}

	##############	ADD YOUR CODE HERE	##############

	traffic_signals = detect_traffic_signals(maze_image)
	arena_parameters['traffic_signals']=traffic_signals

	hruc = detect_horizontal_roads_under_construction(maze_image)
	arena_parameters['horizontal_roads_under_construction'] = hruc

	vruc = detect_vertical_roads_under_construction(maze_image)
	arena_parameters['vertical_roads_under_construction'] = vruc

	medicine_packages_present = detect_medicine_packages(maze_image)
	arena_parameters['medicine_packages_present'] = medicine_packages_present
	##################################################

	return arena_parameters


######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########


if __name__ == "__main__":

	# path directory of images in test_images folder
	img_dir_path = "./public_test_images/"

	# path to 'maze_0.png' image file
	file_num = 0
	img_file_path = img_dir_path + 'maze_' + str(file_num) + '.png'

	# read image using opencv
	maze_image = cv2.imread(img_file_path)

	print('\n============================================')
	print('\nFor maze_' + str(file_num) + '.png')

	# detect and print the arena parameters from the image
	arena_parameters = detect_arena_parameters(maze_image)

	print("Arena Prameters: ", arena_parameters)

	# display the maze image
	cv2.imshow("image", maze_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	choice = input('\nDo you want to run your script on all test images ? => "y" or "n": ')

	if choice == 'y':

		for file_num in range(1, 15):
			# path to maze image file
			img_file_path = img_dir_path + 'maze_' + str(file_num) + '.png'

			# read image using opencv
			maze_image = cv2.imread(img_file_path)

			print('\n============================================')
			print('\nFor maze_' + str(file_num) + '.png')

			# detect and print the arena parameters from the image
			arena_parameters = detect_arena_parameters(maze_image)

			print("Arena Parameter: ", arena_parameters)

			# display the test image
			cv2.imshow("image", maze_image)
			cv2.waitKey(2000)
			cv2.destroyAllWindows()