# import required libraries

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import timeit


# Instead of creating just one class for lane, we create two classes for right and left lane to store it's respective
# properties.

# We define a class for the left lane to collect the x and y coordinates, the radius of curvature and the polynomial
# coefficients.


class Left:
    def __init__(self):
        # Was the line found in the previous frame?
        self.found = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = []
        self.top = []

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = []
        self.fit1 = []
        self.fit2 = []
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0


# We define a class for the right lane to collect the x and y coordinates, the radius of curvature and the polynomial
# coefficients.
class Right:
    def __init__(self):
        # Was the line found in the previous frame?
        self.found = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = []
        self.top = []

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = []
        self.fit1 = []
        self.fit2 = []
        self.fitx = None
        self.pts = []


# This function is used to warp the image so that we obtain a bird's eye view of the road ahead. "Src" stores the initial
# coordinates of the polygon in the image. "Dst" contains the coordinates of the polygon into which we would
# like to warp the original image into. The function cv2.getPerspectiveTransform(src,dst) is used to obtain M which is the
# perspective transform matrix. The function cv2.getPerspectiveTransfor(dst,src) is used to obtain Minv which is the
# inverse perspective transform matrix.
def perspective_transform(img):
    offset = 0
    img_size = (img.shape[1], img.shape[0])
    print(offset)
    src = np.float32([[490, 450], [810, 450], [1250, 720], [0, 720]])
    dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


# This function is used to convert the image to different colour spaces in order to extract the required features from the
# image. Colour space is something that defines an image in terms of dfferent colours
def binary_image(img, yuv_thresh=(0, 110), hsv_thresh=(240, 255), sobel_thresh=(40, 100), sobel_kernel=3):
    # Converting an image to YUV color space. The Y component determines the brightness of the color.The U stands for blue - luminance
    # and the V stands for red - luminance
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Applying a threshold to the V channel
    yuv_channel = yuv[:, :, 2]
    yuv_binary = np.zeros_like(yuv_channel)
    yuv_binary[(yuv_channel >= yuv_thresh[0]) & (yuv_channel <= yuv_thresh[1])] = 1

    # converting an image to the LUV colour space and applying threshold to the L channel
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]
    l_binary = np.zeros_like(l_channel)
    l_thresh_min = 225
    l_thresh_max = 255
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # converting the image to the LAB color spae and applying threshold to the B channel
    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_thresh_min = 155
    b_thresh_max = 200
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    # converting to HLS (hue,lightness and saturation) color space. and applying a threshold to the saturation
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    thresh = (100, 255)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    # Converting an image to HSV(hue, saturation and value) color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Applying a threshold to the V channel
    hsv_channel = hsv[:, :, 2]
    hsv_binary = np.zeros_like(hsv_channel)
    hsv_binary[(hsv_channel >= hsv_thresh[0]) & (hsv_channel <= hsv_thresh[1])] = 1

    # Converting an image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculating Sobel operator. Sobel operators are used to determine the edges in a grayscale image.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculating the gradient magnitude
    magnitude = np.sqrt(sobelx ** 2 + 0.01 * sobely ** 2)
    # Rescaling to 8 bit
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    # Applying a threshold
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    # Combining thresholds
    combined_binary = np.zeros_like(sobel_binary)
    combined_binary[(gray == 1) | (yuv_binary == 1) | (l_binary == 1) | (b_binary == 1) | (binary_output == 1) | (
    hsv_binary == 1) | (sobel_binary == 1)] = 1

    return combined_binary


# This function is used to calculate the x & y coordinates of the image points for both left lane and right lane. It also
# makes sure that when no lane points are detected in a particular frame,  the previous detected image points are used.

def find_lane_video(image, lefty, leftx, righty, rightx):
    # Identify all non zero pixels in the image
    x, y = np.nonzero(np.transpose(image))

    # RIGHT LANE

    if Right.found == False:  # Perform blind search for lane lines
        i = 720  # y coordinate of the bottom pane of the initial window
        j = 630  # y coorfinate of the top pane of window window.
        while j >= 0:
            # Histogram in which bitwise addition is done along a horizontal strip for each x coordinate
            histogram = np.sum(image[j:i, :], axis=0)
            right_peak = np.argmax(
                histogram[640:]) + 640  # Calculates the index of maximum value in the histogram for right lane
            # Here we use np.where to get the path coordinates inside a window which spans 50 pixels wide and 90 pixels height.
            x_idx = np.where((((right_peak - 25) < x) & (x < (right_peak + 25)) & ((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:  # Assuming path points are detected, add to list
                rightx.extend(x_window.tolist())
                righty.extend(y_window.tolist())
            i -= 90  # Keeps iterating while shifting the window vertically up
            j -= 90  # Keeps iterating while shifting the window vertically up
    if not np.sum(righty) > 0:  # If no path points are detected, choose the previous image points
        righty = Right.Y
        rightx = Right.X

    # LEFT LANE

    if Left.found == False:  # Perform blind search for lane lines
        i = 720  # y coordinate of the bottom pane of the initial window
        j = 630  # y coorfinate of the top pane of window window.
        while j >= 0:
            # Histogram in which bitwise addition is done along a horizontal strip for each x coordinate
            histogram = np.sum(image[j:i, :],
                               axis=0)  # Calculates the index of maximum value in the histogram for left lane
            left_peak = np.argmax(histogram[:640])
            # Here we use np.where to get the path coordinates inside a window which spans 50 pixels wide and 90 pixels height.
            x_idx = np.where((((left_peak - 25) < x) & (x < (left_peak + 25)) & ((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:  # Assuming path points are detected, add to list
                leftx.extend(x_window.tolist())
                lefty.extend(y_window.tolist())
            i -= 90  # Keeps iterating while shifting the window vertically
            j -= 90
    if not np.sum(lefty) > 0:
        lefty = Left.Y  # Keeps iterating while shifting the window vertically up
        leftx = Left.X  # # Keeps iterating while shifting the window vertically

    # Converting into array
    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)

    return lefty, leftx, righty, rightx


# This function finds the left and right lane radius of curvature at lane coordinate which is at the bottom of the image
def find_curvature(left_y, left_fy, right_y, right_fy):
    # Scaling x and y from pixels space to meters
    yscaled_per_pix = 30 / 720  # meters per pixel in y dimension
    xscaled_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(left_y * yscaled_per_pix, left_fy * xscaled_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * yscaled_per_pix, right_fy * xscaled_per_pix, 2)

    # Evaluating the formula above at the y value corresponding to the bottom of your image. Hence np.max(left_y)
    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(left_y) + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(right_y) + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


# This function uses cv2.fillPoly() to fill the lane detected with color and returns the final image
def draw_poly(undist, combined_binary, left_fitx, right_fitx, Minv, y_linspace):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # pts_left = np.array([np.transpose(np.vstack([Left.fitx, y_linspace]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y_linspace])))])
    # pts = np.hstack((pts_left, pts_right))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left.fitx, y_linspace])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, y_linspace]))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Mark borders of lanes with color
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0, 255), thickness=40)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    ## Find the position of the car (Using Find Position())
    pts = np.argwhere(newwarp[:, :, 1])
    position, dist_frm_center = find_position(combined_binary, pts)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result, dist_frm_center, position


# This function finds the position of the car in the image by calculating the distance from the center of the image.
# This is not an elegant solution because it doesn't really say where the car is located with respect to it's surroundings.
def find_position(image, pts):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    left = np.min(
        pts[(pts[:, 1] < 640) & (pts[:, 0] > 700)][:, 1])  # Calculates the left lane point towards the extreme left
    right = np.max(
        pts[(pts[:, 1] > 640) & (pts[:, 0] > 700)][:, 1])  # Calculates the right lane point towards the extreme right
    position = (left + right) / 2
    # Scaling x and y from pixels space to meters
    xscaled_per_pix = 3.7 / 700  # meters per pixel in x dimension
    return position, abs((640 - position) * xscaled_per_pix)


# This function calculates coefficients of the curve which approximates right lane.
def polyfit_right(righty, rightx, y_linspace):
    # Find the coefficients of polynomials
    right_coeff = np.polyfit(righty, rightx, 2)  # Coefficients right
    right_fitx = right_coeff[0] * y_linspace ** 2 + right_coeff[1] * y_linspace + right_coeff[2]

    # Sorting in ascending order
    rsort = np.argsort(righty)
    righty = righty[rsort]
    rightx = rightx[rsort]
    Right.X = rightx
    Right.Y = righty

    # Take the mean
    right_fit = np.polyfit(righty, rightx, 2)
    Right.fit0.append(right_fit[0])
    Right.fit1.append(right_fit[1])
    Right.fit2.append(right_fit[2])
    right_fit = [np.mean(Right.fit0), np.mean(Right.fit1), np.mean(Right.fit2)]

    # Fit polynomial to detected pixels
    right_fitx = right_fit[0] * y_linspace ** 2 + right_fit[1] * y_linspace + right_fit[2]
    Right.fitx = right_fitx

    return right_fitx, rightx


# The function calculates coefficients of the curve which approximates the left lane
def polyfit_left(lefty, leftx, y_linspace):
    # Find the coefficients of polynomials
    left_coeff = np.polyfit(lefty, leftx, 2)  # Coefficients left
    left_fitx = left_coeff[0] * y_linspace ** 2 + left_coeff[1] * y_linspace + left_coeff[2]

    # Sorting in ascending order
    lsort = np.argsort(lefty)
    lefty = lefty[lsort]
    leftx = leftx[lsort]
    Left.X = leftx
    Left.Y = lefty

    # Recalculate polynomial with intercepts and take the mean
    left_fit = np.polyfit(lefty, leftx, 2)
    Left.fit0.append(left_fit[0])
    Left.fit1.append(left_fit[1])
    Left.fit2.append(left_fit[2])
    left_fit = [np.mean(Left.fit0), np.mean(Left.fit1), np.mean(Left.fit2)]

    # Fit polynomial to detected pixels
    left_fitx = left_fit[0] * y_linspace ** 2 + left_fit[1] * y_linspace + left_fit[2]
    Left.fitx = left_fitx

    return left_fitx, leftx


def process_video(image):
	
	
    Left.__init__(Left)
    Right.__init__(Right)
    try:
        y_linspace = np.linspace(0, image.shape[0] - 1, image.shape[0])

        image_perspective, M, Minv = perspective_transform(image)  # Image warped

        combined_binary = binary_image(image_perspective)  # Image thresholded
        print('a')
        # Initialize empty lists for the lane line pixels
        rightx = []
        righty = []
        leftx = []
        lefty = []

        # Finding x, y coordinates of left & right lanes
        lefty, leftx, righty, rightx = find_lane_video(combined_binary, lefty, leftx, righty, rightx)
        print('b')
        # Designing curve which approximates the left lane
        left_fitx, leftx = polyfit_left(lefty, leftx, y_linspace)
        print('c')
        # Designing curve which approximates the right lane
        right_fitx, rightx = polyfit_right(righty, rightx, y_linspace)

        # Find curvature of left lane and right lane at the bottom of the screen(close to car)
        left_curverad, right_curverad = find_curvature(lefty, leftx, righty, rightx)
        print('d')
        # Only print the radius of curvature every 3 frames for improved readability
        if Left.count % 3 == 0:
            Left.radius = left_curverad
            Right.radius = right_curverad

        # Fill the detected lane with color for representation
        # This function is also used to calculate the position of the car and distance from center of screen
        result, distance_from_center, position = draw_poly(image, combined_binary, left_fitx, right_fitx, Minv,
                                                           y_linspace)

        # Remember recent polynomial coefficients and intercepts. Recycle after every 10 frames.
        if len(Left.fit0) > 10:
            Left.fit0 = Left.fit0[1:]
        if len(Left.fit1) > 10:
            Left.fit1 = Left.fit1[1:]
        if len(Left.fit2) > 10:
            Left.fit2 = Left.fit2[1:]

        if len(Right.fit0) > 10:
            Right.fit0 = Right.fit0[1:]
        if len(Right.fit1) > 10:
            Right.fit1 = Right.fit1[1:]
        if len(Right.fit2) > 10:
            Right.fit2 = Right.fit2[1:]

        Left.count += 1  # Counting the number of frames.
        '''
        # Print distance from center on video
        if position > 640:
            cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (100, 80),
                        fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
        else:
            cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (100, 80),
                        fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)

        # Print radius of curvature on video
        cv2.putText(result, 'Radius of Curvature {}(m)'.format(int((Left.radius + Right.radius) / 2)), (120, 140),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
        '''
        if position > 640:
            print('Vehicle is ', distance_from_center, 'left of center')
        else:
            print('Vehicle is ', distance_from_center, 'right of center')
        print('e')
    except Exception as e:
        cv2.imshow('frame', image)
        print(e)
        return image
    return result


# In[10]:

# final pipeline

# Capture the video from the wecam. Change the value 0 to 1 when using the webcam connected by the USB. Let the value be
# 0 when using the default camera in the laptop
cap = cv2.VideoCapture(0)

# get the frams per second
fps = cap.get(6)

# set the hieght and width of the image
cap.set(3, 1280)
cap.set(4, 1024)
width = cap.get(3)
hieght = cap.get(4)

while (True):

    # tic and toc are used to time the process
    tic1 = timeit.default_timer()
    # Capture frame-by-frame
    ret, image = cap.read()
    toc1 = timeit.default_timer()
    time1 = toc1 - tic1
    print('time to get image:', time1)

    tic = timeit.default_timer()
    # Call the final process video function to process each frame and send the final result of the distance from the
    # vehicle to the a variable called final
    final = process_video(image)
    toc = timeit.default_timer()
    time = toc - tic

    print('time to process the code:', time)

    # print('fps:',fps)
    print('width:', width)
    print('hieght :', hieght)

    # showing the final frame
    cv2.imshow('frame', final)

    print()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
