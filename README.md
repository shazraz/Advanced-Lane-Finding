# Advanced-Lane-Finding

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration3.jpg "Chessboard Image"
[image2]: ./camera_cal/calibration3_cal.jpg "Chessboard Image with corners drawn"

## 1. Camera Calibration

The first step of the exercise is to correctly calibrate the camera and and calculate the distortion co-efficients so that the input images can be undistorted prior to processing. The OpenCV functions ```cv2.findChessboardCorners()``` and ```cv2.drawChessboardCorners()``` are used to first find the corners of a 9x6 chessboard and determine the calibration matrix to calibrate the camera. The image below shows an example of the corners identified.

Original Image             |  Image with corners 
:-------------------------:|:-------------------------:
<img src="./camera_cal/calibration3.jpg" width="400">  |   <img src="./camera_cal/calibration3_cal.jpg" width="500"> 

The identified corners are stored in the ```img_points``` array and transformed to the prepared ```obj_points``` array which consists of a square grid. The image below shows an example of an undistorted chessboard image once the calibration matrix has been calculated. Please refer to code block 4 in the project notebook for details.

<img src="./output_images/UndistortedChessboard.jpg">

## 2. Image pipeline

### 2.1 Undistort Image
The first step in the image pipeline is to undistort the input images using the calibration matrix calculated earlier. The image below shows a comparison of a distorted and undistorted test image.

<img src="./output_images/UndistortedTest.jpg">

Next a combination of gradient and color thresholding is applied to extract the lane locations from the image.

The sobel-x and sobel-y binaries are first obtained by thresholding the sobel gradients in each direction, the result binary images are bit-wise AND'ed to extract the lane lines and stripes with high confidence. This approach works well for images that are well illuminated and the lane lines are pronounced which is the case for the project video. A bit-wise AND operation on these images captures mostly the lane lines and eliminates the noise picked up by the sobel-y binary. A gaussian blur is then applied to the combined image to further smooth the image.

### 2.2 Color & Gradient Thresholding
Next, a yellow and white threshold is applied to the image in the HSV color space as shown in the code block below:

```
def yellow_selector(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow_threshold = np.asarray([10, 100, 100])
    upper_yellow_threshold = np.asarray([50, 255, 255])
    binary_img = cv2.inRange(img, lower_yellow_threshold, upper_yellow_threshold)
    #binary_img = cv2.bitwise_and(img, img, mask=mask)
    return binary_img/255

def white_selector(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_white_threshold = np.array([0,0,200], dtype = np.uint8)
    upper_white_threshold = np.array([255,30,255], dtype = np.uint8)
    binary_img = cv2.inRange(img, lower_white_threshold, upper_white_threshold)
    return binary_img/255
```
The upper and lower bounds for each color were obtained after experimentation with the images using Adobe Photoshop. Photoshop's color selection feature was then used to obtain the HSV (HSB in photoshop) values for the lane lines at several points and used as a starting point for the color selector functions. The resulting binaries were OR'ed to obtain both the yellow and white lane lines.

Next, the image was transformed into HLS color space and the S and H channels were extracted. It was found that under certain lighting conditions, the H and S channels were more reliable than the color selectors. As a result, the H and S channels were bitwise AND'ed and the resulting image was OR'ed with the combinted yellow and white threshold image. This final image is refered to as ```color_output``` in the code block below. This binary image was OR'ed with the result of the sobel gradient operations ```grad_output``` to obtain a final output image.

```
x_binary = abs_sobel_thresh(image, 'x', 5, sobel_thresh)
y_binary = abs_sobel_thresh(image, 'y', 3, sobel_thresh)
S_binary = HLS_threshold(image, 'S', S_thresh)
H_binary = HLS_threshold(image, 'H', H_thresh)
W_binary = white_selector(image)
Y_binary = yellow_selector(image)

color_output = OR_binaries(OR_binaries(W_binary, Y_binary), AND_binaries(S_binary, H_binary))
grad_output = blur_gradient(AND_binaries(x_binary, y_binary))
output = OR_binaries(color_output, grad_output)
```
The complete operation with individual binaries and their combinations is shown in the image below:
<img src="./output_images/threshold_image.jpg">

### 2.3 Perspective Transformation




