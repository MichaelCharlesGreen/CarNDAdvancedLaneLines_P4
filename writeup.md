##Writeup
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/straight_lines1_copy.jpg "Road Transformed"
[image3]: ./output_images/thresholded_image.png "Binary Example"
[image4]: ./output_images/undistorted_and_warped_with_points.png "Warp Example"
[image5]: ./output_images/color_fit_lines_copy.jpg "Fit Visual"
[image6]: ./output_images/video_frame.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3rd cell of the IPython notebook at "./Advanced_Lane_Finding.ipynb". The image below is produced in the fourth code cell.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

As above, the camera calibration and distortion coefficients were used along with the `cv2.undistort()` function to form the distortion-corrected image.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. The 17th code cell, implementing the function, `threshold()`, produces the image displayed below.

I used the absolute Sobel gradients in the x and y directions, the magnitude threshold gradient and the direction threshold gradients. HLS proved to be a good color space to use.

Here's an example of my output for this step:

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is a function called `warp()`, which appears in the 23rd code cell of the IPython notebook.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```
# Define the source and destination points for the perspective transform.
# These points define polygons used in the perspective transform.
# The src polygon define the lane on a straight line image.
# The dst polygon deine the shape we'd like the src polygon to be in the transformed image.
src_top_left = (575,464) # 464 is the y-value for the top of the lane polygon.
src_top_right = (707,464)
src_bottom_left = (201,720) # 720 is the y-value for the bottom of the image/ bottom of the lane polygon.
src_bottom_right = (1104,720)
# From this, the equations for the left and right lane lines are:
# left lane line: y = -0.68x + 857.58
# right lane line: y = 0.64x + 8.10
# solving for x:
# -0.68x + 857.58 = 0.64x + 8.10
# The left and right lane lines intersect at:
# (x, y): (643.55, 419.97)
# The center of the image is:
# (x, y): (640, 360)
# The base of the source polygon is: 903 pixels wide.
# We know, from the U.S. government specifications for highway curvature, require a minimum lane width
# of 12 feet or 3.7 meters and dashed lane lines that of 10 feet or 3 meters length.


# h is the height of the image and is 720 for the test images.
# w is the width of the image and is 1280 for the test images.
h,w = thresholded_image.shape[:2]

# Source polygon.
# The width at the top of this polygon is 132 pixels.
# The width at the bottom of this polybon is 903 pixels.
# The height of this polygon is 256 pixels.
src = np.float32([src_top_left,
                  src_top_right,
                  src_bottom_left,
                  src_bottom_right])

# Destination (post-tranformation image) polygon.
# The width at the top of this polygon is 380 pixels.
# The width at the bottom of this polybon is 380 pixels.
# The height of this polygon is 720 pixels.

# Choosing adj for dst polygon:
# A samller adj results in a wide dst polygon, which may cut-off lane lines.
# A larger adj results in a narrow dst polygon, which, also, may cut-off lane lines.
# Choose an adj such that optimal points for lane lines are produced.
adj = 450
dst = np.float32([(adj,0),
                  (w-adj,0),
                  (adj,h),
                  (w-adj,h)])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 450, 0        | 
| 707, 464      | 830, 0      |
| 201, 720     | 450, 720      |
| 1104, 720      | 830, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

It would be better to generalize here and throughout the codebase for any size image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The `warp()` function produces a binary warped image similar to the warped image displayed above, but in binary, meaning each pixel is either black or white. This image is the one used to fit lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

**Step 5 of 6: Detect Lane Lines and Fit Polynomials** in the IPython notebook is the section where the lines are fit. There are two functions: `sliding_window_polyfit()`, in code cell 28 of the IPython notebook and `subsequent_frames_polyfit()`, code cell 32 of the IPython notebook, which fit lines to points in the binary image.

A histogram of the binary warped image (from function `warp()`, in code cell 23 of the IPython notebook) is taken. This histogram indicates regions in the binary warped image where vertical and near-vertical lines could be. This region is further refined by identifying the regions where lane lines would be rather than a highway divider's edge, noise, or shadow.

The `sliding_window_polyfit()` uses the histogram and refined region and search windows to look for points that look to be a lane's left and right edges and fits a 2nd order polynomial to them.

The `subsequent_frames_polyfit()` is used to fit a 2nd order polynomial to viable points, but rather than using a histogram and search window, it relies on a previous fit to establish a search area.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane and the position of the vehicle with respect to center in calculated in the `calc_curv_rad_and_car_offser()` fucntion in the 34th code cell of the IPython notebook.

Left and right lane edge cloud points are used to fit a 2nd order polynomial using a real-world adjustment factor so that the results are in meters.

These fitted-lines are then used in a formula to calculate the radii.

As for the center offset position of the car... The coefficients for the left and right fitted-lines are used to get the equations for the two lane lines, which are then used to get the x-intercepts for these lines (x-values for the lane lines at the bottom of the image). The center of the car is assumed to be the center of the image. The differnece between the center of the car and the center of the lane is the offset.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The function `process_car_camera()` in code cell 42 of the IPython notebook is used to process image frames from the project video (or, any video or still image). This function makes use of a `Line` class in code cell 41 of the IPython notebook to keep track of lane lines and smooth the processing.

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is likely to fail along steep curves as it does in the challenge video. The abrupt shadow caused by an overpass in the challenge video also causes problems.

I would collect still images from the video where these problems are occurring and use them to refine the pipeline. This could be implementing a lane line radius check, tightening the search windows width, tweaking variables in the `Line` class. I could also further restrict the lane-width constraint on accepting lines.

