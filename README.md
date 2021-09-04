# camera_calibration
For camera's intrinsic and extrinsic calibration. Extrinsic calibration assumes all images are captured from one static pose.

**To generate calibration pattern,**

either use calib.io:
https://calib.io/pages/camera-calibration-pattern-generator

Or use pattern generation script that comes with OpenCV:
https://docs.opencv.org/4.5.1/da/d0d/tutorial_camera_calibration_pattern.html

**This implementation follows:**

+ checkerboard:
https://docs.opencv.org/4.5.2/dc/dbb/tutorial_py_calibration.html

+ circle board:
https://docs.opencv.org/4.5.1/d4/d94/tutorial_camera_calibration.html

**Special Notes:**

1. For checkerboard pattern, the number of rows and columns is actually the count of internal corners.
2. For asymmetric circle pattern, the number of rows need to be odd and the columns to be even.
3. For asymmetric circle pattern, the number of columns needs to take into account of indented columns as well to create the pattern properly. It is divided by half in the code to work with OpenCV API.
4. For all pattern, the origin is defined at the top left corner. However, for symmetric patterns, it is sometime rotated by 180 degree.
5. The detected axis of the pattern has z axis pointing into the plate (away from the camera).
