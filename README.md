# camera_calibration
For camera's intrinsic and extrinsic calibration. Extrinsic calibration to be added.

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

+ charuco (not reliable so far):
https://docs.opencv.org/4.5.1/df/d4a/tutorial_charuco_detection.html

**Special Notes:**
1. For asymmetric circle pattern, the number of rows need to be odd and the columns to be even.
2. For ChArUco pattern, the origin is defined at the bottom left corner due to the order marker is created.
3. For other patterns, the origin is defined at the top left corner.
4. The detected axis of the pattern has z axis pointing into the plate (away from the camera).
