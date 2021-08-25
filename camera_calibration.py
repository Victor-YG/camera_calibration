import os
from posixpath import basename, join
import cv2
import glob
import json
import argparse
import numpy as np


def find_checkerboard_pattern_in_images(image_paths, rows, cols, spacing):
    pattern_points = {}
    image_points = {}
    images_overlayed = {}

    # construct pattern
    pattern = np.zeros((rows * cols, 3), np.float32)
    grid    = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    corners = grid * spacing
    pattern[:, 0:2] = corners
    # print(pattern)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find corners points
        ret, corners = cv2.findChessboardCorners(img_gray, (rows, cols), None)

        if ret == True:
            # refine corner points
            corners_accurate = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            image_overlayed = cv2.drawChessboardCorners(image, (rows,cols), corners_accurate, ret)

            # add points
            pattern_points[image_path] = pattern
            image_points[image_path] = corners_accurate
            images_overlayed[image_path] = image_overlayed

        else:
            print("Failed to find pattern in image: '{}'".format(image_path))

    return pattern_points, image_points, images_overlayed


def save_calibration_result(file_path, image_paths, intrinsic, distortion, t_vecs, r_vecs):
    calib_result = {}

    calib_result["intrinsic"] = intrinsic.reshape(-1).tolist()
    calib_result["distortion"] = distortion.tolist()
    calib_result["extrinsic"] = []

    for image_path, r_vec, t_vec in zip(image_paths, r_vecs, t_vecs):
        extrinsic = {}
        extrinsic["image"] = os.path.basename(image_path)
        pose = r_vec.reshape(-1).tolist()
        pose.extend(t_vec.reshape(-1).tolist())
        extrinsic["pose"]  = pose
        # TODO::add pose to matrix conversion
        calib_result["extrinsic"].append(extrinsic)
    
    with open(file_path, "w") as f:
        json.dump(calib_result, indent=4, fp=f)
    
    print("Calibration output saved to: '{}'".format(file_path))


def main():
    #############
    # arguments #
    #############
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input image(s) or folder of images", required=True)
    parser.add_argument("--output", help="File or folder to output the calibration result", default=None, required=False)
    parser.add_argument("--key", help="Keyword of images (e.g. *.png)", default="", required=False)
    parser.add_argument("--pattern", help="JSON file that specifies the calibration pattern.", required=True)
    parser.add_argument("-intrinsic", action="store_true")
    parser.add_argument("-extrinsic", action="store_true")
    parser.add_argument("-save_images", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.input) == False:
        exit("Input path is not valid.")
    
    if os.path.exists(args.pattern) == False:
        exit("Pattern file does not exist.")

    if args.output == None:
        if os.path.isdir(args.input):
            args.output = os.path.join(args.input, "calib_result.json")
        else:
            args.output = os.path.dirname(args.input)
    
    ###############
    # find images #
    ###############
    image_paths = sorted(glob.glob(os.path.join(args.input, args.key)))

    if len(image_paths) == 0:
        exit("Failed to find images in folder '{}' with key '{}'".format(args.input, args.key))

    print("Found images: ")
    for image_path in image_paths:
        print(image_path)
    
    ##########################
    # find pattern in images #
    ##########################
    # load pattern
    with open(args.pattern, 'r') as f:
        pattern = json.load(f)
    
    # find pattern
    if "checkerboard" in pattern:
        spec = pattern["checkerboard"]
        pattern_points, image_points, images_overlayed = find_checkerboard_pattern_in_images(image_paths, spec["rows"], spec["cols"], spec["spacing"])

    if len(image_points) == 0:
        exit("Failed to find any pattern.")

    #############
    # calibrate #
    #############
    overlayed = list(images_overlayed.values())
    image_size = overlayed[0].shape[0:-1]
    ret, intrinsic, distortion, r_vecs, t_vecs = cv2.calibrateCamera(list(pattern_points.values()), list(image_points.values()), image_size, None, None)

    #######################
    # save output as JSON #
    #######################
    save_calibration_result(args.output, image_paths, intrinsic, distortion, t_vecs, r_vecs)
    
    # save loverlayed images
    if args.save_images:
        for image_path, image in images_overlayed.items():
            folder, file_name = os.path.split(image_path)
            base_name, ext = os.path.splitext(file_name)
            output_path = os.path.join(folder, base_name + "_overlayed" + ext)
            cv2.imwrite(output_path, image)


if __name__ == "__main__":
    main()
