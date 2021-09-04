import os
import cv2
import glob
import json
import math
import argparse
import numpy as np

# TODO::use logging module to handle print output

def create_pattern(spec):
    # TODO::draw the pattern out and verify
    pattern = np.zeros((spec["cols"] * spec["rows"], 3), np.float32)

    # create grid
    grid = np.mgrid[0 : spec["cols"], 0 : spec["rows"]].T.reshape(-1, 2)
    grid = grid * spec["grid_size"]

    # handle asymmetric
    if "asymmetric" in spec:
        if spec["asymmetric"] == True:
            new_grid = []

            # keep only even half
            for r in range(spec["rows"]):
                for c in range(spec["cols"]):
                    if (r + c) % 2 == 0:
                        new_grid.append(grid[c + r * spec["cols"]])

            # adjust scaling
            grid = np.array(new_grid) / math.sqrt(2)
            
            # half the feature count
            pattern = np.zeros((len(grid), 3), np.float32)

    # apply offset
    grid[:, 0] += spec["x_offset"]
    grid[:, 1] += spec["y_offset"]

    pattern[:, 0:2] = grid
    return pattern


def find_checkerboard_pattern_in_images(image_paths, spec):
    create_pattern(spec)

    pattern_points = {}
    image_points = {}
    images_overlayed = {}

    # construct pattern
    pattern = create_pattern(spec)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find corners points
        ret, corners = cv2.findChessboardCorners(img_gray, (spec["cols"], spec["rows"]), None)

        if ret == True:
            # refine corner points
            corners_accurate = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            # TODO::extract the below common part either by creating a wrapper function or as a sub function (depends on how ChArUco should be handled)
            image_overlayed = cv2.drawChessboardCorners(image, (spec["cols"], spec["rows"]), corners_accurate, ret)

            # add points
            pattern_points[image_path] = pattern
            image_points[image_path] = corners_accurate
            images_overlayed[image_path] = image_overlayed

        else:
            print("[WARN]: Failed to find pattern in image: '{}'".format(image_path))

    return pattern_points, image_points, images_overlayed


def find_circle_pattern_in_images(image_paths, spec):
    pattern_points = {}
    image_points = {}
    images_overlayed = {}

    # construct pattern
    pattern = create_pattern(spec)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find circle centers
        if spec["asymmetric"]:
            if spec["cols"] % 2 == 0 and spec["rows"] % 2 == 1:
                pattern_size = (math.floor(spec["cols"] / 2), spec["rows"])
            else:
                exit("[FAIL]: asymmetric pattern is actual symmetric.")
            success, centers = cv2.findCirclesGrid(img_gray, pattern_size, None, cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
        else:
            pattern_size = (spec["cols"], spec["rows"])
            success, centers = cv2.findCirclesGrid(img_gray, pattern_size, None, cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)

        if success == True:
            # Draw and display the corners
            image_overlayed = cv2.drawChessboardCorners(image, pattern_size, centers, success)

            # add points
            pattern_points[image_path] = pattern
            image_points[image_path] = centers
            images_overlayed[image_path] = image_overlayed

        else:
            print("[WARN]: Failed to find pattern in image: '{}'".format(image_path))

    return pattern_points, image_points, images_overlayed


def save_calibration_result(file_path, image_paths, image_size, intrinsic, distortion, t_vecs, r_vecs):
    calib_result = {}
    calib_result["width"] = image_size[1]
    calib_result["height"] = image_size[0]
    calib_result["intrinsic"] = intrinsic.reshape(-1).tolist()
    calib_result["distortion"] = distortion.tolist()
    calib_result["extrinsic"] = []

    for image_path, r_vec, t_vec in zip(image_paths, r_vecs, t_vecs):
        extrinsic = {}
        extrinsic["image"] = os.path.basename(image_path)
        
        pose = r_vec.reshape(-1).tolist()
        pose.extend(t_vec.reshape(-1).tolist())
        extrinsic["pose"]  = pose

        mat = np.eye(4, 4)
        mat_rot, jacobian = cv2.Rodrigues(r_vec)
        mat[0:3, 0:3] = mat_rot
        mat[0:3, 3] = t_vec[:, 0]
        extrinsic["matrix"] = mat.flatten().tolist()

        # TODO::add pose to matrix conversion
        calib_result["extrinsic"].append(extrinsic)
    
    with open(file_path, "w") as f:
        json.dump(calib_result, indent=4, fp=f)
    
    print("[INFO]: Calibration output saved to: '{}'".format(file_path))


def save_overlayed_images(output_folder, images_overlayed, intrinsic, distortion, r_vecs, t_vecs):
    output_folder = os.path.join(os.path.dirname(output_folder), "overlayed")
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)

    for (image_path, image), r_vec, t_vec in zip(images_overlayed.items(), r_vecs, t_vecs):
        image_output = cv2.drawFrameAxes(image, intrinsic, distortion, r_vec, t_vec, 30)
        
        _, file_name = os.path.split(image_path)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(output_folder, base_name + "_overlayed" + ext)
        cv2.imwrite(output_path, image_output)

    print("[INFO]: Saved overlayed images to: '{}'".format(output_folder))


def main():
    #############
    # arguments #
    #############
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input image(s) or folder of images", required=True)
    parser.add_argument("--output", help="File or folder to output the calibration result", default=None, required=False)
    parser.add_argument("--key", help="Keyword of images (e.g. *.png)", default="", required=False)
    parser.add_argument("--pattern", help="JSON file that specifies the calibration pattern.", required=True)
    parser.add_argument("-extrinsic", help="Extrinsic calibration. Combine points from multiple static images.", action="store_true")
    parser.add_argument("-save_images", help="Save overlayed images.", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.input) == False:
        exit("[FAIL]: Input path is not valid.")
    
    if os.path.exists(args.pattern) == False:
        exit("[FAIL]: Pattern file does not exist.")

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
        exit("[FAIL]: Failed to find images in folder '{}' with key '{}'".format(args.input, args.key))

    for image_path in image_paths:
        print("[INFO]: Found images: '{}'".format(image_path))
    
    ##########################
    # find pattern in images #
    ##########################
    # load pattern
    with open(args.pattern, 'r') as f:
        pattern = json.load(f)
    
    # find pattern
    if "checkerboard" in pattern:
        spec = pattern["checkerboard"]
        pattern_points, image_points, images_overlayed = find_checkerboard_pattern_in_images(image_paths, spec)
    elif "circles" in pattern:
        spec = pattern["circles"]
        pattern_points, image_points, images_overlayed = find_circle_pattern_in_images(image_paths, spec)
    else:
        exit("[FAIL]: Unrecognized pattern name.")

    if len(image_points) == 0:
        exit("[FAIL]: Failed to find any pattern.")

    #############
    # calibrate #
    #############
    overlayed = list(images_overlayed.values())
    image_size = overlayed[0].shape[0:-1]

    pattern_points_calib = list(pattern_points.values())
    image_points_calib = list(image_points.values())
    if args.extrinsic:
        pattern_points_calib = np.concatenate(pattern_points_calib)
        image_points_calib = np.concatenate(image_points_calib)
    
    reprojection_err, intrinsic, distortion, r_vecs, t_vecs = cv2.calibrateCamera(pattern_points_calib, image_points_calib, image_size, None, None)

    print("[INFO]: Reprojection error: {}".format(reprojection_err))

    #######################
    # save output as JSON #
    #######################
    save_calibration_result(args.output, image_paths,image_size, intrinsic, distortion, t_vecs, r_vecs)
    
    # save loverlayed images
    if args.save_images:
        save_overlayed_images(args.output, images_overlayed, intrinsic, distortion, r_vecs, t_vecs)


if __name__ == "__main__":
    main()
