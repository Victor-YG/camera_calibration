import os
import cv2
import glob
import json
import math
import argparse
import numpy as np

# TODO::use logging module to handle print output
# TODO::test circle pattern and ChArUco pattern

def create_pattern(spec):
    # TODO::draw the pattern out and verify
    pattern = np.zeros((spec["rows"] * spec["cols"], 3), np.float32)

    # create grid
    grid = np.mgrid[0 : spec["rows"], 0 : spec["cols"]].T.reshape(-1, 2)
    grid = grid * spec["grid_size"]

    # handle asymetric
    if "asymmetric" in spec:
        if spec["asymetric"]:
            new_grid = []

            # keep only even half
            for r in range(spec["rows"]):
                for c in range(spec["cols"]):
                    if r + c % 2 == 0:
                        new_grid.append(grid[c + r * spec["cols"]])

            # adjust scaling
            grid = new_grid / math.sqrt(2)

    # apply offset
    grid[:, 0] += spec["x_offset"]
    grid[:, 1] += spec["y_offset"]

    pattern[:, 0:2] = grid
    # print(pattern)
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
        ret, corners = cv2.findChessboardCorners(img_gray, (spec["rows"], spec["cols"]), None)

        if ret == True:
            # refine corner points
            corners_accurate = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            # TODO::extract the below common part either by creating a wrapper function or as a sub function (depends on how ChArUco should be handled)
            image_overlayed = cv2.drawChessboardCorners(image, (spec["rows"], spec["cols"]), corners_accurate, ret)

            # add points
            pattern_points[image_path] = pattern
            image_points[image_path] = corners_accurate
            images_overlayed[image_path] = image_overlayed

        else:
            print("[WARN] Failed to find pattern in image: '{}'".format(image_path))

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
        if spec["asymetric"]:
            ret, centers = cv2.findCirclesGrid(img_gray, (spec["rows"], spec["cols"]), cv2.CALIB_CB_ASYMMETRIC_GRID)
        else:
            ret, centers = cv2.findCirclesGrid(img_gray, (spec["rows"], spec["cols"]), cv2.CALIB_CB_SYMMETRIC_GRID)

        if ret == True:
            # Draw and display the corners
            image_overlayed = cv2.drawChessboardCorners(image, (spec["rows"], spec["cols"]), centers, ret)

            # add points
            pattern_points[image_path] = pattern
            image_points[image_path] = centers
            images_overlayed[image_path] = image_overlayed

        else:
            print("[WARN] Failed to find pattern in image: '{}'".format(image_path))

    return pattern_points, image_points, images_overlayed


def find_charuco_pattern_in_iamges(image_paths, spec):
    point_ids = {}
    points = {}
    images_overlayed = {}

    # create dictionary
    if spec["dictionary"] == "AruCo_DICT_4X4":
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    elif spec["dictionary"] == "AruCo_DICT_6x6":
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    # create pattern
    pattern = cv2.aruco.CharucoBoard_create(spec["rows"], spec["cols"], spec["grid_size"], spec["marker_size"], aruco_dict)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, aruco_dict)

        if len(corners) > 0:
            # refine corners
            corners_accurate = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # interpolate corners
            count, corners_final, ids_final = cv2.aruco.interpolateCornersCharuco(corners, ids, img_gray, pattern)
            if count > 0:
                points[image_path] = corners_final
                point_ids[image_path] = ids_final

                # Draw and display the corners
                image_overlayed = cv2.drawChessboardCorners(image_path, (spec["rows"], spec["cols"]), corners_final)
                images_overlayed[image_path] = image_overlayed

    return pattern, point_ids, points, images_overlayed


def find_charuco_pattern_in_iamges_2(image_paths, spec):
    pattern_points = {}
    image_points = {}
    images_overlayed = {}

    # construct pattern
    pattern = create_pattern(spec)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    pattern = cv2.aruco.CharucoBoard_create(spec["rows"], spec["cols"], 1, .8, aruco_dict)

    for image_path in image_paths:
        image = cv2.imread(image_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find circle centers
        if spec["asymetric"]:
            ret, centers = cv2.findCirclesGrid(img_gray, (spec["rows"], spec["cols"]), cv2.CALIB_CB_ASYMMETRIC_GRID)
        else:
            ret, centers = cv2.findCirclesGrid(img_gray, (spec["rows"], spec["cols"]), cv2.CALIB_CB_SYMMETRIC_GRID)

        if ret == True:
            # Draw and display the corners
            image_overlayed = cv2.drawChessboardCorners(image, (spec["rows"], spec["cols"]), centers, ret)

            # add points
            pattern_points[image_path] = pattern
            image_points[image_path] = centers
            images_overlayed[image_path] = image_overlayed

        else:
            print("[WARN] Failed to find pattern in image: '{}'".format(image_path))

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
    
    print("[INFO] Calibration output saved to: '{}'".format(file_path))


def save_overlayed_images(output_folder, images_overlayed):
    output_folder = os.path.join(os.path.dirname(output_folder), "overlayed")
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)
    
    for image_path, image in images_overlayed.items():
        _, file_name = os.path.split(image_path)
        base_name, ext = os.path.splitext(file_name)
        output_path = os.path.join(output_folder, base_name + "_overlayed" + ext)
        cv2.imwrite(output_path, image)
    
    print("[INFO] Saved overlayed images to: '{}'".format(output_folder))


def main():
    #############
    # arguments #
    #############
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input image(s) or folder of images", required=True)
    parser.add_argument("--output", help="File or folder to output the calibration result", default=None, required=False)
    parser.add_argument("--key", help="Keyword of images (e.g. *.png)", default="", required=False)
    parser.add_argument("--pattern", help="JSON file that specifies the calibration pattern.", required=True)
    # parser.add_argument("-intrinsic", action="store_true")
    # parser.add_argument("-extrinsic", action="store_true")
    parser.add_argument("-save_images", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.input) == False:
        exit("[FAIL] Input path is not valid.")
    
    if os.path.exists(args.pattern) == False:
        exit("[FAIL] Pattern file does not exist.")

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
        exit("[FAIL] Failed to find images in folder '{}' with key '{}'".format(args.input, args.key))

    print("[INFO] Found images: ")
    for image_path in image_paths:
        print(image_path)
    
    ##########################
    # find pattern in images #
    ##########################
    # load pattern
    with open(args.pattern, 'r') as f:
        pattern = json.load(f)
    
    # find pattern
    # TODO::modify the behavior to return dictionary with keys for all images but None if pattern not found
    if "checkerboard" in pattern:
        spec = pattern["checkerboard"]
        pattern_points, image_points, images_overlayed = find_checkerboard_pattern_in_images(image_paths, spec)
    elif "circle" in pattern:
        spec = pattern["circle"]
        pattern_points, image_points, images_overlayed = find_circle_pattern_in_images(image_paths, spec)
    elif "ChArUco" in pattern:
        spec = pattern["ChArUco"]
        pattern, pattern_points, image_points, images_overlayed = find_charuco_pattern_in_iamges(image_paths, spec)

    if len(image_points) == 0:
        exit("[FAIL] Failed to find any pattern.")

    #############
    # calibrate #
    #############
    overlayed = list(images_overlayed.values())
    image_size = overlayed[0].shape[0:-1]

    if "ChArUco" in pattern:
        points = [x for x in image_points.values() if len(x) >= 4]
        point_ids = [x for x in pattern_points.values() if len(x) >= 4]
        ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(points, point_ids, pattern, image_size, None, None)
    else:
        ret, intrinsic, distortion, r_vecs, t_vecs = cv2.calibrateCamera(list(pattern_points.values()), list(image_points.values()), image_size, None, None)

    #######################
    # save output as JSON #
    #######################
    save_calibration_result(args.output, image_paths, intrinsic, distortion, t_vecs, r_vecs)
    
    # save loverlayed images
    if args.save_images:
        save_overlayed_images(args.output, images_overlayed)


if __name__ == "__main__":
    main()
