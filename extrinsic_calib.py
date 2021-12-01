import glob
import pickle
from typing import List, Tuple

import cv2
import numpy as np

from algo import mirror_calib
from os import path
import itertools
from visual import visual


def transform(rvec, tvec, objp):
    R = cv2.Rodrigues(rvec)[0]

    return ((R @ objp.T) + tvec).T 


def correct_order(corners: np.ndarray,
                  vertice_shape):
    if corners[0, 0, 1] - corners[-1, 0, 1] > 0:
        corners = corners[::-1]
    
    corners = corners.reshape((vertice_shape[1], vertice_shape[0], 1, 2))
    corners = corners[:, ::-1]
    corners = corners.reshape((-1, 1, 2))
        
    return corners


def check_Rt(mtx, dist, obj_mir, imgpoints, images):
    for i in range(3):
        img = cv2.imread(images[i])
        points = np.array(obj_mir[i].reshape(-1, 1, 3))
        projAxes, _ = cv2.projectPoints(
            points, 
            np.array((0., 0., 0.)).reshape(3, 1),
            np.array((0., 0., 0.)).reshape(3, 1),
            mtx, 
            dist
        )
        # projAxes = projAxes.reshape(-1)
        projAxes = np.array(projAxes)
    
        for p in projAxes:
            cv2.circle(img, (int(p[0, 0]), int(p[0, 1])), 2, (255, 0, 0), thickness=-1)

        cv2.imshow('check', img)
        cv2.waitKey(0)

    # projAxes = projAxes.reshape(-1)

    return


def get_area(p0, p1, p2):
    area = 0.5 * ((p0[0]*(p1[1]-p2[1])) + (p1[0]*(p2[1]-p0[1])) + (p2[0]*(p0[1]-p1[1])))
    return abs(area)


def get_ref_mirror_points(centers:np.ndarray):
    assert len(centers) >= 3, "number of detected checker board must greater than 2"
    max_area = 0
    ref_point = []
    
    for p1, p2, p3 in itertools.combinations(range(len(centers)), 3):
        area = get_area(centers[p1], centers[p2], centers[p3])
        if area > max_area:
            max_area = area
            ref_point = (p1, p2, p3)
    
    return ref_point


def calib(img_root, intrinsic_fname):
    
    with open(path.join(img_root, "checkboard_monitor.pkl"), "rb") as f:
        data = pickle.load(f)
        objp = data['objp']
        vertice_shape = objp.shape[:2] # [W, H]
        objp = objp.reshape((-1, 3))

        monitor_size = (*data['mon_size'], 0)
        monitor_res = data['mon_res']

    
    print("Object points:")
    print(objp)
    print("Vertice shape:")
    print(vertice_shape)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    centers = []
    detected_name = []

    images = sorted(glob.glob(path.join(img_root, '*.png')))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, vertice_shape, None)
        # If found, add object points, image points (after refining them)
        if ret:
            detected_name.append(fname)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
            corners2 = correct_order(corners2, vertice_shape)
            imgpoints.append(corners2)
            centers.append(np.mean(corners2, axis=0).reshape(-1))
            # Draw and display the corners
            cv2.drawChessboardCorners(img, vertice_shape, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey()
    


    cv2.destroyAllWindows()

    p0, p1, p2 = get_ref_mirror_points(centers)
    print("The 3 images used for extrinsic calib are:", detected_name[p0], detected_name[p1], detected_name[p2])

    if intrinsic_fname == '':
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[0:2], None, None
        )
        rvec0, tvec0 = rvecs[p0]
        rvec1, tvec1 = rvecs[p1]
        rvec2, tvec2 = rvecs[p2]
        
    else:
        with open(intrinsic_fname, 'rb') as f:
            data = pickle.load(f)
            mtx = data['mtx']
            dist = data['dist']

        ret, rvec0, tvec0 = cv2.solvePnP(objpoints[p0], imgpoints[p0], mtx, dist)
        ret, rvec1, tvec1 = cv2.solvePnP(objpoints[p1], imgpoints[p1], mtx, dist)
        ret, rvec2, tvec2 = cv2.solvePnP(objpoints[p2], imgpoints[p2], mtx, dist)
        
    obj_mir = np.array((
        transform(rvec0, tvec0, objp),
        transform(rvec1, tvec1, objp),
        transform(rvec2, tvec2, objp)
    ))
    # check_Rt(mtx, dist, obj_mir, imgpoints, images)
    R, t, d = mirror_calib(objp, obj_mir)

    cam_calib = dict()
        # 'mtx': np.eye(3), 
        # 'dist': np.zeros((1, 5)),
        # 'R': np.eye(3),
        # 't': np.zeros(3)
    

    # cam_calib['mtx'] = mtx
    # cam_calib['dist'] = dist
    cam_calib['R'] = R
    cam_calib['t'] = t
    cam_calib['mon_res'] = monitor_res[:2]
    cam_calib['mon_size'] = monitor_size[:2]
    

    pickle.dump(cam_calib, open(path.join(img_root, "calib.pkl"), "wb"))

    print("Intrinsic:")
    print(mtx)
    print(dist)
    print("Extrinsic:")
    print("R:", R)
    print("t:", t)
    print("d:", d)

    visual(R, t, monitor_size)


if __name__ == "__main__":
    

    # monitor_name = 'Dell274K'
    # camera_name = 'Logitech'
    
    monitor_name = input("Monitor name: ")
    camera_name = input("Camera name: ")

    img_root = f'data/extrinsic/mon{monitor_name}_cam{camera_name}'
    intrinsic_fname = f'data/intrinsic/cam{camera_name}/calib.pkl'
    calib(img_root, intrinsic_fname)