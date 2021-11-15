import cv2
import numpy as np
import glob
from os import path
import pickle


def calib(img_root):

    vertice_shape = (9, 6)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros(
        (vertice_shape[0]*vertice_shape[1], 3), 
        np.float32
    )
    objp[:,:2] = np.mgrid[:vertice_shape[0], :vertice_shape[1]].T.reshape(-1, 2)
    
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
            # corners2 = correct_order(corners2, vertice_shape)
            imgpoints.append(corners2)
            centers.append(np.mean(corners2, axis=0).reshape(-1))
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, vertice_shape, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey()
    


    cv2.destroyAllWindows()

    print("number of samples:", len(imgpoints))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[0:2], None, None
    )
    print(mtx, dist)

    output = {
        'mtx': mtx,
        'dist': dist
    }

    pickle.dump(
        output, 
        open(path.join(img_root, "calib.pkl"), "wb")
    )



if __name__ == "__main__":

    camera_name = 'G5'
    img_root = f'data/intrinsic/cam{camera_name}'
    
    calib(img_root)