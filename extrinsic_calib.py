import numpy as np
import cv2
import glob
import pickle

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# objp *= 2
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey()
    else:
        print(fname)


cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img.shape[0:2], None, None
)
print(objpoints[0].shape)
print(imgpoints[0].shape)

cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}

cam_calib['mtx'] = mtx
cam_calib['dist'] = dist
print("Camera parameters:")
print(cam_calib)
print()
print(rvecs[0])
print(tvecs[0])

pickle.dump(cam_calib, open("calib_cam_external.pkl", "wb"))

