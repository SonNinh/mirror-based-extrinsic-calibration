import os
import cv2
import numpy as np
import glob
from os import path
import pickle


def calib(img_root, vertice_shape):

    

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

    images = sorted(glob.glob(path.join(img_root, '*.png')))

    if len(images) > 5:
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, vertice_shape, None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
                imgpoints.append(corners2)
    
    else:
        cap = cv2.VideoCapture(2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        img_id = 0

        ret, img = cap.read()

        while(ret):
            ret, img = cap.read()
            cv2.imshow('cam', img)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ret_board, corners = cv2.findChessboardCorners(gray, vertice_shape, None)
            # If found, add object points, image points (after refining them)
            # if ret_board:
            #     objpoints.append(objp)
            #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
            #     imgpoints.append(corners2)
            #     cv2.drawChessboardCorners(img, vertice_shape, corners2, ret)
            #     cv2.imshow("board", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"{img_root}/{img_id}.png", img)
                img_id += 1
            
            

        cap.release()
        cv2.destroyAllWindows()


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

    camera_name = 'Logitech'
    img_root = f'data/intrinsic/cam{camera_name}'
    if not os.path.isdir(img_root):
        os.mkdir(img_root)
        
    vertice_shape = (8, 6)
    
    calib(img_root, vertice_shape)