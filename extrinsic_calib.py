import glob
import pickle
from typing import List, Tuple

import cv2
import numpy as np

from algo import mirror_calib
from os import path
import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pylab as pl




class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def visual(R:np.ndarray, 
           t:np.ndarray, 
           monitor_size:Tuple # (width, height)
           ) -> None:


    origin = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]) * 200
    
    # camera origin
    cam_orogin = origin.reshape(-1, 2, 3)

    # monitor origin
    monitor_origin = (R @ origin.T).T + t
    monitor_origin = monitor_origin.reshape(-1, 2, 3)

    # monitor
    mon_points = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.]
    ]) * monitor_size
    mon_points = (R @ mon_points.T).T + t
    mon_points = np.repeat(mon_points, 2, axis=0)
    mon_points = np.roll(mon_points, 1, axis=0).reshape(-1, 2, 3)
    
    allpoints = np.concatenate((
        cam_orogin.reshape(-1, 3),
        monitor_origin.reshape(-1, 3),
        mon_points.reshape(-1, 3)
    ))
    xmin, ymin, zmin = np.min(allpoints, axis=0) - 100
    xmax, ymax, zmax = np.max(allpoints, axis=0) + 100
    
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel('Y')
    ax.set_ylim(ymin, ymax)
    ax.set_zlabel('Z')
    ax.set_zlim(zmin, zmax)

    # set equal ratio xyz axis
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))


    # Here we create the arrows:
    for i in range(4):
        ax.plot(mon_points[i, :, 0], mon_points[i, :, 1], mon_points[i, :, 2], color='b')

    arrow_prop_dict = dict(mutation_scale=10, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D(cam_orogin[0, :, 0], cam_orogin[0, :, 1], cam_orogin[0, :, 2], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D(cam_orogin[1, :, 0], cam_orogin[1, :, 1], cam_orogin[1, :, 2], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D(cam_orogin[2, :, 0], cam_orogin[2, :, 1], cam_orogin[2, :, 2], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    a = Arrow3D(monitor_origin[0, :, 0], monitor_origin[0, :, 1], monitor_origin[0, :, 2], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D(monitor_origin[1, :, 0], monitor_origin[1, :, 1], monitor_origin[1, :, 2], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D(monitor_origin[2, :, 0], monitor_origin[2, :, 1], monitor_origin[2, :, 2], **arrow_prop_dict, color='g')
    ax.add_artist(a)

    # nx = monitor_origin[0, 1] - monitor_origin[0, 0]
    # ny = monitor_origin[1, 1] - monitor_origin[1, 0]
    # nz = monitor_origin[2, 1] - monitor_origin[2, 0]

    plt.show()
    

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
    return area

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


def calib(img_root, intrinsic_fname=''):
    with open(path.join(img_root, "checkboard_monitor.pkl"), "rb") as f:
        data = pickle.load(f)
        objp = data['objp']
        vertice_shape = objp.shape[:2] # [W, H]
        objp = objp.reshape((-1, 3))

        monitor_size = (*data['mon_size'], 0)
    
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
            # cv2.drawChessboardCorners(img, vertice_shape, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey()
    


    cv2.destroyAllWindows()

    p0, p1, p2 = get_ref_mirror_points(centers)
    print("The 3 images used for extrinsic calib are:", detected_name[p0], detected_name[p1], detected_name[p2])

    if intrinsic_fname is '':
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

    cam_calib = {
        'mtx': np.eye(3), 
        'dist': np.zeros((1, 5)),
        'R': np.eye(3),
        't': np.zeros(3)
    }

    cam_calib['mtx'] = mtx
    cam_calib['dist'] = dist
    cam_calib['R'] = R
    cam_calib['t'] = t

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

    monitor_name = 'Dell24'
    camera_name = 'G5'

    img_root = f'data/extrinsic/mon{monitor_name}_cam{camera_name}'
    intrinsic_fname = f'data/intrinsic/cam{camera_name}/calib.pkl'
    calib(img_root, intrinsic_fname)