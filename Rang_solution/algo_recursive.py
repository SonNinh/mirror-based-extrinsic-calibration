import pickle
from typing import Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d



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
           monitor_size:Tuple, # (width, height)
           gaze_origin=None,
           gaze_direction=None,
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


    if gaze_origin is not None:
        vector_len = 600
        segments = np.stack(
            (gaze_origin + gaze_direction*vector_len, gaze_origin),
            axis=1
        ) 
        for point in segments:
            ax.plot(point[:, 0], point[:, 1], point[:, 2])


    plt.show()


def rot_x(alpha, points):
    R = np.array((
        (1, 0, 0),
        (0, np.cos(alpha), -np.sin(alpha)),
        (0, np.sin(alpha), np.cos(alpha))
    ))
    return np.matmul(R, points), R


def rot_y(alpha, points):
    R = np.array((
        (np.cos(alpha), 0, np.sin(alpha)),
        (0, 1, 0),
        (-np.sin(alpha), 0, np.cos(alpha))
    ))
    return np.matmul(R, points), R


def rot_z(alpha, points):
    R = np.array((
        (np.cos(alpha), np.sin(alpha), 0),
        (-np.sin(alpha), np.cos(alpha), 0), 
        (0, 0, 1)
    ))
    return np.matmul(R, points), R


def add_noise(x, level):
    return x + np.random.normal(0, level, x.shape)


def create_synthesis_data():
    # reference object in screen coordinate
    obj_ref_X = np.array((
        (0., 0., 0.),
        (100., 0., 0.),
        (200., 0., 0.),
        (0., 50., 0.),
        (100., 50., 0.)
    )) # (N, 3)

    # level of noise 
    noise_level = 0.05
    # reference object in camera coordinate
    obj_ref, R1 = rot_y(np.random.uniform(0, 180)/180*np.pi, obj_ref_X.T)
    obj_ref = add_noise(obj_ref, noise_level)
    obj_ref, R2 = rot_x(np.random.uniform(0, 180)/180*np.pi, obj_ref)
    obj_ref = add_noise(obj_ref, noise_level)
    obj_ref, R3 = rot_z(np.random.uniform(0, 180)/180*np.pi, obj_ref)
    obj_ref = add_noise(obj_ref, noise_level)
    obj_ref = obj_ref.T
    
    # translation matrix
    t = [np.random.uniform(0, 200), np.random.uniform(0, 180), np.random.uniform(0, 180)]
    obj_ref += np.array(t)
    obj_ref = add_noise(obj_ref, noise_level)
    print("Reference object in Camera coordinate:")
    print(obj_ref)

    gaze_origin = np.array((
        (120., 20., 305.),
        (121., 21., 305.),
        (121., 21., 305.),
        (123., 20., 305.),
        (123., 20., 305.)
    ))
    gaze_direction = obj_ref - gaze_origin
    R = np.matmul(R2, R1)
    R = np.matmul(R3, R)

    return gaze_origin, gaze_direction, obj_ref_X, R, t


def solve(gaze_origin:np.ndarray, 
          gaze_direction:np.ndarray,
          obj_ref_X:np.ndarray
          ):

    n_point = obj_ref_X.shape[0]
    A = np.zeros((3*n_point, 9+n_point))
    

    A[:, :3] = np.tile(np.eye(3), (n_point, 1))

    xy = np.repeat(obj_ref_X[:, :2], 3, axis=0)
    xy = np.repeat(xy, 3, axis=1)

    A[:, 3:9] = np.multiply(
        np.tile(np.eye(3), (n_point, 2)),
        xy
    )

    A[:, 9:] = np.repeat(np.eye(n_point), 3, axis=0) * gaze_direction.reshape(-1, 1)
   
    # B = np.tile(gaze_origin, n_point).reshape(-1)
    B = gaze_origin.reshape(-1)

    Z = np.linalg.lstsq(A, B, rcond=None)[0]

    t = Z[:3]
    r1 = Z[3:6]
    r1 /= np.linalg.norm(r1)
    r2 = Z[6:9]
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    R = np.array((r1, r2, r3)).T

    # orthoganal procrustes problem
    u, s, vh = np.linalg.svd(R)
    R = u @ vh

    k = Z[9:]

    return R, t, k


def solve2(gaze_origin:np.ndarray, 
          gaze_direction:np.ndarray,
          obj_ref_X:np.ndarray,
          R:np.ndarray
          ):

    n_point = obj_ref_X.shape[0]
    A = np.zeros((3*n_point, 3+n_point))
    
    A[:, :3] = np.tile(np.eye(3), (n_point, 1))
    A[:, 3:] = np.repeat(np.eye(n_point), 3, axis=0) * gaze_direction.reshape(-1, 1)
   
    # B = np.tile(gaze_origin, n_point).reshape(-1)
    B = (gaze_origin.reshape(-1, 3) - np.matmul(R, obj_ref_X.T).T).reshape(-1)

    Z = np.linalg.lstsq(A, B, rcond=None)[0]

    return Z[:3], Z[3:]


def rotation_error(R1, R2):
    sum_error = 0
    max_error = 0
    for i in range(3):
        angle = np.abs(
            np.arccos(np.dot(R1[:, i], R2[:, i]))*180/np.pi
        )
        if angle > max_error:
            max_error = angle
        sum_error += angle
    return max_error, sum_error


def translation_error(T1, T2):
    total = np.sum(np.abs(T1 - T2))
    return total, total*100/np.sum(np.abs(T2))


def main():
    # gaze_origin, gaze_direction, obj_ref_X, R_org, T_org = create_synthesis_data()

    mon_name = "G5"
    cam_name = "G5"
    with open(f"/home/sonnnb/Projects/gaze-point-estimation/demo/data/data_mon{mon_name}_cam{cam_name}.pkl", "rb") as f:
        data = pickle.load(f)
 
    gaze_origin = np.array(data['gaze_origin']).reshape(-1, 3)
    gaze_direction = np.array(data['gaze_forward']).reshape(-1, 3)
    obj_ref_X = np.array(data['object_ref']).reshape(-1, 3)

    print(obj_ref_X)

    R, T, k = solve(gaze_origin, gaze_direction, obj_ref_X)


    # # compute the error in the first iteration
    # max_error, sum_error = rotation_error(R_org, R)
    # print(f'The rotation error - maximum angle error: %.2f, total angle error: %.2f.' % (max_error, sum_error))
    # tran_error, per_error = translation_error(T, T_org)
    # print(f'The tranlation error in the first iteration - absolute: %.2f, percentage: %.1f%%.' % (tran_error, per_error))

    # # apply second time with the rotation matrix found in the first stage
    # T, k = solve2(gaze_origin, gaze_direction, obj_ref_X, R)

    # # compute the error in the second iteration
    # tran_error, per_error = translation_error(T, T_org)
    # print(f'The tranlation error in the second iteration - absolute: %.2f, percentage: %.1f%%.' % (tran_error, per_error))
    
    print(k)
    visual(R, T, (344, 194, 0), gaze_origin[10:14], gaze_direction[10:14])
    return R, T, k  


if __name__ == "__main__":

    main()