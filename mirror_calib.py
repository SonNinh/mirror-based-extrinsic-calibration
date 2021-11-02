import numpy as np
from numpy.linalg import norm
import cv2



def distance_points_planes(P: np.ndarray, 
                         norm_vec: np.ndarray, 
                         d: np.ndarray) -> np.ndarray:
    '''
    Params:
        P: (N, 3)
            batch of points
        norm_vec: (M, 3)
            batch of normal unit vectors of planes
        d: (M, 1)
            batch of distances from origin to planes
    Returns:
        distance: (M, N)
    '''
    distance = np.matmul(norm_vec, P.T) + d
    return distance


def forward(obj_ref: np.ndarray, 
            norm_vec: np.ndarray, 
            mirror_distance: np.ndarray) -> np.ndarray:
    '''
    Compute mirrored object of referenc object via mirror's poses
    '''
    distance = distance_points_planes(obj_ref, norm_vec, mirror_distance) # (M, N)

    obj_mir = np.expand_dims(obj_ref, axis=0) - 2 * np.expand_dims(distance, axis=-1) * np.expand_dims(norm_vec, axis=-2) # (M, N, 3)

    return obj_mir
    


def backward(obj_mir: np.ndarray):
    Q = obj_mir - np.roll(obj_mir, -1, axis=0)
    M = np.matmul(np.transpose(Q, (0, 2, 1)), Q)
    print(M)

    e_val, e_vec = np.linalg.eig(M)
    print(e_val)
    print(e_vec)
    # return mij  


def main():
    # reference object
    obj_ref = np.array((
        (1., 1., -4.),
        (3., 1., -4.),
        (1., 5., -4.)
    )) # (N, 3)

    # 3 predefined mirror poses
    norm_vec = np.array((
        (0., 0., -1.),
        (1., 0., -1.),
        (-1., 0.5, -1.)
    )) # (M, 3)
    norm_vec /= np.linalg.norm(norm_vec, axis=1, keepdims=True)
    mirror_distance = np.array((7., 8., 9.)).reshape(-1, 1) # (M, 1)

    obj_mir = forward(obj_ref, norm_vec, mirror_distance)

    mij = np.cross(norm_vec[1], norm_vec[2])

    est_mij = backward(obj_mir)

    # print(mij)

    # print(est_mij)


if __name__ == "__main__":
    main()


