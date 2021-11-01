import numpy as np
from numpy.linalg import norm


def forward():
    obj_ref = np.array((
        (1., 1., -4.),
        (3., 1., -4.),
        (1., 5., -4.)
    ), )

    norm_vec = np.array((
        (0., 0., -1.),
        (1., 0., -1.),
        (-1., 0.5, -1.)
    ))
    norm_vec /= np.linalg.norm(norm_vec, axis=1, keepdims=True)
    mirror_distance = np.array((7., 8., 9.))

    

print(forward())
