import numpy as np
from numpy import random


from pytest import fixture
import random
from skimage import data
from scipy.ndimage.interpolation import shift

from pystxmtools.corrections.register import register_frames_stack

@fixture
def test_stack():
    image = data.camera()
    n_frames = 10
    # shifts = (round(random.uniform(1, 50), 2), round(random.uniform(1, 50), 2))
    im_shift_stack = []
    for n in range(n_frames):
        shifts = (round(random.uniform(1, 50), 2), round(random.uniform(1, 50), 2))
        el = [shift(image, shifts), 0]
        im_shift_stack.append(el)
    return np.asarray(im_shift_stack)


def test_register_frame_stack(test_stack):
    "Test something in register_frame_stack "
    aligned_frames, calc_shifts = register_frames_stack(frames=test_stack[:,0])

    assert aligned_frames.shape[0] == test_stack[:,0].shape[0]
    assert calc_shifts == test_stack[:,1]


# if __name__ == "__main__":
#     test_stack()
