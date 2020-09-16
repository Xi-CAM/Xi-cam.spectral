import numpy as np
from numpy import random


from pytest import fixture
import random
from skimage import data
from scipy.ndimage.interpolation import shift

from pystxmtools.corrections.register import register_frames_stack

# @fixture
def test_stack():
    image = data.camera()
    n_frames = 10
    # shifts = (round(random.uniform(1, 50), 2), round(random.uniform(1, 50), 2))
    im_shift_stack = []
    for n in range(n_frames):
        shifts = (round(random.uniform(1, 50), 2), round(random.uniform(1, 50), 2))
        el = [shift(image, shifts), shifts]
        im_shift_stack.append(el)
    return np.asarray(im_shift_stack)


def test_register_frame_stack():
    "Test something in register_frame_stack "
    image = data.camera()
    n_frames = 10
    # shifts = (round(random.uniform(1, 50), 2), round(random.uniform(1, 50), 2))
    im_stack = []
    shift_stack = []
    for n in range(n_frames-1):
        shifts = [round(random.uniform(1, 50), 2), round(random.uniform(1, 50), 2)]
        shift_stack.append(shifts)

        shifted_frames = shift(image, shifts)
        im_stack.append(shifted_frames)
    # aligned_frames, calc_shifts = register_frames_stack(frames=test_stack[:,0])
    aligned_frames, calc_shifts = register_frames_stack(np.asarray(im_stack))

    return aligned_frames, calc_shifts, shift_stack, shifts, np.asarray(im_stack)

    # assert calc_shifts == shift_stack
    # assert calc_shifts == test_stack[:,1]


if __name__ == "__main__":
    test_stack()
