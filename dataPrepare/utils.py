# Author: koyui
# Util tools for data preparation

import numpy as np

UINT8_MAX = 2 ** 8
POS_MAX = 64
BITS_MAX = 8
diagBits = [3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3]

def uint8_2_List(state, bitCounts=8):
    """
        Turn binary state into list.
    """
    return np.unpackbits(state, bitorder='little')[:bitCounts]

def check_border(x, y):
    """
        Check whether (x, y) is in border.
    """
    if x < 1 or x > 8:
        return False
    if y < 1 or y > 8:
        return False
    return True

def pos2bit(x, y):
    """
        Converting a position on the board to one of the 64 bits
    """
    return (x - 1) * 8 + (y - 1)

def bit2pos(b):
    return b // 8 + 1, b % 8 + 1

def clearBit(b):
    """
        Create a mask with bit b as 0 and the other bits as 1
    """
    mask = ~(np.uint8(1) << np.uint8(b))
    return mask
