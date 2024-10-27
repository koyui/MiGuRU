# Author: koyui
# This is a backup of previously unrunnable code,
# but the functions in it may still be useful,
# so I'll keep it for now.

import numpy as np
from dataPrepare.utils import *

mainDiag = np.load('data/mainDiag.npy')
antiDiag = np.load('data/antiDiag.npy')
sf = np.load('data/sf.npy')

def hlineSubtract(state: np.uint64):
    """
        Subtract 8 horizonal lines uint8 states.
    """
    mask = np.uint8(0b11111111)
    hlines = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        hlines[i] = np.uint8(state & mask)
        state >>= np.uint8(8)
    return hlines

def vlineSubstract(state: np.uint64):
    """
        Subtract 8 vertical lines uint8 states.
    """
    mask = np.uint8(0b11111111)
    vlines = np.zeros(8, dtype=np.uint8)
    for i in np.arange(0, 8, dtype=np.uint8):
        vlines |= (uint8_2_List(np.uint8(state & mask)) << i)
        state >>= np.uint(8)
    return vlines

def diagSubtract(state: np.uint64):
    """
        Subtract 11 main diagonal lines and 11 anti diagonal lines uint8 states.
        Discard 4 diagonals since that only when length > 3 it may generate succcessors
    """
    mainDiags = np.zeros(11, dtype=np.uint8)
    antiDiags = np.zeros(11, dtype=np.uint8)
    states = np.full(8, state, dtype=np.uint64)
    for i in range(11):
        main = mainDiag[i]
        anti = antiDiag[i]
        mainBits = np.bitwise_and(main, states)
        antiBits = np.bitwise_and(anti, states)
        mainDiags[i] = np.packbits(mainBits != 0, bitorder='little')[0]
        antiDiags[i] = np.packbits(antiBits != 0, bitorder='little')[0]
    return mainDiags, antiDiags

def hPlace(lines) -> np.uint64:
    state = np.uint64(0)
    for line in lines:
        state <<= np.uint8(8)
        state |= line
    return state

def vPlace(lines) -> np.uint64:
    state = np.uint64(0)
    for line in lines:
        state <<= np.uint8(8)
        state |= line
    return state

def calculate_successor(player: np.uint64, opponent: np.uint64) -> np.uint64:
    """
        From two uint64 state, calculate the successor uint64 state.
    """
    hPlayer = hlineSubtract(player)
    hOpponent = hlineSubtract(opponent)
    vPlayer = vlineSubstract(player)
    vOpponent = vlineSubstract(opponent)
    mainPlayer, antiPlayer = diagSubtract(player)
    mainOpponent, antiOpponent = diagSubtract(opponent)

    hRes = np.zeros(8, dtype=np.uint8)
    vRes = np.zeros(8, dtype=np.uint8)
    mainRes = np.zeros(8, dtype=np.uint8)
    antiRes = np.zeros(8, dtype=np.uint8)

    # sf Index 0-4 correspond to actual index 3-7 as well as gird 4-8
    for i in range(8):
        hRes[i] = sf[4][hPlayer[i]][hOpponent[i]]
        vRes[i] = sf[4][vPlayer[i]][vOpponent[i]]
    for i in range(11):
        if i <= 5:
            b = i + 3
        else:
            b = 11 - i
        mainRes[i] = sf[b - 3][mainPlayer[i]][mainOpponent[i]]
        antiRes[i] = sf[b - 3][antiPlayer[i]][antiOpponent[i]]