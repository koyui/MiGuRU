# Author: koyui
# This File is designed to prepare data for Successor Calculation and State Aggregation

import numpy as np
from utils import *

sf = np.zeros((BITS_MAX + 1, UINT8_MAX, UINT8_MAX, BITS_MAX, 3), dtype=np.uint8)
# 0: Valid bit
# 1: Update Player State
# 2: Update Opponent State

for b in np.arange(3, BITS_MAX + 1, dtype=np.uint8):          # At least three to make a legal move.
    for i in np.arange(0, 1 << b, dtype=np.uint8):            # i:PlayerState
        for j in np.arange(0, 1 << b, dtype=np.uint8):        # j:OpponentState
            if b == 8 and i == 0b00001000 and j == 0b01110000:
                print()
            res = np.uint8(0)
            if i & j:
                continue
            iList = uint8_2_List(i, b)
            jList = uint8_2_List(j, b)
            kList = np.logical_not(np.bitwise_or(iList, jList))
            for sp in range(b):                         # Enumerate empty spaces
                sf[b][i][j][sp][0] = 0                  # Valid move
                sf[b][i][j][sp][1] = (i | (1 << sp))    # Player Old State + new space
                sf[b][i][j][sp][2] = j                  # Opponent Old State
                if kList[sp]:
                    valid = False                       # Whether sp is legal to place chess
                    meetOpponentChess = False
                    nowPos = sp
                    while nowPos >= 1:
                        nowPos -= 1
                        if not meetOpponentChess and iList[nowPos]:
                            break
                        if kList[nowPos]:
                            break
                        if jList[nowPos]:
                            meetOpponentChess = True
                            continue
                        if meetOpponentChess and iList[nowPos]:
                            valid = True
                            break

                    if valid:
                        sf[b][i][j][sp][0] = 1
                        meetOpponentChess = False
                        nowPos = sp
                        while nowPos >= 1:
                            nowPos -= 1
                            if not meetOpponentChess and iList[nowPos]:
                                break
                            if kList[nowPos]:
                                break
                            if jList[nowPos]:
                                meetOpponentChess = True
                                sf[b][i][j][sp][1] |= (1 << nowPos)
                                sf[b][i][j][sp][2] &= clearBit(nowPos)
                                continue
                            if meetOpponentChess and iList[nowPos]:
                                break

                    valid = False
                    meetOpponentChess = False
                    nowPos = sp
                    while nowPos < b - 1:
                        nowPos += 1
                        if not meetOpponentChess and iList[nowPos]:
                            break
                        if kList[nowPos]:
                            break
                        if jList[nowPos]:
                            meetOpponentChess = True
                            continue
                        if meetOpponentChess and iList[nowPos]:
                            valid = True
                            break

                    if valid:
                        sf[b][i][j][sp][0] = 1
                        meetOpponentChess = False
                        nowPos = sp
                        while nowPos < b - 1:
                            nowPos += 1
                            if not meetOpponentChess and iList[nowPos]:
                                break
                            if kList[nowPos]:
                                break
                            if jList[nowPos]:
                                meetOpponentChess = True
                                sf[b][i][j][sp][1] |= (1 << nowPos)
                                sf[b][i][j][sp][2] &= clearBit(nowPos)
                                continue
                            if meetOpponentChess and iList[nowPos]:
                                break

np.save('../data/sf.npy', sf)

