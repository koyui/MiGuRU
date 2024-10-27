# Author: koyui
# This File is designed to prepare masks for a certain position to subtract lines and diagonals

import numpy as np
import json
from utils import *

# Enumerating the start of the main diagonal
mainDiagStart = [(i, 1) for i in reversed(range(1, 7))] + [(1, i) for i in range(2, 7)]
mainDiag = [[] for _ in range(11)]

# Enumerating the start of the anti diagonal
antiDiagStart = [(1, i) for i in range(3, 9)] + [(i, 8) for i in range(2, 7)]
antiDiag = [[] for _ in range(11)]

for i, pos in enumerate(mainDiagStart):
    x, y = pos
    while True:
        if not check_border(x, y):
            break
        mainDiag[i].append((x, y))
        x += 1
        y += 1

for diag in mainDiag:
    print(diag)

for i, pos in enumerate(antiDiagStart):
    x, y = pos
    while True:
        if not check_border(x, y):
            break
        antiDiag[i].append((x, y))
        x += 1
        y -= 1

for diag in antiDiag:
    print(diag)


mask = np.zeros((POS_MAX, 10), dtype=np.uint64)
# Masks:
#   0:h, 1:v, 2:main diag, 3:anti diag
# Which diagonal:
#   4:mainNo, 5:antiNo
# Chess pos for diagonal:
#   6:mainPos, 7:antiPos
# All mask for the Chessboard
#   8:all
# Neighbor Mask
#   9:neighbor

for b in range(POS_MAX):             # Iterate over all board positions
    x, y = bit2pos(b)
    hMask = sum([2 ** pos2bit(x, i) for i in range(1, 9)])
    vMask = sum([2 ** pos2bit(i, y) for i in range(1, 9)])
    mainMask = 0
    mainNo = 12                 # 11 diags in total, 12 means diag not exist.
    mainPos = 0
    for i, diag in enumerate(mainDiag):
        if (x, y) in diag:
            mainMask = sum([2 ** pos2bit(*pos) for pos in diag])
            mainNo = i
            mainPos = diag.index((x, y))
            break
    antiMask = 0
    antiNo = 12                 # 11 diags in total, 12 means diag not exist.
    antiPos = 0
    for i, diag in enumerate(antiDiag):
        if (x, y) in diag:
            antiMask = sum([2 ** pos2bit(*pos) for pos in diag])
            antiNo = i
            antiPos = diag.index((x, y))
            break

    mask[b][0] = hMask
    mask[b][1] = vMask
    mask[b][2] = mainMask
    mask[b][3] = antiMask
    mask[b][4] = mainNo
    mask[b][5] = antiNo
    mask[b][6] = mainPos
    mask[b][7] = antiPos
    mask[b][8] = ~np.uint64(hMask | vMask | mainMask | antiMask)
    mask[b][9] = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in directions:
        if check_border(x + dx, y + dy):
            mask[b][9] |= np.uint64(1 << pos2bit(x + dx, y + dy))

vHash = {}
mainDiagHash = [{} for _ in range(11)]
antiDiagHash = [{} for _ in range(11)]
RevHash = np.zeros((3, UINT8_MAX, POS_MAX), dtype=np.uint64)
# 0: vertical
# 1: mainDiagonal
# 2: antiDiagonal

# Construct vertical hash Dict
for i in range(UINT8_MAX):
    for y in range(1, 9):
        decoder = np.array([1 << pos2bit(x, y) for x in range(1, 9)], dtype=np.uint64)
        base = np.array(uint8_2_List(np.uint8(i)), dtype=np.uint64)
        ori_state = int(np.sum(base * decoder, dtype=np.uint64))
        vHash[ori_state] = i
        for x in range(1, 9):
            RevHash[0][i][pos2bit(x, y)] = ori_state

# Construct main hash Dict
for diagNo in range(11):
    for i in range(1 << diagBits[diagNo]):
        if diagNo == 5 and i == 0b01110000:
            print()
        decoder = np.zeros(BITS_MAX, dtype=np.uint64)
        for cnt, pos in enumerate(mainDiag[diagNo]):
            decoder[cnt] = 1 << pos2bit(*pos)
        base = np.array(uint8_2_List(np.uint8(i)), dtype=np.uint64)
        ori_state = int(np.sum(base * decoder, dtype=np.uint64))
        mainDiagHash[diagNo][ori_state] = i
        for pos in mainDiag[diagNo]:
            RevHash[1][i][pos2bit(*pos)] = ori_state

        decoder = np.zeros(BITS_MAX, dtype=np.uint64)
        for cnt, pos in enumerate(antiDiag[diagNo]):
            decoder[cnt] = 1 << pos2bit(*pos)
        base = np.array(uint8_2_List(np.uint8(i)), dtype=np.uint64)
        ori_state = int(np.sum(base * decoder, dtype=np.uint64))
        antiDiagHash[diagNo][ori_state] = i
        for pos in antiDiag[diagNo]:
            RevHash[2][i][pos2bit(*pos)] = ori_state

# Construct a list from position to the diagonal it belongs
# 0 is Main Diagonal
# 1 is Anti Diagonal
# If diagBelonging is -1, don't calculate the diagonal
diagBelonging = [[-1 for _ in range(POS_MAX)], [-1 for _ in range(POS_MAX)]]
for b in range(POS_MAX):
    for cnt, diag in enumerate(mainDiag):
        if bit2pos(b) in diag:
            diagBelonging[0][b] = cnt
    for cnt, diag in enumerate(antiDiag):
        if bit2pos(b) in diag:
            diagBelonging[1][b] = cnt


hashDict = {
    'vHash': vHash,
    'mainDiagHash': mainDiagHash,
    'antiDiagHash': antiDiagHash,
    'diagBelonging': diagBelonging
}

np.save('../data/mask.npy', mask)
np.save('../data/revHash.npy', RevHash)
with open('../data/hashDict.json', 'w') as f:
    json.dump(hashDict, f, indent=4)

