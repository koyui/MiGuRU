# Author: koyui
# This File is designed to support bits state aggregation and successor functions support

import numpy as np
import json
import time
from tqdm import tqdm

with open('data/hashDict.json', 'r') as f:
    hashDict = json.load(f)

vHash = hashDict['vHash']
mainDiagHash = hashDict['mainDiagHash']
antiDiagHash = hashDict['antiDiagHash']
"""
    hashes uint64 -> uint8
    Extract lines features to uint8 state
"""

diagBelonging = hashDict['diagBelonging']
"""
    diagBelonging(2 x POS_MAX)
    
    Construct a list from position to the diagonal it belongs
    - 0 is Main Diagonal
    - 1 is Anti Diagonal
    
    If diagBelonging is -1, don't calculate the diagonal
"""


mask = np.load('data/mask.npy')
"""
    Masks(POS_MAX x 9)
    pos -> mask for lines and diagonals:
        - 0:h, 1:v, 2:main diag, 3:anti diag
        
    Which diagonal:
        - 4:mainNo, 5:antiNo
        
    Chess pos for diagonal:
        - 6:mainPos, 7:antiPos
    All mask for the Chessboard
        - 8:all
    # Neighbor Mask
        - 9:neighbor
"""

revHash = np.load('data/revHash.npy')
"""
    revHash(3 x UINT8_MAX x POS_MAX)
    - 0: vertical
    - 1: mainDiagonal
    - 2: antiDiagonal
    
    Convert the uint8 state to uint64 state, position specified
"""

sf = np.load('data/sf.npy')
"""
    sf(9 x UINT8_MAX x UINT8_MAX x 8 x 3)
    dim0:
        - bits 1 to 8
    dim1:
        - Player state
    dim2:
        - Opponent state
    dim3:
        - New position for the chess on a 8-bit line
    dim4:
        - 0: Valid bit
        - 1: Update Player State
        - 2: Update Opponent State
"""

diagBits = [3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3]
def uint64_2_matrix(state: np.uint64):
    """
        Turn binary uint64 state into 8x8 Matrix.
    """
    uint8_list = np.frombuffer(state.tobytes(), dtype=np.uint8)
    return np.unpackbits(uint8_list, bitorder='little').reshape((8, 8))

def uint8_2_List(state, bitCounts=8):
    """
        Turn binary uint8 state into list.
    """
    return np.unpackbits(state, bitorder='little')[:bitCounts]

def bit2pos(b):
    """
        Turn 64-bit into x, y pos. Note that x, y is 0-indexed
    """
    return np.uint64(b >> np.uint64(3)), np.uint64(b & np.uint64(0b111))

def pos2bit(x, y):
    """
        Turn x, y pos into 64-bit. Note that x, y is 0-indexed
    """
    return np.uint64((x << 3) + y)

def calculate_successor(player: np.uint64, opponent: np.uint64):
    """
        From two uint64 state, calculate the successor (action tuple and 2uint64 state) as a tuple.
    """
    successor = []
    space = ~(player | opponent)
    for b in range(64):                                         # Enumerate bit
        b = np.uint64(b)
        if opponent & mask[b][9]:                               # b has an opponent neighbor
            if space & (np.uint64(1) << b):                     # b is space where chess can be played.
                legal = False
                x, y = np.uint64(b >> np.uint64(3)), np.uint64(b & np.uint64(0b111))
                playerOld = player & mask[b][8]
                opponentOld = opponent & mask[b][8]
                playerUpdate = np.uint64(0)
                opponentUpdate = np.uint64(0)

                hMask = mask[b][0]
                hPlayer = hMask & player
                hOpponent = hMask & opponent
                h1 = (hPlayer & (np.uint64(0b11111111) << (x << np.uint64(3)))) >> (x << np.uint64(3))
                h2 = (hOpponent & (np.uint64(0b11111111) << (x << np.uint64(3)))) >> (x << np.uint64(3))
                hRes = sf[8][h1][h2][y]
                if hRes[0]:
                    legal = True
                    hPlayer = (hRes[1] << (x << np.uint64(3)))
                    hOpponent = (hRes[2] << (x << np.uint64(3)))
                playerUpdate |= hPlayer
                opponentUpdate |= hOpponent

                vMask = mask[b][1]
                vPlayer = vMask & player
                vOpponent = vMask & opponent
                v1 = vHash[str(vPlayer)]
                v2 = vHash[str(vOpponent)]
                vRes = sf[8][v1][v2][x]
                if vRes[0]:
                    legal = True
                    vPlayer = revHash[0][vRes[1]][b]
                    vOpponent = revHash[0][vRes[2]][b]
                playerUpdate |= vPlayer
                opponentUpdate |= vOpponent

                mainNo = mask[b][4]
                antiNo = mask[b][5]

                if mainNo != 12:
                    mainDiagMask = mask[b][2]
                    mainPos = mask[b][6]
                    mainDiagPlayer = mainDiagMask & player
                    mainDiagOpponent = mainDiagMask & opponent
                    main1 = mainDiagHash[mainNo][str(mainDiagPlayer)]
                    main2 = mainDiagHash[mainNo][str(mainDiagOpponent)]
                    mainRes = sf[diagBits[mainNo]][main1][main2][mainPos]
                    if mainRes[0]:
                        legal = True
                        mainDiagPlayer = revHash[1][mainRes[1]][b]
                        mainDiagOpponent = revHash[1][mainRes[2]][b]
                    playerUpdate |= mainDiagPlayer
                    opponentUpdate |= mainDiagOpponent

                if antiNo != 12:
                    antiDiagMask = mask[b][3]
                    antiPos = mask[b][7]
                    antiDiagPlayer = antiDiagMask & player
                    antiDiagOpponent = antiDiagMask & opponent
                    anti1 = antiDiagHash[antiNo][str(antiDiagPlayer)]
                    anti2 = antiDiagHash[antiNo][str(antiDiagOpponent)]
                    antiRes = sf[diagBits[antiNo]][anti1][anti2][antiPos]
                    if antiRes[0]:
                        legal = True
                        antiDiagPlayer = revHash[2][antiRes[1]][b]
                        antiDiagOpponent = revHash[2][antiRes[2]][b]
                    playerUpdate |= antiDiagPlayer
                    opponentUpdate |= antiDiagOpponent

                if legal:
                    successor.append(((int(x), int(y)), (playerOld | playerUpdate, opponentOld | opponentUpdate)))
    return successor

def construct_state(board):
    """
        To be consistent with the current interface,
        this function may be subsequently deprecated if all states in frameworks are changed to uint64.
    """
    player = np.uint64(0)
    opponent = np.uint64(0)
    for x in range(8):
        for y in range(8):
            b = pos2bit(x, y)
            if board[x][y] == 'B':
                player |= (np.uint64(1) << b)
                continue
            if board[x][y] == 'W':
                opponent |= (np.uint64(1) << b)
    return player, opponent

def reconstruct_board(player: np.uint64, opponent: np.uint64):
    """
        To be consistent with the current interface,
        this function may be subsequently deprecated if all states in frameworks are changed to uint64.
    """
    newBoard = []
    for b in range(64):
        b = np.uint64(b)
        if not b & np.uint64(0b111):
            newBoard.append([])
        if player & (np.uint64(1) << b):
            newBoard[-1].append('B')
            continue
        if opponent & (np.uint64(1) << b):
            newBoard[-1].append('W')
            continue
        newBoard[-1].append('E')
    return newBoard

def successor_api(gamestate, nowPlayer):
    board = gamestate.board
    player, opponent = construct_state(board, nowPlayer)
    # For debug
    # print(player, opponent, nowPlayer)
    successors = calculate_successor(player, opponent)
    for i, (act, state) in enumerate(successors):
        newState = gamestate.deepcopy()
        newState.board = reconstruct_board(*state, nowPlayer)
        newState.currentPlayer = newState.getOpponentPlayer(newState.currentPlayer)
        successors[i] = (act, newState)
    return successors

def bitCounts(state: np.uint64):
    return bin(state).count('1')

# For Debug

# black = np.uint64(8925885759488)
# white = np.uint64(580999673776965632)
# black_matrix = uint64_2_matrix(black)
# white_matrix = uint64_2_matrix(white)
# print(uint64_2_matrix(black))
# print(uint64_2_matrix(white))
# seen_matrix = black_matrix * 2 + white_matrix
# successors = calculate_successor(black, white)
# for act, state in successors:
#     print('action', act)
#     print('State:')
#     black_matrix = uint64_2_matrix(state[0])
#     white_matrix = uint64_2_matrix(state[1])
#     new_matrix = black_matrix * 2 + white_matrix
#     board = reconstruct_board(*state, 'B')
#     for line in board:
#         print(line)

# startTime = time.time()
# black = np.uint64(8925885759488)
# white = np.uint64(580999673776965632)
# for i in range(100000):
#     successors = calculate_successor(black, white)
# print(time.time() - startTime)