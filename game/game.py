import cProfile
import copy
import game.utils
import pygame
from stateSupport.stateSupport import bitCounts
from stateSupport.stateSupport import calculate_successor
from stateSupport.stateSupport import construct_state
from stateSupport.stateSupport import reconstruct_board
import numpy as np
import time

class GameState:
    """
    size: size of the board(6 or 8)
    board: 2d list filled with {PLAYER_BLACK, PLAYER_WHITE, EMPTY_SQUARE}
    currentPlayer: {PLAYER_BLACK, PLAYER_WHITE}
    """

    PLAYER_BLACK = "B"
    PLAYER_WHITE = "W"
    EMPTY_SQUARE = "E"

    RESULT_WHITE_WIN = "W_WIN"
    RESULT_BLACK_WIN = "B_WIN"
    RESULT_DRAW = "DRAW"
    RESULT_ON = "ON"

    BOARD_UNIT = 40
    COLOR_BLACK = [0, 0, 0]
    COLOR_WHITE = [255, 255, 255]
    COLOR_BLUE = [0, 0, 255]
    COLOR_RED = [255, 0, 0]
    COLOR_GREEN = [0, 255, 0]
    CHESS_REDIUS = 18
    POINT_REDIUS = 2

    SIZE = 8
    CACHE_SIZE = 150000
    boardMin = BOARD_UNIT
    boardMax = BOARD_UNIT * (SIZE + 1)              # May need to be modified later
    screen = None
    stateDict = {}
    dictCount = 0

    def __init__(self, size=8):
        self.size = size
        self.state = (np.uint64(34628173824), np.uint64(68853694464))     # (Black, White)
        self.currentPlayer = self.PLAYER_BLACK

    def deepcopy(self):
        state = GameState()
        state.size = self.size
        state.state = copy.deepcopy(self.state)
        state.currentPlayer = self.currentPlayer
        return state

    def board(self):
        return reconstruct_board(*self.state)

    def display(self):
        print(f"Current Player: {self.currentPlayer}")
        for i in range(self.size):
            print('   ', i, end='')
        print()
        for i, line in enumerate(self.board()):
            print(i, line)
        print()

    def init_draw(self):
        pygame.init()
        pygame.display.set_caption("OTHELLO")
        screen = pygame.display.set_mode((self.BOARD_UNIT * (self.size + 2), self.BOARD_UNIT * (self.size + 2)))
        self.draw(screen)
        return screen

    def draw(self, screen):
        pygame.event.pump()
        screen.fill([125, 95, 24])
        for h in range(1, self.size + 2):
            pygame.draw.line(screen, self.COLOR_BLACK, [self.boardMin, h * self.BOARD_UNIT], [self.boardMax, h * self.BOARD_UNIT], 1)
            pygame.draw.line(screen, self.COLOR_BLACK, [h * self.BOARD_UNIT, self.boardMin], [h * self.BOARD_UNIT, self.boardMax], 1)
        pygame.draw.rect(screen, self.COLOR_BLACK, [self.boardMin - 4, self.boardMin - 4, self.BOARD_UNIT * (self.size) + 9, self.BOARD_UNIT * (self.size) + 9], 3)
        color = self.COLOR_BLACK if self.currentPlayer == self.PLAYER_BLACK else self.COLOR_WHITE
        pygame.draw.circle(screen, color, [self.BOARD_UNIT / 2, self.BOARD_UNIT / 2], self.CHESS_REDIUS / 2)
        for row, line in enumerate(self.board()):
            for col, item in enumerate(line):
                if item != self.EMPTY_SQUARE:
                    color = self.COLOR_BLACK if item == self.PLAYER_BLACK else self.COLOR_WHITE
                    pos = [self.BOARD_UNIT * (col + 1) + self.BOARD_UNIT / 2, self.BOARD_UNIT * (row + 1) + self.BOARD_UNIT / 2]
                    pygame.draw.circle(screen, color, pos, self.CHESS_REDIUS)
        for (row, col), _ in self.getSuccessors():
            color = self.COLOR_BLUE if self.currentPlayer == self.PLAYER_BLACK else self.COLOR_GREEN
            pos = [self.BOARD_UNIT * (col + 1) + self.BOARD_UNIT / 2, self.BOARD_UNIT * (row + 1) + self.BOARD_UNIT / 2]
            pygame.draw.circle(screen, color, pos, self.POINT_REDIUS)
        pygame.display.flip()

    def getXMatrix(self):
        X = list()
        for line in self.board():
            l = list()
            for item in line:
                if item == self.PLAYER_BLACK:
                    l.append(1)
                elif item == self.PLAYER_WHITE:
                    l.append(-1)
                else:
                    l.append(0)
            X.append(l)
        return X

    def getOpponentPlayer(self, player):
        return self.PLAYER_BLACK if player == self.PLAYER_WHITE else self.PLAYER_WHITE

    def getNextStateNoAction(self):
        state = self.deepcopy()
        state.currentPlayer = state.getOpponentPlayer(self.currentPlayer)
        return state

    def getNextState(self, action):
        if not self.checkValidAction(action):
            return None
        successor = self.deepcopy()
        board = self.board()
        
        board[action[0]][action[1]] = self.currentPlayer
        directions = [  
            (0, 1), (0, -1), (1, 0), (-1, 0),  
            (1, 1), (1, -1), (-1, 1), (-1, -1)  
        ]  
        x, y = action
        for dx, dy in directions:  
            xi, yi = x + dx, y + dy
            s = 0
            while 0 <= xi < self.size and 0 <= yi < self.size:  
                if board[xi][yi] == self.EMPTY_SQUARE:
                    break
                if s == 0 and board[xi][yi] == self.currentPlayer:
                    break
                if board[xi][yi] != self.currentPlayer:
                    s = 1
                if s == 1 and board[xi][yi] == self.currentPlayer:
                    while (xi, yi) != action:
                        xi -= dx  
                        yi -= dy
                        board[xi][yi] = self.currentPlayer
                    break
                xi += dx  
                yi += dy
        successor.currentPlayer = self.getOpponentPlayer(self.currentPlayer)
        successor.state = construct_state(board)
        return successor

    def getSuccessors(self):
        """
        Get list of successors (action, nextState)
        NOTE: if no valid action, return []
        """
        # For debug
        if self.currentPlayer == self.PLAYER_BLACK:
            s1, s2 = self.state
        else:
            s2, s1 = self.state
        if (s1, s2) in self.stateDict:
            res = self.stateDict[(s1, s2)]
        else:
            res = calculate_successor(s1, s2)
            if self.dictCount == self.CACHE_SIZE:
                self.stateDict.popitem()
            else:
                self.dictCount += 1
            self.stateDict[(s1, s2)] = res
        successors = []
        for act, s in res:
            successor = GameState()
            if self.currentPlayer == self.PLAYER_BLACK:
                successor.state = s
            else:
                successor.state = (s[1], s[0])
            successor.currentPlayer = self.getOpponentPlayer(self.currentPlayer)
            successors.append((act, successor))
        return successors
    
    def checkValidAction(self, action: tuple):
        x, y = action
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False

        board = self.board()
        if board[x][y] != self.EMPTY_SQUARE:
            return False
        directions = [  
            (0, 1), (0, -1), (1, 0), (-1, 0),  
            (1, 1), (1, -1), (-1, 1), (-1, -1)  
        ]  
        for dx, dy in directions:  
            xi, yi = x + dx, y + dy
            s = 0
            while 0 <= xi < self.size and 0 <= yi < self.size:  
                if board[xi][yi] == self.EMPTY_SQUARE:
                    break
                if s == 0 and board[xi][yi] == self.currentPlayer:
                    break
                if board[xi][yi] != self.currentPlayer:
                    s = 1
                if s == 1 and board[xi][yi] == self.currentPlayer:
                    return True
                xi += dx  
                yi += dy
        return False 
    
    def checkGameResult(self):
        s1, s2 = self.state
        ctrBlack = bitCounts(s1)
        ctrWhite = bitCounts(s2)
        if (ctrWhite + ctrBlack == self.size * self.size) or (len(self.getSuccessors()) == 0 and len(self.getNextStateNoAction().getSuccessors()) == 0):
            if ctrBlack > ctrWhite:
                return self.RESULT_BLACK_WIN
            elif ctrWhite > ctrBlack:
                return self.RESULT_WHITE_WIN
            else:
                return self.RESULT_DRAW
        return self.RESULT_ON
    
    def isBlackWin(self):
        return self.checkGameResult() == self.RESULT_BLACK_WIN
    
    def isWhiteWin(self):
        return self.checkGameResult() == self.RESULT_WHITE_WIN
    
    def isDraw(self):
        return self.checkGameResult() == self.RESULT_DRAW
    
    def isOn(self):
        return self.checkGameResult() == self.RESULT_ON

    def isCurrentPlayerWin(self):
        return self.isBlackWin() if self.currentPlayer == self.PLAYER_BLACK else self.isWhiteWin()

    def isOpponentPlayerWin(self):
        return self.isBlackWin() if self.currentPlayer == self.PLAYER_WHITE else self.isWhiteWin()

class Agent:
    """
    An agent must define a getAction method
    """
    def getAction(self, state: GameState):
        """
        The Agent will receive a GameState and
        must return an action tuple (x, y)
        """
        raise RuntimeError("Agent function getAction must be implemented!")
    
    def update(self, oldState: GameState, oldAction, state: GameState, action):
        """
        For Agent to update its state
        """
        return

    def dataFlowUpdate(self, action: (int, int)):
        """
        For Agent to update its dataFlow
        """
        return
    
class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__(self, blackAgent:Agent, whiteAgent:Agent, size=8, display=True, gui=False):
        self.gameState = GameState(size)
        self.blackAgent = blackAgent
        self.whiteAgent = whiteAgent
        self.display = display
        self.gui = gui
        self.gameProcess = []
        self.dataFlow = None

    def setDataFlow(self, data):
        self.dataFlow = copy.deepcopy(data)

    def run(self, returnScore = False):
        """
        Main control loop for game play.
        """
        if self.display:
            self.gameState.display()
        if self.gui:
            screen = self.gameState.init_draw()
        while self.gameState.isOn():
            if len(self.gameState.getSuccessors()) == 0:
                self.gameState = self.gameState.getNextStateNoAction()
                if self.gui:
                    self.gameState.draw(screen)
                    pygame.event.clear()

            nowStep = len(self.gameProcess)
            if self.dataFlow is not None:
                self.blackAgent.dataFlowUpdate(self.dataFlow[nowStep])
                self.whiteAgent.dataFlowUpdate(self.dataFlow[nowStep])

            if self.gameState.currentPlayer == self.gameState.PLAYER_BLACK:
                action = self.blackAgent.getAction(self.gameState)
                self.gameProcess.append((self.gameState, action))
                if self.display:
                    print (f"Black takes action: {action}")
            else:
                action = self.whiteAgent.getAction(self.gameState)
                self.gameProcess.append((self.gameState, action))
                if self.display:
                    print (f"White takes action: {action}")
            nextState = self.gameState.getNextState(action)
            if nextState is None:
                raise RuntimeError(f"Invalid action {action} for player {self.gameState.currentPlayer}")
            if len(self.gameProcess) > 1:
                if nextState.currentPlayer == GameState.PLAYER_BLACK:
                    for oldGameState in reversed(self.gameProcess):
                        if oldGameState[0].currentPlayer == GameState.PLAYER_BLACK:
                            oldBlackState, oldBlackAction = oldGameState
                            break
                    self.blackAgent.update(oldBlackState, oldBlackAction, self.gameState, action)
                else:
                    for oldGameState in reversed(self.gameProcess):
                        if oldGameState[0].currentPlayer == GameState.PLAYER_WHITE:
                            oldWhiteState, oldWhiteAction = oldGameState
                            break
                    self.whiteAgent.update(oldWhiteState, oldWhiteAction, self.gameState, action)
            self.gameState = nextState
            if self.display:
                self.gameState.display()
                print ('--------------------------------------------')
            if self.gui:
                self.gameState.draw(screen)
                pygame.event.clear()
        result = self.gameState.checkGameResult()
        if self.display:
            print (result)
        import os
        pygame.quit()
        if returnScore:
            return bitCounts(self.gameState.state[0]) - bitCounts(self.gameState.state[1])
        return result
