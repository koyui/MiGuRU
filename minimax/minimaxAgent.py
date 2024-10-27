# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game.game import GameState, Agent
import numpy as np
from game.utils import matrixAsList
from game import utils
import random

evaluationDict = {}
dictCount = 0
CACHE_SIZE = 150000

class MinimaxAgent(Agent):
    """
    minimax agent
    """
    INF = float("inf")

    def __init__(self, depth, evaluationFunction, whichAgent, test=False, epsilon=0):
        self.depth = depth
        self.evaluationFunction = evaluationFunction
        self.whichAgent = whichAgent
        self.test = test
        self.epsilon = epsilon

    def getAction(self, gameState: GameState):
        def max_lv(gameState, depth, alpha, beta):
            if depth > self.depth or not gameState.isOn():
                return self.evaluationFunction(gameState, self.whichAgent)
            best_value = -self.INF
            successors = gameState.getSuccessors()
            if len(successors) == 0:
                nextState = gameState.getNextStateNoAction()
                best_value = max(best_value, min_lv(nextState, depth, alpha, beta))
                alpha = max(alpha, best_value)
                return best_value
            for action, nextState in successors:
                best_value = max(best_value, min_lv(nextState, depth, alpha, beta))
                if best_value > beta:
                    return best_value
                alpha = max(alpha, best_value)
            return best_value
    
        def min_lv(gameState, depth, alpha, beta):
            if not gameState.isOn():
                return self.evaluationFunction(gameState, self.whichAgent)
            best_value = self.INF
            successors = gameState.getSuccessors()
            if len(successors) == 0:
                nextState = gameState.getNextStateNoAction()
                best_value = min(best_value, max_lv(nextState, depth + 1, alpha, beta))
                beta = min(beta, best_value)
                return best_value
            for action, nextState in successors:
                best_value = min(best_value, max_lv(nextState, depth + 1, alpha, beta))
                if best_value < alpha:
                    return best_value
                beta = min(beta, best_value)
            return best_value

        if self.test:
            legalActions = gameState.getSuccessors()
            if utils.flipCoin(self.epsilon):
                return random.choice(legalActions)[0]

        best_action = (-1, -1)
        best_value = -self.INF
        alpha = -self.INF
        beta = self.INF
        for action, state in gameState.getSuccessors():
            value = min_lv(state, 1, alpha, beta)
            if value >= best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
        return best_action

def positionHeuristic(gameState:GameState, whichAgent):
    W = np.array([[500,-25,10,5,5,10,-25,500],
                [-25,-45,1,1,1,1,-45,-25],
                [10,1,3,2,2,3,1,10],
                [5,1,2,1,1,2,1,5],
                [5,1,2,1,1,2,1,5],
                [10,1,3,2,2,3,1,10],
                [-25,-45,1,1,1,1,-45,-25],
                [500,-25,10,5,5,10,-25,500]])
    X = np.array(gameState.getXMatrix())
    if whichAgent == gameState.PLAYER_WHITE:
        X = -X
    return int(sum(sum(X * W)))

def mobilityHeuristic(gameState:GameState, whichAgent):
    return len(gameState.getSuccessors())

def stableHeuristic(gameState:GameState, whichAgent):
    def calcStable(color):
        UNDETERMINED = False
        STABLE = True
        stableMap = [[UNDETERMINED for _ in range(gameState.size)] for _ in range(gameState.size)]
        def check_pos_stable(x, y):
            if x < 0 or x >= gameState.size or y < 0 or y >= gameState.size:
                return True
            return stableMap[x][y]
        def check_dir_stable(x, y, dx, dy):
            return check_pos_stable(x + dx, y + dy) or check_pos_stable(x - dx, y - dy)
        from queue import Queue
        q = Queue()
        boardMax = gameState.size - 1
        board = gameState.board()
        if board[0][0] == color:
            q.put((0, 0))
        if board[0][boardMax] == color:
            q.put((0, boardMax))
        if board[boardMax][0] == color:
            q.put((boardMax, 0))
        if board[boardMax][boardMax] == color:
            q.put((boardMax, boardMax))
        stable_count = 0
        while not q.empty():
            x, y = q.get()
            if board[x][y] != color:
                continue
            if stableMap[x][y] == STABLE:
                continue
            flag = False
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                if not check_dir_stable(x, y, dx, dy):
                    flag = True
                    break
            if flag:
                continue
            stableMap[x][y] = STABLE
            stable_count += 1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:  
                nx, ny = x + dx, y + dy  
                if 0 <= nx < gameState.size and 0 <= ny < gameState.size and board[nx][ny] == color:
                    q.put((nx, ny))
        return stable_count
    return calcStable(whichAgent) - calcStable(gameState.getOpponentPlayer(whichAgent))

def parityHeuristic(gameState:GameState, whichAgent):
    return 1 if (len(matrixAsList(gameState.board(), gameState.EMPTY_SQUARE)) & 1) else -1

def weightedEvaluationFunction(gameState:GameState, whichAgent, weights=[0.8575068335017162, -0.29288926993945635, 0.46950848970104814, 0.10602256740111472]):
    if gameState.isBlackWin():
        return MinimaxAgent.INF if whichAgent == gameState.PLAYER_BLACK else -MinimaxAgent.INF
    elif gameState.isWhiteWin():
        return MinimaxAgent.INF if whichAgent == gameState.PLAYER_WHITE else -MinimaxAgent.INF
    global evaluationDict, dictCount, CACHE_SIZE
    if gameState.state in evaluationDict:
        return evaluationDict[gameState.state]
    else:
        if dictCount == CACHE_SIZE:
            evaluationDict.popitem()
            dictCount -= 1
        res = weights[0] * positionHeuristic(gameState, whichAgent) \
            + weights[1] * mobilityHeuristic(gameState, whichAgent) \
            + weights[2] * stableHeuristic(gameState, whichAgent) \
            + weights[3] * parityHeuristic(gameState, whichAgent)
        evaluationDict[gameState.state] = res
        dictCount += 1
        return res
