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

from game import utils
from game.game import GameState
from stateSupport.stateSupport import bitCounts
from minimax.minimaxAgent import parityHeuristic, stableHeuristic, mobilityHeuristic, positionHeuristic

class FeatureExtractor:
    def getFeatures(self, state: GameState, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        utils.raiseNotDefined()

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for:
    - bias
    # - distance from center
    # - #player's chess
    # - #opponent's chess
    - Heuristics from minimax
    """

    def getFeatures(self, state, action):
        features = utils.Counter()

        features["bias"] = 1.0

        board_size = state.size

        # center_point = (board_size - 1) / 2
        # features["distance-from-center"] = (abs(center_point-x) + abs(center_point-y)) / (board_size ** 2)

        state_in_bits = state.state[0] if state.currentPlayer == utils.playerBlack() else state.state[1]
        features["#player's chess"] = bitCounts(state_in_bits) / (board_size ** 2)

        # op_state_in_bits = state.state[1] if state.currentPlayer == utils.playerBlack() else state.state[0]
        # features["#opponent's chess"] = bitCounts(op_state_in_bits) / (board_size ** 2)

        features[str(action)] = 1.0

        features["parityHeuristic"] = parityHeuristic(state, state.currentPlayer)
        features["stableHeuristic"] = stableHeuristic(state, state.currentPlayer) / (board_size ** 2)
        features["mobilityHeuristic"] = mobilityHeuristic(state, state.currentPlayer) / (board_size ** 2)
        features["positionHeuristic"] = positionHeuristic(state, state.currentPlayer) / (board_size ** 2) / 10

        # features["op-parityHeuristic"] = parityHeuristic(nextState, nextState.currentPlayer)
        # features["op-stableHeuristic"] = stableHeuristic(nextState, nextState.currentPlayer) / (board_size ** 2)
        # features["op-mobilityHeuristic"] = mobilityHeuristic(nextState, nextState.currentPlayer) / (board_size ** 2)
        # features["op-positionHeuristic"] = positionHeuristic(nextState, nextState.currentPlayer) / (board_size ** 2) / 10

        features.divideAll(10)
        return features
