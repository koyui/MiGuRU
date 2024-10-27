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
import os
import json
import random
from pathlib import Path
import time
from game.game import Agent, GameState
from game import utils
from reinforcement.featureExtractors import FeatureExtractor, SimpleExtractor

# Used for debugger
from stateSupport.stateSupport import uint64_2_matrix

class QLearningAgent(Agent):
    def __init__(self, extractor: FeatureExtractor, alpha, epsilon, discount, use_data=False):
        """
        extractor   - with function extractor.getFeatures(state, action)
        alpha    - learning rate
        epsilon  - exploration rate
        discount    - discount factor
        """
        self.extractor = extractor
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(discount)
        self.values = utils.Counter()
        self.weights = None
        self.use_data = use_data
        self.nowAction = None

    def dataFlowUpdate(self, action: (int, int)):
        self.nowAction = action
        return

    def update(self, oldState, oldAction, state, action):
        """
            oldState --- oldAction   state --- action ---> nxtState
            nxtState.currentPlayer == oldState.currentPlayer
        """
        nextState = state.getNextState(action)
        self._update(oldState, oldAction, nextState, nextState.isCurrentPlayerWin() - nextState.isOpponentPlayerWin())

    def _update(self, state, action, nextState, reward: float):
        """
           update weights based on transition
        """
        raise RuntimeError("Not Implement")

    def getPolicy(self, state):
        """
        policy(s) = arg_max_{a in actions} Q(s,a)
        """
        return self._computeActionFromQValues(state)

    def getWeights(self):
        return self.weights
    
    def getAction(self, state):
        """
          state: can call state.getLegalActions()

          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise. If there are
          no legal actions, return None.
        """
        if self.use_data:
            return self.nowAction
        else:
            legalActions = self._getLegalActions(state)
            action = None
            if utils.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self._computeActionFromQValues(state)
            return action
    
    def _getValue(self, state):
        """
        V(s) = max_{a in actions} Q(s,a)
        """
        return self._computeValueFromQValues(state)

    def _getQValue(self, state, action):
        raise RuntimeError("Not Implement")

    def _computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
          If there are no legal actions, return a value of 0.0.
        """
        actions = self._getLegalActions(state)
        value = float("-inf")
        for action in actions:
            value = max(value, self._getQValue(state, action))
        if len(actions) == 0:
            value = 0.0
        return value

    def _computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  If there
          are no legal actions, return None.
        """
        actions = self._getLegalActions(state)
        best_action = None
        value = float("-inf")
        for action in actions:
            v = self._getQValue(state, action)
            if best_action is None or value < v:
                value = max(value, v)
                best_action = action
        return best_action

    def _getLegalActions(self, state: GameState):
        legalActions = [item[0] for item in state.getSuccessors()]
        return legalActions

class ApproximateQAgent(QLearningAgent):
    def __init__(self, extractor=SimpleExtractor(), alpha=0.2, epsilon=0.05, discount=0.99, weights_path=None, use_data=False):
        """
        extractor   - with function extractor.getFeatures(state, action)
        alpha    - learning rate
        epsilon  - exploration rate
        discount    - discount factor
        """
        QLearningAgent.__init__(self, extractor, alpha, epsilon, discount, use_data)
        self.weights = utils.Counter()
        if weights_path is not None:
            with open(weights_path, 'r') as file:
                weights = json.load(file)
            for k, v in weights.items():
                self.weights[k] = v
            print (f"Load weights at {weights_path}.")

    def _update(self, state, action, nextState, reward: float):
        """
           update weights based on transition
        """
        # assert state.currentPlayer == nextState.currentPlayer == GameState.PLAYER_WHITE
        diff = (reward + self.discount * self._getValue(nextState)) - self._getQValue(state, action)
        features = self.extractor.getFeatures(state, action)
        for key in features:
            self.weights[key] += self.alpha * diff * features[key]

    def _getQValue(self, state, action):
        """
          return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        Qvalue = 0
        weights = self.getWeights()
        features = self.extractor.getFeatures(state, action)
        for key in features:
            Qvalue += weights[key] * features[key]
        return Qvalue

    def save_weights(self, weights_path=None):
        if weights_path is None:
            t = time.localtime()
            weights_path = Path("reinforcement") / Path("weights") / Path(f"weights_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.json")
        weights_path = Path(weights_path)
        os.makedirs(weights_path.parent, exist_ok=True)
        with open(weights_path, 'w') as file: 
            json.dump(dict(self.weights), file)
        print (f"Save weights at {weights_path}.")