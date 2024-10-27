import random
import time
from typing import Tuple
from game.game import Agent
from copy import deepcopy

class dataPlayingAgent(Agent):

    def dataFlowUpdate(self, action: Tuple[int, int]):
        self.nowAction = action
        return

    def getAction(self, state):
        return self.nowAction

if __name__ == "__main__":
    from game.game import Game
    # game = Game(mouseAgent(), mouseAgent(), gui=True)
    # game.run()