import random
from game.game import Agent

class randomAgent(Agent):
    def getAction(self, state):
        legalActions = [item[0] for item in state.getSuccessors()]
        return random.choice(legalActions)


if __name__ == "__main__":
    from game.game import Game
    # game = Game(mouseAgent(), mouseAgent(), gui=True)
    # game.run()
