import pygame
from game.game import Agent

class mouseAgent(Agent):
    def getAction(self, state):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = event.pos
                    row = (y - state.BOARD_UNIT) // state.BOARD_UNIT
                    col = (x - state.BOARD_UNIT) // state.BOARD_UNIT
                    if state.checkValidAction((row, col)):
                        return (row, col)

if __name__ == "__main__":
    from game.game import Game
    game = Game(mouseAgent(), mouseAgent(), gui=True)
    game.run()