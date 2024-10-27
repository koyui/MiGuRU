from game.game import Agent

class KeyboardAgent(Agent):
    def getAction(self, state):
        x, y = map(int, input("Your next step: ").split(' '))
        return (x, y)

if __name__ == "__main__":
    from game.game import Game
    game = Game(KeyboardAgent(), KeyboardAgent())
    game.run()