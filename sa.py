import math
from random import random
from minimax.minimaxAgent import MinimaxAgent, weightedEvaluationFunction
from game.utils import playerBlack, playerWhite
from game.game import Game, GameState
from matplotlib import pyplot as plt

class SA:
    def __init__(self, minimaxDepth=3, iter=1, T0=1, Tf=1e-5, alpha=0.99):
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.depth = minimaxDepth
        self.ws = []
        self.Ts = []

    def generate_new(self, weights):
        for i in range(len(weights)):
            weights[i] += self.T * (random() * 2 - 1)
            if weights[i] > 1.0:
                weights[i] = 1.0
            elif weights[i] < -1.0:
                weights[i] = -1.0
        return weights

    def cmp(self, bestWeights, newWeights):
        score = 0
        bestAgent = MinimaxAgent(self.depth, lambda g, a: weightedEvaluationFunction(g, a, weights=bestWeights), playerBlack())
        newAgent = MinimaxAgent(self.depth, lambda g, a: weightedEvaluationFunction(g, a, weights=newWeights), playerWhite())
        game = Game(bestAgent, newAgent, display=False)
        score -= game.run(returnScore=True)
        newAgent = MinimaxAgent(self.depth, lambda g, a: weightedEvaluationFunction(g, a, weights=newWeights), playerBlack())
        bestAgent = MinimaxAgent(self.depth, lambda g, a: weightedEvaluationFunction(g, a, weights=bestWeights), playerWhite())
        game = Game(newAgent, bestAgent, display=False)
        score += game.run(returnScore=True)
        return score / 64 / 64

    def Metrospolis(self, delta):
        return delta > 0 or random() < math.exp(delta / self.T)

    def run(self, initWeights):
        bestWeights = initWeights
        ctr = 0
        while self.T > self.Tf:       
            for _ in range(self.iter): 
                newWeights = self.generate_new(bestWeights)
                if self.Metrospolis(self.cmp(bestWeights, newWeights)):
                    bestWeights = newWeights
                self.ws.append(bestWeights.copy())
                self.Ts.append(self.T)
            self.T = self.T * self.alpha
            ctr += 1
            if ctr % 10 == 0:
                print (self.T, bestWeights)
        return bestWeights
    
    def plot(self):
        plt.plot(range(len(self.ws)), [t[0] for t in self.ws], "r")
        plt.plot(range(len(self.ws)), [t[1] for t in self.ws], "g")
        plt.plot(range(len(self.ws)), [t[2] for t in self.ws], "b")
        plt.plot(range(len(self.ws)), [t[3] for t in self.ws], "y")
        plt.savefig("figs/sa.png")

depth = 1

# experienceWeights = [0.8575068335017162, -0.29288926993945635, 0.46950848970104814, 0.10602256740111472]
experienceWeights = [0.5, 0.5, 0.5, 0.5]
sa = SA(minimaxDepth=depth)
bestWeights = sa.run(experienceWeights)
sa.plot()

bestAgent = MinimaxAgent(depth, lambda g, a: weightedEvaluationFunction(g, a, weights=bestWeights), playerBlack())
myAgent = MinimaxAgent(depth, lambda g, a: weightedEvaluationFunction(g, a, weights=experienceWeights), playerWhite())
game = Game(bestAgent, myAgent, display=False)
# print ("new weight win!" if game.run() == GameState.RESULT_BLACK_WIN else "new weight lose")

myAgent = MinimaxAgent(depth, lambda g, a: weightedEvaluationFunction(g, a, weights=experienceWeights), playerBlack())
bestAgent = MinimaxAgent(depth, lambda g, a: weightedEvaluationFunction(g, a, weights=bestWeights), playerWhite())
game = Game(myAgent, bestAgent, display=False)
# print ("new weight win!" if game.run() == GameState.RESULT_WHITE_WIN else "new weight lose")

print(bestWeights)