from naiveAgents.keyboardAgent import KeyboardAgent
from naiveAgents.mouseAgent import mouseAgent
from minimax.minimaxAgent import MinimaxAgent, positionHeuristic, mobilityHeuristic, stableHeuristic, parityHeuristic, weightedEvaluationFunction
from tqdm import trange
from naiveAgents.randomAgent import randomAgent
from reinforcement.learningAgents import ApproximateQAgent
# from reinforcement.learningAgents import NNQAgent
from reinforcement.featureExtractors import SimpleExtractor
from game.game import GameState

def dataLearningTestRandom():
    from game.game import Game

    agentB = randomAgent()
    agentW = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0, epsilon=0, discount=1,
                               weights_path='reinforcement/final_weight/e15000_dataTraining.json')
    w_win = 0
    b_win = 0
    draw = 0
    for _ in trange(200):
        game = Game(agentB, agentW, display=False, gui=True)
        ret = game.run()
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
    print(f"\nw win: {w_win}/200")

def playerTest():
    from game.game import Game

    agentB = mouseAgent()
    agentW = mouseAgent()
    game = Game(agentB, agentW, display=False, gui=True)
    ret = game.run()
    if ret == GameState.RESULT_BLACK_WIN:
        print("B wins")
    else:
        print("W wins")


def gameBasedMinimaxTestRandom():
    from game.game import Game

    agentB = randomAgent()
    agentW = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0, epsilon=0, discount=1,
                               weights_path='reinforcement/final_weight/e3700_minimaxTest.json')
    w_win = 0
    b_win = 0
    draw = 0
    for _ in trange(200):
        game = Game(agentB, agentW, display=False, gui=True)
        ret = game.run()
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
    print(f"\nw win: {w_win}/200")

def MinimaxTestRandom():
    from game.game import Game

    agentB = randomAgent()
    agentW = MinimaxAgent(2, weightedEvaluationFunction, "W")
    w_win = 0
    b_win = 0
    draw = 0
    for _ in trange(100):
        game = Game(agentB, agentW, display=False, gui=True)
        ret = game.run()
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
    print(f"\nw win: {w_win}/100")


def randomBasedTestMinimax():
    from game.game import Game

    agentB = MinimaxAgent(1, weightedEvaluationFunction, "B", test=True, epsilon=0.1)
    agentW = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0, epsilon=0.1, discount=1,
                               weights_path='reinforcement/final_weight/e8700_randomTest.json')
    w_win = 0
    b_win = 0
    draw = 0
    for _ in trange(100):
        game = Game(agentB, agentW, display=False, gui=True)
        ret = game.run()
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
    print(f"\nw win: {w_win}/100")

def MinimaxBasedTestDataBased():
    from game.game import Game

    agentB = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0, epsilon=0.2, discount=1,
                               weights_path='reinforcement/final_weight/e3700_minimaxTest.json')
    agentW = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0, epsilon=0.2, discount=1,
                               weights_path='reinforcement/final_weight/e15000_dataTraining.json')
    w_win = 0
    b_win = 0
    draw = 0
    for _ in trange(200):
        game = Game(agentB, agentW, display=False, gui=True)
        ret = game.run()
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
    print(f"\nw win: {w_win}/200")

def Minimax2TestMinimax1():
    from game.game import Game

    agentB = MinimaxAgent(1, weightedEvaluationFunction, "B", test=True, epsilon=0.05)
    agentW = MinimaxAgent(2, weightedEvaluationFunction, "W", test=True, epsilon=0.05)
    w_win = 0
    b_win = 0
    draw = 0
    for _ in trange(100):
        game = Game(agentB, agentW, display=False, gui=True)
        ret = game.run()
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
    print(f"\nw win: {w_win}/100")

if __name__ == "__main__":
    dataLearningTestRandom()