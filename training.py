from tqdm import trange
from game.game import Game, GameState
from naiveAgents.randomAgent import randomAgent
from naiveAgents.dataPlayingAgent import dataPlayingAgent
from reinforcement.learningAgents import ApproximateQAgent, SimpleExtractor
 
def reinforcementTraining():
    epoch = 10000
    # agentB = MinimaxAgent(1, weightedEvaluationFunction, utils.playerBlack())
    agentB = randomAgent()
    agentW = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0.2, epsilon=0.3, discount=1, weights_path=None)

    # train
    w_win = 0
    b_win = 0
    draw = 0
    # fileName = 'minimaxTest'
    fileName = 'randomTest'
    with open(f"reinforcement/weights/{fileName}.txt", 'w') as f:
        f.write('\n')
    for i in trange(epoch):
        game = Game(agentB, agentW, display=False, gui=False)
        ret = game.run()
        agentW.epsilon *= 0.999
        agentW.alpha *= 0.9992
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
        if i % 100 == 0:
            print(f"\nw win: {w_win}/100")
            print(agentW.getWeights())
            agentW.save_weights(f"reinforcement/weights/e{i}_{fileName}.json")
            print(f'Now epsilon:{agentW.epsilon}')
            print(f'Now alpha:{agentW.alpha}')
            with open(f"reinforcement/weights/{fileName}.txt", 'a+') as f:
                f.write(f"\nw win: {w_win}/100\n")
                f.write(str(agentW.getWeights()) + '\n')
                f.write(f'Now epsilon:{agentW.epsilon}\n')
                f.write(f'Save weight at e{i}_{fileName}.json\n')
                f.write(f'Now alpha:{agentW.alpha}\n\n')
            w_win = 0

def dataTraining():
    from dataset.readData import read_all
    dataFlow = read_all()
    print("Data loading done!")
    agentB = dataPlayingAgent()
    # agentW = dataPlayingAgent()
    agentW = ApproximateQAgent(extractor=SimpleExtractor(), alpha=0.2, epsilon=0.3, discount=1, weights_path=None,
                               use_data=True)
    # train
    w_win = 0
    b_win = 0
    draw = 0
    fileName = 'dataTraining'
    with open(f"reinforcement/weights/{fileName}.txt", 'w') as f:
        f.write('\n')
    for i in trange(50000):
        data = dataFlow[i]
        game = Game(agentB, agentW, display=False, gui=False)
        game.setDataFlow(data)
        ret = game.run()
        agentW.epsilon *= 0.999
        if ret == GameState.RESULT_WHITE_WIN:
            w_win += 1
        elif ret == GameState.RESULT_BLACK_WIN:
            b_win += 1
        else:
            draw += 1
        if i % 200 == 0:
            print(f"\nw win: {w_win}/200")
            print(agentW.getWeights())
            agentW.save_weights(f"reinforcement/weights/e{i}_{fileName}.json")
            print(f'Now epsilon:{agentW.epsilon}')
            print(f'Now alpha:{agentW.alpha}')
            with open(f"reinforcement/weights/{fileName}.txt", 'a+') as f:
                f.write(f"\nw win: {w_win}/200\n")
                f.write(str(agentW.getWeights()) + '\n')
                f.write(f'Now epsilon:{agentW.epsilon}\n')
                f.write(f'Save weight at e{i}_{fileName}.json\n')
                f.write(f'Now alpha:{agentW.alpha}\n\n')
            w_win = 0

if __name__ == "__main__":
    # reinforcementTraining()
    dataTraining()