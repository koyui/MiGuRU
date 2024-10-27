from typing import Tuple
import numpy as np
from game.game import Game, GameState, Agent
from game import utils
from tqdm import trange
from dataset.readData import read_all

class countDataPlayingAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.actions = []

    def dataFlowUpdate(self, action: Tuple[int, int]):
        self.nowAction = action
        return

    def getAction(self, state):
        self.actions.append(self.nowAction)
        return self.nowAction

dataFlow = read_all()
print("Data loading done!")

board = np.zeros((8, 8), np.int32)
for i in trange(50000):
    data = dataFlow[i]
    AgentB = countDataPlayingAgent()
    AgentW = countDataPlayingAgent()
    game = Game(AgentB, AgentW, display=False, gui=False)
    game.setDataFlow(data)
    ret = game.run()
    if ret == GameState.RESULT_WHITE_WIN:
        for r, c in utils.matrixAsList(game.gameState.board(), utils.playerWhite()):
            board[r, c] += 1
        for r, c in utils.matrixAsList(game.gameState.board(), utils.playerBlack()):
            board[r, c] -= 1
    elif ret == GameState.RESULT_BLACK_WIN:
        for r, c in utils.matrixAsList(game.gameState.board(), utils.playerWhite()):
            board[r, c] -= 1
        for r, c in utils.matrixAsList(game.gameState.board(), utils.playerBlack()):
            board[r, c] += 1
    if i % 10000 == 0:
        print(board)
print(board)

# matrix of action(winner) - action(loser)
# [[13669   394  1400   739   719   650   453 14141]
#  [  596 -7199  -594  -408   712  -394 -7665   426]
#  [  700   689  1017   797   826  2051  -557  1541]
#  [  864  -452   502     0     0  1978  -600   956]
#  [ 1110 -1242   345     0     0 -1671   326  1000]
#  [  571  -617  1643  2577   237  1837   642   798]
#  [  -15 -7359  -669   879  -142  -471 -7784   521]
#  [14101   813  1016   726   956  1244   360 14496]]
