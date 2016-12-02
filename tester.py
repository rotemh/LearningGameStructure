import MCTS.game as game
import MCTS.mcts as mcts
import RL.RLAgent as RLAgent
import numpy as np

from Queue import Queue
from threading import Thread

GAMES_PER_DIFFICULTY = 10
MAX_GAME_MOVES = 42

global_queue = Queue(maxsize = 0)
results_q = Queue(maxsize = 0)

def worker(q, rq):
    while True:
        player, time_limit = q.get()
        print "Running game - %s %s " % (str(player), str(time_limit))
        game_result = custom_vs_uct_game(player,time_limit)
        rq.put(game_result)
        q.task_done()

def test_policy_vs_MCTS(player, mcts_times=None,verbose=False):
    """
    Takes in a MCTS/game.py Player instance, and compares it with MCTS players of varying quality.
    Returns a score vector of length len(mcts_times) with win percentages from 0 to 1 for each difficulty
    """
    if mcts_times == None:
        mcts_times = [.01,.05,.2,.5,1,2]
    if verbose:
        total_time = np.sum(mcts_times)*MAX_GAME_MOVES*GAMES_PER_DIFFICULTY
        print "Estimated testing time: %d seconds" % total_time

    for t in xrange(8):
        t = Thread(target=worker, args=(global_queue,results_q))
        t.setDaemon(True)
        t.start()
    
    score = []
    for time_limit in mcts_times:
        wins = 0; ties = 0; losses = 0
        for game_number in xrange(GAMES_PER_DIFFICULTY):
            global_queue.put((player,time_limit))
        global_queue.join()

        # Drain results
        while not results_q.empty():
            game_result = results_q.get()
            if game_result > 0:
                wins +=1
            elif game_result ==0:
                ties +=1
            else:
                losses +=1
        if verbose:
            print "For %.2f-second UCT, won %d, tied %d, lost %d" \
                      % (time_limit,wins,ties,losses)
        score.append(float(wins)/GAMES_PER_DIFFICULTY)

    return score

def custom_vs_uct_game(player1,uct_time_limit=1.0):
    """
    Run a custom player versus a uct MCTS player

    Returns 1 if custom player won, 0 if tie, -1 if MCTS won
    """
    # The higher the time_limit, the better the players will perform
    board = game.ConnectFourBoard()
    player2 = game.ComputerPlayer('mcts', mcts.uct, uct_time_limit)

    sim = game.Simulation(board, player1, player2)
    result = sim.run(visualize=False,state_action_history=False)
    return result

def test_qValues(player, verbose=False):
    """
    Takes in a MCTS/game.py RLPlayer instance, and assesses how it assigns Q-Values to certain 
    board layouts. 
    """
    assert(type(player) is RLPlayer)
    r = ConnectFourBoard.RED
    b = ConnectFourBoard.BLACK
    e = ConnectFourBoard.EMPTY

    if verbose:
        print("Testing Q Values generated:")
        print("1. Obvious Win Scenarios should hava q-value of 1.")

    #A. Obvious Win Scenarios should have a q-value of 1
    
    #Case 1: unseen case in MCTS, red column of three, and black column of three, red turn
    if verbose:
        print("\t Testing Case 1")
    board = game.ConnectFourBoard()
    for i in xrange(3):
        board.state[0][i] = r
        board.state[6][i] = b
    board.turn = r
    assert(player.get_q_value(board) == 1)

    #Case 2: three horizontal red tokens stacked on three horizontal black tokens, black turn
    if verbose:
        print("\t Testing Case 2")
    board = game.ConnectFourBoard()
    for i in xrange(2, 5):
        board.state[i][1] = r
        board.state[i][0] = b
    board.turn = b
    assert(player.get_q_value(board) == 1)

    #Case 3: 
    if verbose:
        print("\t Testing Case 3")
    board = game.ConnectFourBoard()
    for i in xrange(3):
        board.state[3][i] = r
    for i in [1, 2, 4]:
        board.state[i][0] = b
    board.turn = r
    assert(player.get_q_value(board) == 1)


    #Case 4: more crowded:
    if verbose:
        print("\t Testing Case 4")
    board = game.ConnectFourBoard()
    for i in xrange(3):
        board.state[3][i] = r
    redPositions = [(0,0), (1,0), (1,1), (2, 2), (2, 3), (4, 0), (4,2), (5, 0), (5,1)]
    for (col, row) in redPositions:
        board.state[col][row] = r 
    blackPositions = [(2, 0), (2, 1), (4, 1), (4, 3)]
    for (col, row) in blackPositions:
        board.state[col][row] = b
    assert(player.get_q_value(board) == 1)

    #B. Compare Q-Values
    if verbose:
        print("2. Comparing Q-Values.")
    boardA = game.ConnectFourBoard()
    boardB = game.ConnectFourBoard()
    blackPositions = [(1,0), (2, 0)]
    redPositions = [(1,1), (2, 1), (3, 0)]
    for (col, row) in blackPositions:
        boardA.state[col][row] = b 
        boardB.state[col][row] = b
    for (col, row) in redPositions:
        boardA.state[col][row] = r 
        boardB.state[col][row] = r
    boardA.state[0][0] = b 
    boardB.state[5][0] = b #less trapped, should have higher Q
    assert(player.get_q_value(boardA) < player.get_q_value(boardB))

    #C. Creating a Trap, 
    if verbose:
        print("3. Creating a Trap.")
        print("\t Testing Case 1")
    #Case 1: look forward two.
    board = game.ConnectFourBoard()
    for i in xrange(2, 4):
        board.state[i][1] = r
        board.state[i][0] = b
    board.turn = b 
    action = player.choose_action(board)
    assert(action.col == 4 or action.col == 1) 

    #Case 2: Look Forward three
    if verbose:
        print("\t Testing Case 2")
    board = game.ConnectFourBoard()
    blackPositions = [(0,0), (1,1), (2,0), (2,1), (3,1), (5,3), (6,0), (6,2)]
    redPositions = [(0,1), (1,0), (3,0), (3,2), (5,0), (5,1), (5,2), (6,1)]
    for (col, row) in blackPositions:
        board.state[col][row] = b 
    for (col, row) in redPositions:
        board.state[col][row] = r 
    board.turn = b 
    action = player.choose_action(board)
    assert(action.col == 4)

    #D. Don't place a token that creates a trap in your favor, but let's your opponent win on the next turn
    if verbose:
        print("4. Don't let opponent win, by focusing only on your win condition.")
    board = game.ConnectFourBoard()
    blackPositions = [(0,0), (1,1), (3,1), (3,3), (3,5), (4,0), (4,1), (4,2), (6,0)]
    redPositions = [(1,0), (3,0), (3,2), (3,4), (4,3), (6,1)]
    for (col, row) in blackPositions:
        board.state[col][row] = b 
    for (col, row) in redPositions:
        board.state[col][row] = r
    board.turn = b 
    action = player.choose_action(board)
    assert(action.col != 2)

    return 

def main():
    test_player = game.ComputerPlayer('mcts',mcts.uct, .7)
    score = test_policy_vs_MCTS(test_player, verbose=True)
    print "Score vector: ", score
    test_qValues(player, True)


if __name__ == "__main__":
    main()
