import MCTS.game as game
import MCTS.mcts as mcts
import numpy as np

GAMES_PER_DIFFICULTY = 10
MAX_GAME_MOVES = 42

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

    score = []
    for time_limit in mcts_times:
        wins = 0; ties = 0; losses = 0
        for game_number in xrange(GAMES_PER_DIFFICULTY):
            game_result = custom_vs_uct_game(player,time_limit)
            if game_result > 0:
                wins +=1
            elif game_result ==0:
                ties +=1
            else:
                losses +=1
        if verbose:
            print "For %.2f-second UCT, won %d, tied %d, lost %d" % (time_limit,wins,ties,losses)
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

def main():
    test_player = game.ComputerPlayer('mcts',mcts.uct, .7)
    score = test_policy_vs_MCTS(test_player, verbose=True)
    print "Score vector: ", score


if __name__ == "__main__":
    main()
