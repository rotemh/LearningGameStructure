from MCTS.sim import *
from RL.SupervisedPolicy import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
from MCTS.mcts import *
from amcts import *
import numpy as np
import scipy.io as sio
from tester import test_policy_vs_MCTS

MAX_GAME_MOVES = 42

def create_players():
    """
    Construct the set of player types we'd like to test our algorithm against.
    """
    policy_agent = SupervisedPolicyAgent((144,144,3),7)
    policy_agent.load_train_results()
    value_agent = SupervisedValueNetworkAgent((144,144,3))
    value_agent.load_train_results()
    def random_algo(board):
        actions = board.get_legal_actions()
        action = np.random.choice(list(actions))
        return action

    policy_player = PolicyPlayer('policy_only', policy_agent)
    value_player = ValuePlayer('value_only', value_agent)
    random_player = ComputerPlayer('random', random_algo)
    uct_p05_player =  ComputerPlayer('UCT_p05s', mcts.uct, 0.05)
    uct_p3_player =  ComputerPlayer('UCT_p3s', mcts.uct, 0.3)
    uct_1s_player = ComputerPlayer('UCT_1s', mcts.uct, 1)
    uct_3s_player = ComputerPlayer('UCT_3s', mcts.uct, 3)
    fast_amcts_player = AMCTSPlayer('AMCTS_1s', 1, policy_agent=policy_agent, value_agent=value_agent)
    amcts_player_policy_only = AMCTSPlayer('AMCTS_1s_policy_only', 1, policy_agent=policy_agent)
    amcts_player_value_only = AMCTSPlayer('AMCTS_1s_value_only', 1, value_agent=value_agent)
    slow_amcts_player = AMCTSPlayer('AMCTS_3s', 3, policy_agent=policy_agent, value_agent=value_agent)

    return [policy_player, value_player, random_player, uct_1s_player, \
        uct_p05_player, uct_p3_player, uct_1s_player, uct_3s_player, \
        fast_amcts_player, amcts_player_value_only, amcts_player_policy_only, slow_amcts_player]

def compare_players(player1, player2, verbose=False, num_games = 5):
    """
    Takes in two MCTS/game.py Player instance, and compares them with each other
    Returns two score vectors from 0 to 1 of % of games starting player won 
    given first player started, and then given second started

    num_games - the number of games played through for each side
    """
    if verbose:
        time_per_move = 0
        try:
            if player1.time_limit != None:
                time_per_move += player1.time_limit
        except AttributeError:
            pass
        try:
            if player2.time_limit != None:
                time_per_move += player2.time_limit
        except AttributeError:
            pass
        time_per_move = time_per_move/2. # take the average
        total_time = 2*num_games*MAX_GAME_MOVES*time_per_move
        print "Testing %s vs. %s" %(player1.name, player2.name)
        print "Max testing time: %d seconds" % total_time

    first_start_score = 0
    for game in xrange(num_games):
        episode = generate_two_player_game(player1, player2)
        win_player_id = np.argmax( episode[-1]['reward'] )
        if verbose:
            winner_player_color = lambda x: player1.name if x==0 else player2.name
            print "Winner player is " + winner_player_color(win_player_id)
        if np.sum(episode[-1]['reward'] == 0):
            pass
        elif win_player_id == 0:
            first_start_score+=1
        else:
            pass

    second_start_score = 0
    for game in xrange(num_games):
        episode = generate_two_player_game(player2, player1)
        win_player_id = np.argmax( episode[-1]['reward'] )
        if verbose:
            winner_player_color = lambda x: player2.name if x==0 else player1.name
            print "Winner player is " + winner_player_color(win_player_id)
        if np.sum(episode[-1]['reward'] == 0):
            pass
        elif win_player_id == 0:
            second_start_score+=1
        else:
            pass
    return (first_start_score/float(num_games), second_start_score/float(num_games))


def main():
    policy_player, value_player, random_player, uct_1s_player, \
        uct_p05_player, uct_p3_player, uct_1s_player, uct_3s_player, \
        fast_amcts_player, amcts_player_value_only, amcts_player_policy_only, slow_amcts_player = create_players()

    win_pct_1, win_pct_2 = compare_players(uct_p05_player, policy_player, verbose=True, num_games=20)
    print "first start: %.3f, second start: %.3f" %(win_pct_1, win_pct_2)



if __name__ == "__main__":
    main()
