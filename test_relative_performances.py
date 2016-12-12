from MCTS.sim import *
from RL.SupervisedPolicy import *
from RL.SupervisedValueNetwork import *
from MCTS.game import * 
from MCTS.mcts import *
from amcts import *
import numpy as np
import scipy.io as sio
from tester import test_policy_vs_MCTS
import csv

MAX_GAME_MOVES = 42
import tensorflow as tf

tf.python.control_flow_ops = tf
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

    def custom_algo(board):
        """
        Tries to play in the middle column, then in the 3rd and 5th, and then elsewhere, randomly
        """
        actions = board.get_legal_actions()
        middle_columns = [2,3,4]
        center_column = 3
        middle_actions = []
        center_action = False
        for action in actions:
            if action.col == center_column:
                center_action = action
            if action.col in middle_columns:
                middle_actions.append(action)

        if center_action:
            return center_action

        if len(middle_actions) > 0:
            return np.random.choice(middle_actions)

        return np.random.choice(actions)

    p_only = PolicyPlayer('policy_only', policy_agent)
    v_only = ValuePlayer('value_only', value_agent)
    random = ComputerPlayer('random', random_algo)
    custom_center_player = ComputerPlayer('custom_center', custom_algo)

    uct_p05 =  ComputerPlayer('UCT_p05s', mcts.uct, 0.05)
    uct_p2 =  ComputerPlayer('UCT_p2s', mcts.uct, 0.2)
    uct_p5 =  ComputerPlayer('UCT_p5s', mcts.uct, 0.5)
    uct_1s = ComputerPlayer('UCT_1s', mcts.uct, 1)
    uct_2s = ComputerPlayer('UCT_2s', mcts.uct, 2)

    amcts_v_p05 = AMCTSPlayer('AMCTS_v_p05', 0.05, value_agent=value_agent)
    amcts_v_p3 = AMCTSPlayer('AMCTS_v_p3s', 0.3, value_agent=value_agent)
    amcts_v_p5 = AMCTSPlayer('AMCTS_v_p5s', .5, value_agent=value_agent)
    amcts_v_1 = AMCTSPlayer('AMCTS_v_p1s', 1, value_agent=value_agent)
    amcts_v_3 = AMCTSPlayer('AMCTS_v_p3s', 3, value_agent=value_agent)

    amcts_p05 = AMCTSPlayer('AMCTS_p05s', 0.05, policy_agent=policy_agent, value_agent=value_agent)
    amcts_p3 = AMCTSPlayer('AMCTS_p3s', 0.3, policy_agent=policy_agent, value_agent=value_agent)
    amcts_p5 = AMCTSPlayer('AMCTS_p3s', 0.5, policy_agent=policy_agent, value_agent=value_agent)
    amcts_1 = AMCTSPlayer('AMCTS_1s', 1, policy_agent=policy_agent, value_agent=value_agent)
    amcts_3 = AMCTSPlayer('AMCTS_3s', 3, policy_agent=policy_agent, value_agent=value_agent)


    return [p_only,v_only,random,custom_center_player, \
            uct_p05,uct_p2,uct_p5,uct_1s,uct_2s,\
            amcts_v_p05,amcts_v_p3,amcts_v_p5,amcts_v_1,amcts_v_3]

def compare_players(player1, player2, verbose=False, symmetric=False, num_games = 5):
    """
    Takes in two MCTS/game.py Player instance, and compares them with each other
    The "1st start score" is the fraction of games won by player1 when player1 starts
    The "2nd start score" is the fraction of games won by player2 when player2 starts

    If "symmetric" is disabled, only returns 1st start score
    Otherwise, computes both and returns ("1st start score", "2nd start score")

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
        total_time = (1 + symmetric)*num_games*MAX_GAME_MOVES*time_per_move
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

    if not symmetric:
        return first_start_score/float(num_games)

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
    players = create_players()
    pp_only,v_only,random,custom_center_player, \
            uct_p05,uct_p2,uct_p5,uct_1s,uct_2s,\
            amcts_v_p05,amcts_v_p3,amcts_v_p5,amcts_v_1,amcts_v_3 = players
    
    #players = [random, custom_center_player, uct_p05,uct_p2,uct_p5,uct_1s,uct_2s]
    black_list = [random, custom_center_player, uct_p05,uct_p2,uct_p5,uct_1s,uct_2s]
    weights_file = open("./comparisons/players_simple.txt", "a+")

    if weights_file.readline() == "":
        HEADER_LINE = "FIRST_PLAYER, "
        for player in players:
            HEADER_LINE += player.name + ", "
            weights_file.write(HEADER_LINE+"\n")

    for player1 in players:
        data_line = player1.name + ": "
        for player2 in players:
            if not (player1 in black_list and player2 in black_list):
                p1_win_rate = compare_players(player1, player2, num_games=20, symmetric=False)
                data_line += ("%.3f, "%p1_win_rate)
        weights_file.write(data_line + "\n")


if __name__ == "__main__":
    main()
