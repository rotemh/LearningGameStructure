import game
import algo
import time
import inp
import mcts

from IPython.display import display, display_html, display_markdown, IFrame

from pprint import pprint

def simulate_game(uct):
    #make_game_vis()#
    time_limit_1 = 0.4
    time_limit_2 = 0.4

    board = game.ConnectFourBoard()
    player_1 = game.ComputerPlayer('mcts', uct, time_limit_1)
    player_2 = game.ComputerPlayer('alpha-beta', algo.alpha_beta_algo, time_limit_2)
    player_3 = game.ConnectFourHumanPlayer('me', inp.connect_four_console_source)
    sim = game.Simulation(board, player_1, player_2)
    episode = sim.run(visualize=True, json_visualize=False,state_action_history=True)
    time.sleep(0.3)
    #return sim.board.current_player_id() # why would we need an id?
    return episode

def generate_uct_game(time_limit=1.0):
    # The higher the time_limit, the better the players will perform
    board = game.ConnectFourBoard()
    player_1 = game.ComputerPlayer('mcts', mcts.uct, time_limit)
    player_2 = game.ComputerPlayer('mcts', mcts.uct, time_limit)
    sim = game.Simulation(board, player_1, player_2)
    history = sim.run(visualize=False,state_action_history=True)
    return history

def generate_custom_policy_game(algo_1, algo_2):
    # Play using two custom algorithms
    player_1 = game.ComputerPlayer('algo_1', algo_1)
    player_2 = game.ComputerPlayer('algo_2', algo_2)

def make_game_vis():
    frame = IFrame('vis/index.html', 490, 216)
    display(frame)

def run_final_test(uct):
    losses = 0
    for i in xrange(10):
        loser = simulate_game(uct)
        if loser == 0:
            losses += 1
            if losses > 1:
                lose()
                return
    win()

def win():
    display_html("""<div class="alert alert-success">
    <strong>You win!!</strong>
    </div>

<p>Stonn sits back in shock, displaying far more emotion than any Vulcan should.</p>

<p>"Cadet, it looks like your thousands of years in the mud while we Vulcans
explored the cosmos were not in vain. Congratulations."</p>

<p>The class breaks into applause! Whoops and cheers ring through the air as
Captain James T. Kirk walks into the classroom to personally award you with
the Kobayashi Maru Award For Excellence In Tactics.</p>

<p>The unwelcome interruption of your blaring alarm clock brings you back to
reality, where in the year 2200 Earth's Daylight Savings Time was finally
abolished by the United Federation of Planets.</p>""", raw=True)

def lose():
    display_html("""<div class="alert alert-failure">
    <strong>You can only lose once :(</strong>
    </div>""", raw=True)

#history = simulate_game(mcts.uct)
#import pdb;pdb.set_trace()