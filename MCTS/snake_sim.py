import game
import snake_board
import snake_player
import inp
import mcts

def simulate_game(uct):
    time_limit_1 = 0.1
    time_limit_2 = 0.1
    time_limit_3 = 0.1
    board = snake_board.SnakeBoard()
    player_1 = game.ComputerPlayer('mcts_1', uct, time_limit_1)
    player_2 = game.ComputerPlayer('mcts_2', uct, time_limit_2)
    player_3 = snake_player.SnakePlayer('human', inp.snake_random_source(time_limit_3))
    sim = game.Simulation(board, player_1, player_2)
    sim.run(visualize=True, json_visualize=True)

    return sim.board.current_player_id()

simulate_game(mcts.uct_fixed_horizon)

# up 1
# right 2
# down 3
# left 4
