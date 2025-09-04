import math
import random
import time
import matplotlib.pyplot as plt

# Constants for the game
ROWS = 7
COLS = 6
EMPTY = '.'
PLAYER_X = 'X'  # Computer (MAX player)
PLAYER_O = 'O'  # Human (MIN player)

# Initialize the game board with empty spaces
def create_board():
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]

# Print the current state of the board
def print_board(board):
    for row in board:
        print("".join(row))
    print()

# Check if the top row of the column is empty, indicating a valid move
def is_valid_location(board, col):
    return board[0][col] == EMPTY

# Return a list of columns that are not full
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

# The user input is a column number, we need to find the next open row in that column
def get_next_open_row(board, col): 
    for row in range(ROWS-1, -1, -1): # from last row to top row 
        if board[row][col] == EMPTY: 
            return row

# Place the player's piece in the specified location
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Check for a winning move in all directions: horizontal, vertical, and diagonal
def winning_move(board, piece):
    # Horizontal check
    for row in range(ROWS):
        for col in range(COLS-3):
            if all(board[row][col+i] == piece for i in range(4)):
                return True

    # Check vertical locations for win
    for row in range(ROWS-3):
        for col in range(COLS):
            if all(board[row+i][col] == piece for i in range(4)):
                return True

    # Check positively sloped diagonals
    for row in range(ROWS-3):
        for col in range(COLS-3):
            if all(board[row+i][col+i] == piece for i in range(4)):
                return True

    # Check negatively sloped diagonals
    for row in range(3, ROWS):
        for col in range(COLS-3):
            if all(board[row-i][col+i] == piece for i in range(4)):
                return True

    return False

# Determine if the game is over (win or draw)
def is_terminal_node(board):
    return winning_move(board, PLAYER_X) or winning_move(board, PLAYER_O) or len(get_valid_locations(board)) == 0

# Evaluate a window of four cells for scoring
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_O if piece == PLAYER_X else PLAYER_X

    # Scoring based on the number of pieces in the window
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    # Penalize if the opponent is close to winning
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

# Calculate the score of the board for the given piece
def score_position(board, piece):
    score = 0

    # Score center column to encourage central play to allow for more opportunities for a win
    center_array = [board[i][COLS//2] for i in range(ROWS)] # COLS//2 is the center column
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for row in range(ROWS):
        row_array = [board[row][i] for i in range(COLS)]
        for col in range(COLS-3):
            window = row_array[col:col+4]
            score += evaluate_window(window, piece)

    # Score Vertical
    for col in range(COLS):
        col_array = [board[row][col] for row in range(ROWS)]
        for row in range(ROWS-3):
            window = col_array[row:row+4]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for row in range(ROWS-3):
        for col in range(COLS-3):
            window = [board[row+i][col+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for row in range(3, ROWS):
        for col in range(COLS-3):
            window = [board[row-i][col+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

# Minimax algorithm without alpha-beta pruning
def minimax_no_pruning(board, depth, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_X):
                return (None, 100000000000000)  # None indicates no specific column move as the game is over
            elif winning_move(board, PLAYER_O):
                return (None, -10000000000000)
            else:  # Game is over,draw, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, PLAYER_X))
    if maximizingPlayer:
        value = -math.inf # Initialize value to negative infinity, to ensure that any score calculated will be higher than this initial value
        best_col = random.choice(valid_locations) # serves as a default choice in case no better column is found.
        for col in valid_locations:
            row = get_next_open_row(board, col) # find the next open row in the column
            b_copy = [row[:] for row in board] # create a copy of the board
            drop_piece(b_copy, row, col, PLAYER_X) # drop the piece in the copy of the board
            new_score = minimax_no_pruning(b_copy, depth-1, False)[1] # recursively call minimax_no_pruning to evaluate the score of the new board state
            if new_score > value:
                value = new_score
                best_col = col
        return best_col, value

    else:  # Minimizing player
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [row[:] for row in board]
            drop_piece(b_copy, row, col, PLAYER_O)
            new_score = minimax_no_pruning(b_copy, depth-1, True)[1] # now call with True to indicate that the next call is the MAX player's turn
            if new_score < value:
                value = new_score
                best_col = col
        return best_col, value

# Minimax algorithm with alpha-beta pruning to find the best move
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_X):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_O):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, PLAYER_X))
    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [row[:] for row in board]
            drop_piece(b_copy, row, col, PLAYER_X)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1] # [1] is the score of the new board state
            if new_score > value:
                value = new_score
                best_col = col

            '''If alpha is greater than or equal to beta, 
            the minimizing player will avoid this branch because they have a better option elsewhere in the tree.
            Therefore, further exploration of this branch is unnecessary, 
            and the algorithm can "prune" it, skipping the evaluation of any further nodes in this branch.'''
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value

    else:  # Minimizing player
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = [row[:] for row in board]
            drop_piece(b_copy, row, col, PLAYER_O)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            # Update beta to the minimum value found so far to ensure the minimizing player chooses the least score possible for MAX
            beta = min(beta, value) 
            if alpha >= beta:
                break
        return best_col, value

# Main function to play the game
def play_game():
    board = create_board()
    game_over = False
    turn = 0  # Computer starts first

    print("Let us play Seven Six Puzzle. I am X and you are O.")

    # Ask the user for the depth of the minimax algorithm
    while True:
        try:
            depth = int(input("Enter the depth for the minimax algorithm between 1 and 6: "))
            if 1 <= depth <= 6:
                break
            else:
                print("Please enter a depth between 1 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 6.")

    # Ask the user if they want to use alpha-beta pruning
    while True:
        use_alpha_beta_input = input("Do you want to use alpha-beta pruning? (y/n): ").strip().lower()
        if use_alpha_beta_input in ['y', 'n']:
            use_alpha_beta = use_alpha_beta_input == 'y' # True if y, False if n
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # Print the initial board
    print("This is the board")
    print_board(board)
    print("and I play first")

    while not game_over:
        if turn == 0:
            # Computer's turn (MAX player)
            if use_alpha_beta:
                col, minimax_score = minimax(board, depth, -math.inf, math.inf, True)
            else:
                col, minimax_score = minimax_no_pruning(board, depth, True)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_X)

                print("This is my move")
                print_board(board)

                if winning_move(board, PLAYER_X):
                    print("The final configuration")
                    print_board(board)  # Reprint the final configuration
                    print(">>> I am the Winner! <<<")
                    game_over = True

        else:
            # Human's turn (MIN player)
            while True:
                user_input = input("Select the column for your next move (1-6) or press 'Q' to quit: ")
                if user_input.lower() == 'q':
                    print(">>> Game has been quit. Goodbye! <<<")
                    game_over = True
                    break

                if user_input.isdigit():
                    col = int(user_input) - 1
                    if 0 <= col < COLS:
                        if is_valid_location(board, col):
                            row = get_next_open_row(board, col)
                            drop_piece(board, row, col, PLAYER_O)

                            print("This is what you played")
                            print_board(board)

                            if winning_move(board, PLAYER_O):
                                print("The final configuration")
                                print_board(board)  # Reprint the final configuration
                                print(">>> You are the Winner! <<<")
                                game_over = True
                            break
                        else:
                            print("Column is full. Please choose another column.")
                    else:
                        print("Invalid column. Please enter a number between 1 and 6.")
                else:
                    print("Invalid input. Please enter a number between 1 and 6 or 'Q' to quit.")

        # Check for a draw
        if not game_over and len(get_valid_locations(board)) == 0:
            print(">>> The game is a draw! <<<")
            game_over = True

        turn += 1
        turn = turn % 2  # Alternates between 0 and 1

if __name__ == "__main__":
    play_game()

def experiment_game(depth=4):
    max_times = []
    min_times = []
    experiment_starts = []

    board = create_board()
    game_over = False
    turn = 0  # MAX player starts first

    # Record the start index of the experiment
    experiment_starts.append(len(max_times))

    while not game_over:
        if turn == 0:
            # MAX player's turn (with alpha-beta pruning)
            start_time = time.time()
            col, minimax_score = minimax(board, depth, -math.inf, math.inf, True)
            end_time = time.time()
            max_times.append(end_time - start_time)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_X)

                if winning_move(board, PLAYER_X):
                    game_over = True

        else:
            # MIN player's turn (without alpha-beta pruning)
            start_time = time.time()
            col, minimax_score = minimax_no_pruning(board, depth, False)
            end_time = time.time()
            min_times.append(end_time - start_time)

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_O)

                if winning_move(board, PLAYER_O):
                    game_over = True

        # Check for a draw
        if not game_over and len(get_valid_locations(board)) == 0:
            game_over = True

        turn += 1
        turn = turn % 2  # Alternates between 0 and 1

    # Calculate average and total times
    avg_max_time = sum(max_times) / len(max_times)
    total_max_time = sum(max_times)
    avg_min_time = sum(min_times) / len(min_times)
    total_min_time = sum(min_times)

    # Calculate how much faster MAX is compared to MIN
    if avg_min_time > 0:
        speedup_ratio = avg_min_time / avg_max_time
        speedup_text = f"MAX player is {speedup_ratio:.2f} times faster than MIN player on average."
    else:
        speedup_text = "MIN player did not make any moves, so speed comparison is not applicable."

    avg_max_text = f"Average MAX player (with pruning) move time: {avg_max_time:.4f} seconds"
    total_max_text = f"Total MAX player (with pruning) move time: {total_max_time:.4f} seconds"
    avg_min_text = f"Average MIN player (without pruning) move time: {avg_min_time:.4f} seconds"
    total_min_text = f"Total MIN player (without pruning) move time: {total_min_time:.4f} seconds"

    # Plot the times
    plt.figure(figsize=(10, 5))
    plt.plot(max_times, label='MAX player (with pruning)', marker='o')
    plt.plot(min_times, label='MIN player (without pruning)', marker='x')

    plt.xlabel('Move Number')
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of Move Times with and without Alpha-Beta Pruning')
    plt.legend()
    plt.grid(True)

    # Add text annotations to the plot
    plt.text(0.55, 0.95, speedup_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
    plt.text(0.55, 0.90, avg_max_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
    plt.text(0.55, 0.85, total_max_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
    plt.text(0.55, 0.80, avg_min_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
    plt.text(0.55, 0.75, total_min_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')

    plt.show()

# Run the experiment
experiment_game()
