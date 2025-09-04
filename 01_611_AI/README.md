# Seven Six Puzzle

## Overview

This project is a Python implementation of the "Seven Six Puzzle," a game similar to Connect Four. It features a command-line interface where a human player can compete against an AI opponent. The AI's decision-making is powered by the minimax algorithm, with an optional alpha-beta pruning optimization for enhanced performance.

This project was developed by Chrysis Andreou (UC1366020) and Mohamad Fatfat (UC1367680) for the MAI611 course at the University of Cyprus.

## Features

*   **Classic Connect Four Gameplay:** A two-player game where the objective is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs.
*   **Player vs. AI:** Compete against a computer-controlled opponent.
*   **Intelligent AI:** The AI uses the minimax algorithm to determine the optimal move.
*   **Adjustable AI Difficulty:** The depth of the minimax search can be configured, allowing for varying levels of AI difficulty.
*   **Performance Optimization:** Includes an implementation of alpha-beta pruning to significantly improve the efficiency of the AI's decision-making process.
*   **Performance Analysis:** An experimental mode is available to compare the move calculation times of the minimax algorithm with and without alpha-beta pruning, visualizing the results with a plot.

## Getting Started

### Prerequisites

*   Python 3.x
*   Matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/seven-six-puzzle.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd seven-six-puzzle
    ```
3.  Install the required library:
    ```bash
    pip install matplotlib
    ```

### How to Play

1.  Run the `play_game()` function in the Python script.
2.  You will be prompted to enter the desired depth for the minimax algorithm (a value between 1 and 6). A higher number will result in a more challenging AI but will take longer to compute its moves.
3.  You will be asked if you want to use alpha-beta pruning. It is recommended to use it for a faster and more responsive AI.
4.  The game board will be displayed. The AI (Player 'X') will make the first move.
5.  On your turn (Player 'O'), enter a column number (1-6) to drop your piece.
6.  The game continues until a player achieves four in a row or the board is full, resulting in a draw.

## The AI Opponent

The AI's intelligence is based on the **minimax algorithm**, a classic decision-making algorithm for two-player games.

### Minimax Algorithm

The minimax algorithm explores the game tree to a specified depth, evaluating the "score" of the board state at each leaf node. It assumes that both players will play optimally.

*   **Maximizing Player (AI - 'X'):** Aims to maximize the score.
*   **Minimizing Player (Human - 'O'):** Aims to minimize the score.

The AI chooses the move that leads to the best possible outcome for itself, assuming the human player will also make the best possible moves.

### Alpha-Beta Pruning

This project also implements **alpha-beta pruning**, an optimization technique for the minimax algorithm. It reduces the number of nodes that need to be evaluated in the search tree by "pruning" branches that are determined to be suboptimal. This results in a significantly faster AI without compromising the quality of its moves.

## Performance Experiment

The `experiment_game()` function is included to demonstrate the performance benefits of alpha-beta pruning. This function simulates a game where one player uses the standard minimax algorithm and the other uses minimax with alpha-beta pruning. It then plots the time taken for each move, providing a clear visual comparison of their efficiency.

To run the experiment:

1.  Uncomment the `experiment_game()` call at the end of the Python script.
2.  Run the script. A plot will be generated showing the move times for both algorithms.

## Code Structure

*   `create_board()`: Initializes the 7x6 game board.
*   `print_board(board)`: Prints the current state of the board.
*   `is_valid_location(board, col)`: Checks if a column is available for a move.
*   `get_valid_locations(board)`: Returns a list of all valid moves.
*   `get_next_open_row(board, col)`: Finds the next open row in a given column.
*   `drop_piece(board, row, col, piece)`: Places a player's piece on the board.
*   `winning_move(board, piece)`: Checks if a player has won.
*   `is_terminal_node(board)`: Determines if the game has ended.
*   `evaluate_window(window, piece)`: Scores a four-cell window.
*   `score_position(board, piece)`: Calculates the overall score of the board.
*   `minimax_no_pruning(board, depth, maximizingPlayer)`: The minimax algorithm without alpha-beta pruning.
*   `minimax(board, depth, alpha, beta, maximizingPlayer)`: The minimax algorithm with alpha-beta pruning.
*   `play_game()`: The main function to run the player vs. AI game.
*   `experiment_game(depth=4)`: Runs the performance comparison experiment.
