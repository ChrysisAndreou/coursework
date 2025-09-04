# Seven Six Puzzle

This project implements a "Seven Six Puzzle" game, a variant of Connect Four, where a human player competes against a computer AI. The AI uses the minimax algorithm, with an optional alpha-beta pruning optimization, to determine its moves.

## Overview

The game is played on a 7x6 grid. The human player is 'O', and the computer is 'X'. The computer always makes the first move. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own tokens.

This repository includes the complete Python source code for the game, along with a detailed report explaining the implementation. An experiment is also included to compare the performance of the minimax algorithm with and without alpha-beta pruning.

## Features

*   **Interactive Gameplay**: Play against the computer AI in the terminal.
*   **Configurable AI Depth**: Choose the depth of the minimax search algorithm (from 1 to 6) to adjust the AI's difficulty.
*   **Alpha-Beta Pruning**: Option to enable alpha-beta pruning to significantly speed up the AI's decision-making process.
*   **Performance Analysis**: An experimental mode to visualize and compare the move calculation times of the AI with and without alpha-beta pruning.

## Getting Started

### Prerequisites

To run this project, you need to have Python 3 and the `matplotlib` library installed.

You can install `matplotlib` using pip:

```bash
pip install matplotlib
```

### How to Run the Game

1.  Clone this repository or download the `code.py` file.
2.  Open your terminal or command prompt.
3.  Navigate to the directory where you saved `code.py`.
4.  Run the following command to start the game:

    ```bash
    python code.py
    ```
5.  When prompted, enter the desired depth for the minimax algorithm (a number between 1 and 6). A higher number will result in a more challenging opponent but will take longer to compute each move.
6.  You will then be asked if you want to use alpha-beta pruning. Enter 'y' for yes or 'n' for no.
7.  The game will start. The computer ('X') will make the first move.
8.  On your turn ('O'), enter the column number (1-6) where you want to drop your piece.
9.  You can quit the game at any time by entering 'Q' or 'q'.

### How to Run the Experiment

The `experiment_game()` function at the end of the `code.py` file is set to run by default after the main game loop. This experiment simulates a game where one player uses the minimax algorithm with alpha-beta pruning and the other uses it without. It then plots the time taken for each move, providing a visual comparison of the performance.

To run only the experiment, you can comment out the `play_game()` call in the `if __name__ == "__main__":` block and ensure `experiment_game()` is called.

## How It Works

The core of the computer's AI is the **minimax algorithm**, a decision-making algorithm commonly used in two-player, turn-based games like Connect Four.

### Minimax Algorithm

The minimax algorithm explores the game tree to a specified depth, evaluating the "goodness" of each possible future board state. It assumes that the opponent will always make the best possible move to minimize the computer's score. The computer, in turn, chooses the move that maximizes its score.

### Alpha-Beta Pruning

**Alpha-beta pruning** is an optimization technique for the minimax algorithm. It reduces the number of nodes that need to be evaluated in the game's search tree by eliminating branches that are guaranteed not to influence the final decision. This significantly improves the algorithm's efficiency, allowing for deeper searches in less time.

### Evaluation Function

To score a given board state, an evaluation function is used. This function assigns a numerical value to the board based on how favorable it is for a particular player. The implementation in this project considers the following factors:

*   **Winning Moves**: A board state with four consecutive pieces for a player is given a very high score.
*   **Potential Wins**: Having three pieces in a row with an empty space is also scored favorably.
*   **Center Control**: Pieces in the center column are given a slight bonus as they offer more opportunities to create four-in-a-row.
*   **Blocking Opponent**: The evaluation function also penalizes board states where the opponent has three pieces in a row, encouraging the AI to play defensively.

## Performance Visualization

The included experiment generates a plot comparing the move times of the minimax algorithm with and without alpha-beta pruning. As shown in the report, enabling alpha-beta pruning results in a significant speedup, making the AI much faster and more efficient. For instance, the report's visualization shows the player with pruning being approximately 4.48 times faster on average.
