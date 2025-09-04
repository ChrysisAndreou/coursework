# Gogen Puzzle Solver

This repository contains a Python-based solver for Gogen puzzles. The program uses a depth-first search (DFS) algorithm within a constraint satisfaction framework to efficiently find solutions to 5x5 Gogen puzzles.

## What is a Gogen Puzzle?

Gogen puzzles present a challenge where a set of words must be formed within a 5x5 grid that contains a mix of pre-filled and empty cells. The objective is to strategically place the remaining 16 letters of the alphabet (excluding Z) into the empty cells. A valid solution requires that all specified words can be formed by traversing adjacent cells—horizontally, vertically, or diagonally—without reusing a cell for the same word.



## How It Works

The solver models the Gogen puzzle as a **Constraint Satisfaction Problem (CSP)**, where the goal is to assign values (letters) to a set of variables (empty cells) under specific constraints.

*   **Variables**: The 16 empty cells on the grid.
*   **Domains**: The 16 remaining letters of the alphabet (excluding the initial nine and the letter 'Z').
*   **Constraints**:
    1.  **Global Constraint**: All 25 letters on the grid must be unique.
    2.  **Local Constraint (Look-Ahead)**: Before placing a letter, the algorithm checks if the current partial grid still allows all words to be formed. If not, it backtracks immediately, pruning the search tree and reducing unnecessary exploration.

The solver employs a **Depth-First Search (DFS)** algorithm to systematically explore potential letter placements, navigating the search space efficiently to find a valid configuration.

## Features

*   Solves 5x5 Gogen puzzles.
*   Implemented using standard Python 3 libraries.
*   Utilizes a Depth-First Search (DFS) algorithm for systematic exploration.
*   Employs a Constraint Satisfaction Problem (CSP) framework for efficient problem modeling.
*   Features look-ahead constraint checking to prune infeasible search paths early.
*   Provides detailed output, including the final solved grid, the step-by-step solution path, and performance metrics.

## Getting Started

### Prerequisites

You need to have Python 3 installed on your system.

### Usage

1.  Clone the repository:
    ```sh
    git clone https://github.com/your-username/gogen-puzzle-solver.git
    cd gogen-puzzle-solver
    ```

2.  Run the solver from the command line with the following format:
    ```sh
    python3 code.py <words_file> <starting_letters>
    ```

    *   `<words_file>`: A text file containing the list of words to be found in the puzzle.
    *   `<starting_letters>`: A 9-character string representing the letters already placed on the grid.

### Example

Using the provided `gwords.txt` file and the starting letters `MGDWLYSJB`:

```sh
python3 code.py gwords.txt MGDWLYSJB
```

## Input File Format

The words file (e.g., `gwords.txt`) must be formatted as follows:
1.  The first line contains an integer representing the total number of words.
2.  Subsequent lines each contain one word to be found in the puzzle.

**Example `gwords.txt`:**

```
12
ACQUIRE
AXLE
CUT
DERV
FIR
JUST
KEG
MAN
PILE
PRY
ROB
WHAM
```

## Code Overview

The solution is structured around several interconnected classes that model the puzzle state, manage the search process, and execute the DFS algorithm.

*   `Search_State`: An abstract base class defining the interface for puzzle states. It includes methods like `goalP`, `get_Successors`, and `same_State`.
*   `Search_Node`: Represents a node within the search tree, holding a specific state and a reference to its parent.
*   `Search`: A base class that implements the generic depth-first search algorithm, maintaining an `open` list (as a stack) and a `closed` set to track visited states.
*   `Gogen_State`: A subclass of `Search_State` that encapsulates the state of the Gogen puzzle. It manages the 5x5 grid, tracks remaining letters, and generates valid successor states.
*   `Gogen_Search`: Extends the `Search` class to handle Gogen-specific logic, such as reading the words from the input file.
*   `Run_Gogen_Search`: The main driver class responsible for initializing the puzzle and executing the search.
