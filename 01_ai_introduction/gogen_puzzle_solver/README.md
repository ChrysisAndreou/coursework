# Gogen Puzzle Solver

This project features a Python-based solver for Gogen puzzles, a type of word puzzle where the objective is to place letters into a 5x5 grid to form a given set of words. The solver employs a depth-first search (DFS) algorithm within a constraint satisfaction framework to efficiently find the solution.

## Table of Contents

- [About The Project](#about-the-project)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Example](#example)

## About The Project

Gogen puzzles challenge a user to fill a 5x5 grid with letters such that a predefined list of words can be formed by moving between adjacent cells (horizontally, vertically, or diagonally). This solver automates the process by modeling the puzzle as a Constraint Satisfaction Problem (CSP).

The key components of this project are:
*   **Depth-First Search:** Systematically explores possible letter placements.
*   **Constraint Satisfaction:** Ensures that all placed letters are unique and that all words can be formed.
*   **Look-Ahead:** A local constraint check is used to prune the search space by backtracking as soon as a partial assignment cannot lead to a valid solution.

## How It Works

The solver is built around a few core classes:

*   `Search_State`: An abstract base class that defines the interface for puzzle states.
*   `Gogen_State`: Represents the state of the Gogen puzzle, including the grid and remaining letters. It handles the logic for generating successor states and checking for goal conditions.
*   `Search`: A generic search algorithm implementation.
*   `Gogen_Search`: Extends the `Search` class to handle the specific logic of the Gogen puzzle.
*   `Run_Gogen_Search`: The main class that initializes and runs the solver.

The program works by starting with the initial nine letters placed on the grid. It then recursively places the remaining letters into the empty cells. After each placement, it checks if the current partial grid is still consistent with the word list. If it's not, it backtracks. This continues until all cells are filled and all words are successfully formed.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You will need Python 3 installed on your system.

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your-username/gogen-puzzle-solver.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd gogen-puzzle-solver
    ```

## Usage

You can run the solver from the command line, providing two arguments: the path to the words file and a 9-character string of starting letters.

```sh
python3 code.py <words_file> <starting_letters>