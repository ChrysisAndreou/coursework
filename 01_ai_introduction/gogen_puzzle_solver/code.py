import sys

# Base class for search states
class Search_State:
    def goalP(self, searcher): 
        # Determines if the current state is a goal state.
        raise NotImplementedError

    def get_Successors(self, searcher):
        # Generates and returns a list of successor states from the current state.
        raise NotImplementedError

    def same_State(self, other_state):
        # Compares the current state with another state to check for equivalence.
        raise NotImplementedError

    def cost_from(self, from_state):
        # Calculates the cost of transitioning from 'from_state' to the current state.
        # used for algorthms like A* not for depth first search
        raise NotImplementedError

    def difference(self, goal_state):
        # Measures the difference between the current state and a goal state.
        # used for algorthms like A* not for depth first search
        raise NotImplementedError

# Node representation in the search tree
class Search_Node:
    def __init__(self, state, parent=None):
        # Initializes the node with a state and an optional parent node.
        self.state = state
        self.parent = parent

    def expand(self, searcher):
        # Generates and returns a list of successor nodes.
        successor_states = self.state.get_Successors(searcher)
        return [Search_Node(s, self) for s in successor_states]

    def goalP(self, searcher):
        # Returns True if the node's state is a goal state, otherwise False.
        return self.state.goalP(searcher)

    def __eq__(self, other):
        # Returns True if the node's state is equal to another node's state.
        return self.state.same_State(other.state)

    def __hash__(self):
        # Returns a hash value for the node based on its state.
        # hash allows for efficient lookup in sets and dictionaries
        return hash(self.state)

# Base class for search algorithms
class Search:
    def run_Search(self, init_state, searcher, search_method):
        self.init_node = Search_Node(init_state)
        self.open = [self.init_node]
        self.closed = set()

        while self.open:
            current_node = self.open.pop()  # Last In First Out for depth-first search 

            if current_node.state in self.closed:
                continue

            self.closed.add(current_node.state)

            if current_node.state.goalP(searcher):
                return self.report_Success(current_node)

            successor_nodes = current_node.expand(searcher)
            for node in successor_nodes: # add successors to open list if not in closed list or already in open list 
                if node.state not in self.closed and all(node.state != n.state for n in self.open):
                    self.open.append(node)  # Append to end for depth-first

        return "Search Fails"

    def report_Success(self, node):
        # Construct the solution path
        path = []
        n = node
        while n:
            path.append(n.state)
            n = n.parent
        path.reverse() # reverse path to start from initial state 

        # Calculate efficiency
        efficiency = len(path) / (len(self.closed) + 1)

        print("===============================")
        print("Search Succeeds")
        print(f"Efficiency: {efficiency:.2f} (Path length: {len(path)} / Nodes visited: {len(self.closed) + 1})")
        print(f"Nodes visited: {len(self.closed) + 1}")
        print("Solution Path:")
        for index, state in enumerate(path, start=1):
            print(f"Node {index}:")
            print()  # Add an empty line between the node index and the state
            print(state)
            print()
        return "Success"

# State representation for the Gogen puzzle
class Gogen_State(Search_State):
    def __init__(self, starting_letters):
        # Initialize the Gogen state with the starting letters 
        self.board = [[None for _ in range(5)] for _ in range(5)]
        starting_positions = [
            (0, 0), (0, 2), (0, 4),
            (2, 0), (2, 2), (2, 4),
            (4, 0), (4, 2), (4, 4)
        ]

        for i, position in enumerate(starting_positions):
            row, col = position
            self.board[row][col] = starting_letters[i]

        alphabet_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - {'Z'}
        self.remaining_letters = alphabet_set - set(starting_letters)
        self.empty_cells = [(r, c) for r in range(5) for c in range(5) if self.board[r][c] is None]

    def get_Successors(self, searcher):
        # Generate and return a list of successor states for the current state
        # check if the successors are consistent with the words 
        successors = []
        if not self.empty_cells: # if no empty cells, return empty list for successors
            return successors

        target_row, target_col = self.empty_cells[0]
        candidate_letters = self.remaining_letters.copy()

        for letter in candidate_letters:
            # Create a copy of the next state
            next_state = self.copy()
            next_state.board[target_row][target_col] = letter
            next_state.empty_cells = self.empty_cells[1:]
            next_state.remaining_letters = self.remaining_letters - {letter}

            # Check consistency 
            consistent = True
            for word in searcher.getWords():
                if not next_state.can_form_word(word, is_partial=True):
                    consistent = False
                    break

            if consistent:
                successors.append(next_state)

        return successors

    def goalP(self, searcher):
        # Check if the current state is a goal state 
        if self.empty_cells:
            return False

        for word in searcher.getWords():
            if not self.can_form_word(word, is_partial=False):
                return False

        for word in searcher.getWords():
            self.show_word_on_grid(word)
        return True

    def can_form_word(self, word, is_partial):
        # Check if the current state can form a given word 
        letter_positions = {letter: set() for letter in word} # dictionary to store the positions of the letters in the word 

        # iterate through the board to find the positions where the letters of the word are located 
        # or if the word is partial, the empty cells where the letters can be placed 
        # each letter can have multiple positions on the board  
        for i in range(5):
            for j in range(5):
                cell_value = self.board[i][j]
                for letter in word:
                    # if the cell value is the letter or if the cell is empty and the letter is in the remaining letters 
                    if cell_value == letter or (is_partial and cell_value is None and letter in self.remaining_letters): 
                        letter_positions[letter].add((i, j))

        for letter in word:
            if not letter_positions[letter]:
                return False

        def dfs(current_position, letter_index, visited_positions):
            # Base case: if the entire word is formed, return True
            if letter_index == len(word):
                return True

            # Get the next letter to find
            next_letter = word[letter_index]

            # Explore all possible positions for the next letter
            for potential_position in letter_positions[next_letter]:
                # Check if the potential position is adjacent and not visited
                if (potential_position not in visited_positions and
                    is_adjacent(current_position, potential_position)):

                    # Mark the position as visited
                    visited_positions.add(potential_position)

                    # Recursively attempt to form the rest of the word
                    if dfs(potential_position, letter_index + 1, visited_positions):
                        return True

                    # Backtrack: unmark the position as visited to try a different path using the same letter  
                    visited_positions.remove(potential_position)

            return False

        def is_adjacent(pos1, pos2):
            # Check if two positions are adjacent on the grid
            return (abs(pos1[0] - pos2[0]) <= 1 and
                    abs(pos1[1] - pos2[1]) <= 1 and
                    pos1 != pos2)

        # Start DFS from each position of the first letter
        for start_position in letter_positions[word[0]]:
            if dfs(start_position, 1, {start_position}): # if the word can be formed, return True
                return True

        return False

    def show_word_on_grid(self, target_word):
        grid_copy = [
            [
                self.board[i][j] if self.board[i][j] in target_word else ' '
                for j in range(5)
            ]
            for i in range(5)
        ]

        print("===============================")
        print(f"Word: {target_word}")
        print("Grid:")
        print('-' * 21)  
        for row in grid_copy:
            row_str = ' | '.join(row)
            print(f"| {row_str} |")
            print('-' * 21)
        print()

    def copy(self):
        new_state = Gogen_State.__new__(Gogen_State)
        new_state.board = [row[:] for row in self.board]
        new_state.remaining_letters = self.remaining_letters.copy()
        new_state.empty_cells = self.empty_cells[:]
        return new_state

    def same_State(self, other_state):
        return self.board == other_state.board

    def __eq__(self, other):
        return self.same_State(other)

    def __hash__(self):
        return hash(str(self.board))

    def __str__(self):
        grid_str = ""
        horizontal_line = '-' * 21  
        for row in self.board:
            row_str = ' | '.join([c if c is not None else ' ' for c in row])
            grid_str += f"{horizontal_line}\n| {row_str} |\n"
        grid_str += horizontal_line  
        return grid_str

# Extends Search to solve the Gogen puzzle
class Gogen_Search(Search):
    def __init__(self, words_file):
        super().__init__()
        self.words = self.read_words(words_file)

    def read_words(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
            num_words = int(lines[0])
            # Strip whitespace and convert to uppercase
            words = [line.strip().upper() for line in lines[1:num_words + 1]]
            return words

    def getWords(self):
        return self.words

    def nthWord(self, i):
        if 0 <= i < len(self.words):
            return self.words[i]
        else:
            return ""

    def run_Search(self, init_state, search_method):
        return super().run_Search(init_state, self, search_method)

# Main class to execute the Gogen search algorithm
class Run_Gogen_Search:
    def __init__(self, words_file, starting_letters):
        # Initialize Gogen_Search with words_file
        self.searcher = Gogen_Search(words_file)

        # Set up initial state
        self.initial_state = Gogen_State(starting_letters)

    def run(self):
        result = self.searcher.run_Search(self.initial_state, "depth_first")
        print(result)

# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 code.py <words_file> <starting_letters>")
        sys.exit(1)

    words_file = sys.argv[1]
    starting_letters = sys.argv[2]

    if len(starting_letters) != 9:
        print("Error: Starting letters string must be exactly 9 characters long.")
        sys.exit(1)

    print(f"Running Gogen Search with words file: {words_file} and starting letters: {starting_letters}")
    runner = Run_Gogen_Search(words_file, starting_letters)
    runner.run()
