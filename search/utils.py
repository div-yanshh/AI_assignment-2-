# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import Coord, CellState, BOARD_N
from .core import Direction, MoveAction
from collections import deque

# Helper functions
def board_to_tuple(board: dict[Coord, CellState]):
    """
    Convert the board dictionary to a sorted tuple of tuples for hashing.
    
    Args:
        board (dict[Coord, CellState]): The current board configuration.
    
    Returns:
        tuple: A sorted tuple representation of the board, where each element is ((row, col), cellState.value).
    """

    return tuple(sorted(((coord.r, coord.c), cellState.value) for coord, cellState in board.items()))

def goal_test(board: dict[Coord, CellState]):
    """
    Check if the board is in the goal state.
    The board is in the goal state if the red frog (CellState.RED) occupies any cell in row 7.
    
    Args:
        board (dict[Coord, CellState]): The current board configuration.
    
    Returns:
        bool: True if the goal is reached, False otherwise.
    """

    for coord, cellState in board.items():
        if cellState == CellState.RED and coord.r == 7:
            return True
        
    return False

def find_jump_sequences(start: Coord, board: dict[Coord, CellState], path=None) -> list[list[Direction]]:
    """
    Recursively find all possible jump sequences starting from 'start' on 'board'.
    Each jump sequence is returned as a list of Directions (each representing a hop). The 'path'
    parameter tracks the coordinates visited in the current jump chain to prevent cycles.
    
    Args:
        start (Coord): The starting coordinate for jump sequences.
        board (dict[Coord, CellState]): The current board configuration.
        path (list[Coord], optional): List of visited coordinates in the current jump chain.
    
    Returns:
        list[list[Direction]]: A list of jump sequences, where each sequence is a list of Directions.
    """
    # Initialize the path if it's the first call.
    if path is None:
        path = []

    # Create a new path to avoid modifying the original.
    new_path = path + [start]
    
    allowed_directions = [
        Direction.Down,
        Direction.DownRight,
        Direction.DownLeft,
        Direction.Right,
        Direction.Left
    ]
    jump_sequences = []
    
    # Iterate through each allowed direction.
    for direction in allowed_directions:

        # Calculate the adjacent cell (the cell to jump over).
        adj_r = start.r + direction.r
        adj_c = start.c + direction.c

        # Check if the adjacent cell is within bounds.
        if not (0 <= adj_r < BOARD_N and 0 <= adj_c < BOARD_N):
            continue
        adjacent = Coord(adj_r, adj_c)

        # Ensure the adjacent cell contains a blue frog.
        if adjacent not in board or board[adjacent] != CellState.BLUE:
            continue
        
        # Calculate the jump destination (landing cell).
        jump_r = adj_r + direction.r
        jump_c = adj_c + direction.c

        # Check if the jump destination is within bounds.
        if not (0 <= jump_r < BOARD_N and 0 <= jump_c < BOARD_N):
            continue
        jump_dest = Coord(jump_r, jump_c)

        # Ensure the landing cell is an unoccupied lily pad.
        if jump_dest not in board or board[jump_dest] != CellState.LILY_PAD:
            continue
        
        # Prevent cycles: check if the jump destination is already in the path.
        if jump_dest in new_path:
            continue
        
        # Simulate the jump.
        jump_move = MoveAction(start, [direction])
        new_board = apply_move(board, jump_move)
        
        subsequent_sequences = find_jump_sequences(jump_dest, new_board, new_path)

        # If further jumps are available, combine them with the current direction.
        if subsequent_sequences:
            for seq in subsequent_sequences:
                jump_sequences.append([direction] + seq)
        else:
            # If no further jumps are available, the sequence is just the current direction.
            jump_sequences.append([direction])
    
    return jump_sequences

def generate_valid_moves(board: dict[Coord, CellState]):
    """
    Generate all valid moves from the current board state.
    The function identifies the red frog's position and generates both simple moves (to an adjacent lily pad)
    and jump moves (using find_jump_sequences). Each move is represented as a MoveAction.
    
    Args:
        board (dict[Coord, CellState]): The current board configuration.
    
    Returns:
        list[MoveAction]: A list of possible moves from the current state.
    """
    moves = []
    red_coords = None

    # Find the red frog's coordinate.
    for coord, cellState in board.items():
        if cellState == CellState.RED:
            red_coords = coord
            break
    
    # If no red frog is found, return an empty list.
    if red_coords is None:
        return moves
    
    # Allowed directions for Red
    allowed_directions = [
        Direction.Down,
        Direction.DownRight,
        Direction.DownLeft,
        Direction.Right,
        Direction.Left,
    ]

    for direction in allowed_directions:

        # Calculate the destination manually to avoid wrapping around the board.
        new_r = red_coords.r + direction.r
        new_c = red_coords.c + direction.c

        # Check if the new coordinates are within bounds.
        if not (0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N):
            continue

        # -- Simple moves --
        simple_dest = Coord(new_r, new_c)
        if simple_dest in board and board[simple_dest] == CellState.LILY_PAD:
            moves.append(MoveAction(red_coords, [direction]))

        # -- Jump moves --
        jump_sequences = find_jump_sequences(red_coords, board)
        for seq in jump_sequences:
            moves.append(MoveAction(red_coords, seq))   

    return moves

def apply_move(board: dict[Coord, CellState], move: MoveAction):
    """
    Apply the given move to the board and return the new board state.
    For a simple move, the destination is computed as the source plus the move's direction.
    For a jump move (where move.directions is a list of Directions), each hop moves the frog 2 cells.
    The function updates the board by removing the red frog from the source (converting it to a lily pad)
    and placing the red frog at the destination (consuming the lily pad there).
    
    Args:
        board (dict[Coord, CellState]): The current board configuration.
        move (MoveAction): The move to apply.
    
    Returns:
        dict[Coord, CellState]: The new board configuration after the move.
    """
    new_board = board.copy()
    source = move.coord
    dest = source
    # Check first direction
    d = move.directions[0]

    # Check if the immediate adjacent cell is a blue frog.
    if board.get(source + d) == CellState.BLUE:
        # If the adjacent cell is a blue frog, treat it as a jump move.
        for d in move.directions:
            dest = dest + (d * 2)
    elif board.get(source + d) == CellState.LILY_PAD:
        # If the adjacent cell is a lily pad, treat it as a simple move.
        dest = source + d

    # Update board: remove the red frog from the source.
    if source in new_board and new_board[source] == CellState.RED:
        new_board[source] = CellState.LILY_PAD

    # Remove the lily pad at the destination (if present).
    if dest in new_board and new_board[dest] == CellState.LILY_PAD:
        del new_board[dest]

    # Place the red frog at the destination.
    new_board[dest] = CellState.RED

    return new_board

def bfs_search(board: dict[Coord, CellState]):
    """
    Perform a breadth-first search (BFS) to find a sequence of moves that moves the red frog to row 7.
    The BFS uses a FIFO queue to store tuples of (board, move path). The board is represented as a dictionary
    and converted to a hashable tuple (using board_to_tuple) for duplicate checking. The first solution found is optimal.
    
    Args:
        board (dict[Coord, CellState]): The initial board configuration.
    
    Returns:
        list[MoveAction] | None: The sequence of moves leading to the goal state, or None if no solution is found.
    """
    
    queue = deque([(board, [])])
    visited = set()
    visited.add(board_to_tuple(board))

    while queue:
        current_board, current_path = queue.popleft()

        # check if current state is the goal state
        if goal_test(current_board):
            return current_path

        # generate all possible moves from the current state
        for move in generate_valid_moves(current_board):
            new_board = apply_move(current_board, move)

            if goal_test(new_board):
                return current_path + [move]

            new_board_tuple = board_to_tuple(new_board)

            # check if the new state has been visited
            if new_board_tuple not in visited:
                visited.add(new_board_tuple)
                queue.append((new_board, current_path + [move]))
    
    # no solution found if all options are exhausted
    return None


def apply_ansi(
    text: str, 
    bold: bool = False, 
    color: str | None = None
):
    """
    Wraps some text with ANSI control codes to apply terminal-based formatting.
    Note: Not all terminals will be compatible!
    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    if color == "g":
        color_code = "\033[32m"
    return f"{bold_code}{color_code}{text}\033[0m"


def render_board(
    board: dict[Coord, CellState], 
    ansi: bool = False
) -> str:
    """
    Visualise the Tetress board via a multiline ASCII string, including
    optional ANSI styling for terminals that support this.

    If a target coordinate is provided, the token at that location will be
    capitalised/highlighted.
    """
    output = ""
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell_state = board.get(Coord(r, c), None)
            if cell_state:
                text = '.'
                color = None
                if cell_state == CellState.RED:
                    text = "R"
                    color = "r"
                elif cell_state == CellState.BLUE:
                    text = "B"
                    color = "b"
                elif cell_state == CellState.LILY_PAD:
                    text = "*"
                    color = "g"

                if ansi:
                    output += apply_ansi(text, color=color)
                else:
                    output += text
            else:
                output += "."
            output += " "
        output += "\n"
    return output
