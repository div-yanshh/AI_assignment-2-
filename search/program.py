# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction
from .utils import render_board
from collections import deque

# Constants
BOARD_N = 8

# Helper functions
def board_to_tuple(board: dict[Coord, CellState]):
    """
    Convert the board to a sorted tuple of tuples for hashing
    """
    return tuple(sorted(((coord.r, coord.c), cellState.value) for coord, cellState in board.items()))

def goal_test(board: dict[Coord, CellState]):
    """
    Check if the board is in the goal state
    """

    for coord, cellState in board.items():
        if cellState == CellState.RED and coord.r == 7:
            return True
        
    return False

def find_jump_sequences(start: Coord, board: dict[Coord, CellState]) -> list[list[Direction]]:
    """
    Find all possible jump sequences starting from the given coordinate
    """

    allowed_directions = [
        Direction.Right,
        Direction.Left,
        Direction.Down,
        Direction.DownRight,
        Direction.DownLeft
    ]

    jump_sequences = []

    # Trying all possible directions
    for direction in allowed_directions:
        # Calculating the adjacent cell
        adj_r = start.r + direction.r
        adj_c = start.c + direction.c

        # Check if the adjacent cell is within bounds.
        if not (0 <= adj_r < BOARD_N and 0 <= adj_c < BOARD_N):
            continue

        adjacent = Coord(adj_r, adj_c)

        # Check if the adjacent cell contains a frog.
        if adjacent not in board or board[adjacent] != CellState.BLUE:
            continue

        # Calculating final jump destination
        jump_r = adj_r + direction.r
        jump_c = adj_c + direction.c

        # Checking if the jump destination is within bounds
        if not (0 <= jump_r < BOARD_N and 0 <= jump_c < BOARD_N):
            continue

        jump_dest = Coord(jump_r, jump_c)

        if jump_dest not in board or board[jump_dest] != CellState.LILY_PAD:
            continue

        # Simulate the possible jump
        jump_move = MoveAction(start, [direction, direction])
        new_board = apply_move(board, jump_move)

        # Recursively find all possible jump sequences from the new board
        subsequent_jump_sequences = find_jump_sequences(jump_dest, new_board)
        if subsequent_jump_sequences:
            for seq in subsequent_jump_sequences:
                jump_sequences.append([direction] + seq)
        else:
            jump_sequences.append([direction])
        
    return jump_sequences


            



def generate_valid_moves(board: dict[Coord, CellState]):
    """
    Generate all valid moves from the current board state
    """
    moves = []
    red_coords = None

    # Find the red frog's coordinate.
    for coord, cellState in board.items():
        if cellState == CellState.RED:
            red_coords = coord
            break
    
    if red_coords is None:
        return moves
    
    # Allowed directions for Red
    allowed_directions = [
        Direction.Right,
        Direction.Left,
        Direction.Down,
        Direction.DownRight,
        Direction.DownLeft
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
            expanded_seq = [d for d in seq for _ in (0,1)]
            moves.append(MoveAction(red_coords, expanded_seq))

    return moves

def apply_move(board: dict[Coord, CellState], move: MoveAction):
    """
    Apply the move to the board
    """
    new_board = board.copy()

    source = move.coord

    dest = source
    for direction in move.directions:
        dest = dest + direction

    # remove the lilypad at the source
    if source in new_board and new_board[source] == CellState.RED:
        del new_board[source]

    # remove the lily pad at the destination because the frog will jump there
    if dest in new_board and new_board[dest] == CellState.LILY_PAD:
        del new_board[dest]

    # place the frog at the destination
    new_board[dest] = CellState.RED

    return new_board

def bfs_search(board: dict[Coord, CellState]):
    """
    Performs a breadth-first search to find the solution
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
            new_board_tuple = board_to_tuple(new_board)

            # check if the new state has been visited
            if new_board_tuple not in visited:
                visited.add(new_board_tuple)
                queue.append((new_board, current_path + [move]))
    
    # no solution found if all options are exhausted
    return None


def search(
    board: dict[Coord, CellState]
) -> list[MoveAction] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `CellState` instances which can be one of
            `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
    
    Returns:
        A list of "move actions" as MoveAction instances, or `None` if no
        solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, ansi=True))

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...
    return bfs_search(board)

    # Here we're returning "hardcoded" actions as an example of the expected
    # output format. Of course, you should instead return the result of your
    # search algorithm. Remember: if no solution is possible for a given input,
    # return `None` instead of a list.
    # return [
    #     MoveAction(Coord(0, 5), [Direction.Down]),
    #     MoveAction(Coord(1, 5), [Direction.DownLeft]),
    #     MoveAction(Coord(3, 3), [Direction.Left]),
    #     MoveAction(Coord(3, 2), [Direction.Down, Direction.Right]),
    #     MoveAction(Coord(5, 4), [Direction.Down]),
    #     MoveAction(Coord(6, 4), [Direction.Down]),
    # ]