# # COMP30024 Artificial Intelligence, Semester 1 2025
# # Project Part A: Single Player Freckers

# from .core import CellState, Coord, Direction, MoveAction
# from .utils import render_board
# from collections import deque
# import time

# # Constants
# BOARD_N = 8

# def goal_test(board: dict[Coord, CellState]) -> bool:
#     """
#     Check if the red frog is in row 7.
#     """
#     for coord, cellState in board.items():
#         if cellState == CellState.RED and coord.r == 7:
#             return True
#     return False

# def find_jump_sequences(board: dict[Coord, CellState], start: Coord, direction: Direction) -> list[MoveAction]:
#     """
#     Find all possible jump sequences from the start position in the given direction.
#     A jump sequence consists of a series of jumps over blue frogs to reach a lily pad.
#     Each jump sequence is represented as a list of MoveActions with a path cost of 1.
#     """
#     possible_jump_sequences = []
#     visited = set()
#     visited.add(start)
#     queue = deque([(start, [])])

#     while queue:
#         current_coord, path = queue.popleft()
#         r, c = current_coord.r, current_coord.c

#         # Check if the next cell in the given direction is a blue frog
#         next_r = r + direction.r
#         next_c = c + direction.c

#         if not (0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N):
#             continue

#         adjacent_coord = Coord(next_r, next_c)
#         # Check if the adjacent cell is a blue frog
#         if board.get(adjacent_coord) == CellState.BLUE:
#             # Check if the cell after the blue frog is a lily pad
#             jump_r = next_r + direction.r
#             jump_c = next_c + direction.c
#             if not (0 <= jump_r < BOARD_N and 0 <= jump_c < BOARD_N):
#                 continue

#             jump_coord = Coord(jump_r, jump_c)
#             if board[jump_coord] == CellState.LILY_PAD and jump_coord not in visited:
#                 # Create a new MoveAction for the jump
#                 jump_action = MoveAction(current_coord, jump_coord)
#                 new_path = path + [jump_action]
#                 possible_jump_sequences.append(new_path)
#                 visited.add(jump_coord)

#                 # Add the new position to the queue for further jumps
#                 queue.append((jump_coord, new_path))

#     return possible_jump_sequences

# def apply_move(board: dict[Coord, CellState], move: MoveAction) -> dict[Coord, CellState]:
#     """
#     Apply the move to the board and return the new board state.
#     For a simple move, the destination is source + direction.
#     For a jump move (where move.directions has more than one element),
#     each direction is applied as a jump, moving the frog 2 cells per jump.
#     """
#     new_board = board.copy()
#     source = move.coord
#     dest = source

#     # If there's a single direction, we consider it a simple move.
#     if len(move.directions) == 1:
#         d = move.directions[0]
#         dest = source + d
#     else:
#         # For jump moves (including multi-hop), each direction causes a 2-cell move.
#         for d in move.directions:
#             dest = dest + (d * 2)

#     # Update board: remove red frog from source, place red frog at destination.
#     # Replace the source cell with a lily pad.
#     if source in new_board and new_board[source] == CellState.RED:
#         new_board[source] = CellState.LILY_PAD

#     # The destination should be a lily pad that gets consumed by the frog.
#     new_board[dest] = CellState.RED
#     if dest in new_board and new_board[dest] == CellState.LILY_PAD:
#         # Remove the lily pad.
#         del new_board[dest]
#         new_board[dest] = CellState.RED

#     return new_board


# def get_possible_moves(board: dict[Coord, CellState]) -> list[MoveAction]:
#     """
#     Get all possible moves from the current board state.
#     A move is represented as a tuple (direction, cell).
#     """
#     possible_moves = []
#     red_coord = None

#     # Find the red frog's position
#     for coord, cellState in board.items():
#         if cellState == CellState.RED:
#             red_coord = coord
#             break

#     if red_coord is None:
#         return possible_moves

#     possible_directions = [
#         Direction.Down,
#         Direction.Left,
#         Direction.Right,
#         Direction.DownLeft,
#         Direction.DownRight,
#     ]

#     for direction in possible_directions:

#         # Calculate the new position based on the direction
#         new_r = red_coord.r + direction.r
#         new_c = red_coord.c + direction.c

#         # Check if the new position is within bounds
#         if not (0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N):
#             continue

#         # -- Simple Moves --
#         simple_dest = Coord(new_r, new_c)
#         if simple_dest and board.get(simple_dest) == CellState.LILY_PAD:
#             possible_moves.append(MoveAction(red_coord, simple_dest))

#         # -- Jump Moves --
#         # Find all possible jump sequences
#         possible_jump_sequences = find_jump_sequences(board, red_coord, direction)
#         for jump_sequence in possible_jump_sequences:
#             # Append each jump sequence list to the possible moves
#             possible_moves.append(jump_sequence)

#     return possible_moves




# def bfs_search(board: dict[Coord, CellState]) -> list[MoveAction] | None:
#     """
#     Perform a breadth-first search to find a sequence of moves
#     that moves the red frog to row 7.
#     """

#     # Initialize the queue with the initial board state and an empty path
#     queue = deque([(board, [])])
#     visited = set()
#     visited.add(tuple(board.items()))

#     while queue:
#         current_board, path = queue.popleft()

#         # Check if the red frog is in row 7
#         if goal_test(current_board):
#             return path
        
#         # Get all possible moves from the current board state
#         possible_moves = get_possible_moves(current_board)

#         # Iterate through each possible move
#         for move in possible_moves:
#             # Apply the move to get the new board state
#             new_board = apply_move(current_board, move)

#             # Check if the new board state has already been visited
#             if tuple(new_board.items()) not in visited:
#                 visited.add(tuple(new_board.items()))
#                 queue.append((new_board, path + [move]))

#     # If no solution is found, return None
#     return None

# def search(board: dict[Coord, CellState]) -> list[MoveAction] | None:
#     """
#     Entry point for the search. Finds a sequence of moves from the initial board
#     state that moves the red frog to row 7. It prints the board state after each
#     move in the solution sequence.
#     """
#     print("Initial board:")
#     print(render_board(board, ansi=True))

#     # Use BFS to get the solution path (a list of MoveActions).
#     solution_path = bfs_search(board)

#     if solution_path is None:
#         print("No solution found.")
#         return None

#     # Apply each move from the solution path and print the board after each move.
#     current_board = board
#     for i, move in enumerate(solution_path, start=1):
#         current_board = apply_move(current_board, move)
#         print(f"\nBoard after move {i} ({move}):")
#         print(render_board(current_board, ansi=True))

#     return solution_path

##########################################################################################################

# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction
from .utils import render_board
from collections import deque
import time

# Constants
BOARD_N = 8

def goal_test(board: dict[Coord, CellState]) -> bool:
    """
    Check if the red frog is in row 7.
    """
    for coord, cellState in board.items():
        if cellState == CellState.RED and coord.r == 7:
            return True
    return False

def find_jump_sequences(board: dict[Coord, CellState], start: Coord, direction: Direction) -> list[list[Direction]]:
    """
    Find all possible jump sequences from the start position in the given direction.
    A jump sequence is returned as a list of directions.
    """
    possible_jump_sequences = []
    visited = set()
    visited.add(start)
    queue = deque([(start, [])])

    while queue:
        current_coord, path = queue.popleft()
        r, c = current_coord.r, current_coord.c

        # Check if the next cell in the given direction is a blue frog.
        next_r = r + direction.r
        next_c = c + direction.c
        if not (0 <= next_r < BOARD_N and 0 <= next_c < BOARD_N):
            continue

        adjacent_coord = Coord(next_r, next_c)
        if board.get(adjacent_coord) != CellState.BLUE:
            continue

        # Check if the cell after the blue frog is a lily pad.
        jump_r = next_r + direction.r
        jump_c = next_c + direction.c
        if not (0 <= jump_r < BOARD_N and 0 <= jump_c < BOARD_N):
            continue

        jump_coord = Coord(jump_r, jump_c)
        if board.get(jump_coord) != CellState.LILY_PAD:
            continue

        if jump_coord in visited:
            continue

        new_path = path + [direction]
        possible_jump_sequences.append(new_path)
        visited.add(jump_coord)
        queue.append((jump_coord, new_path))

    return possible_jump_sequences

def apply_move(board: dict[Coord, CellState], move: MoveAction) -> dict[Coord, CellState]:
    """
    Apply the move to the board and return the new board state.
    For a simple move, the destination is source + direction.
    For a jump move (where move.directions is a list of Directions),
    each jump moves the frog 2 cells per jump.
    """
    new_board = board.copy()
    source = move.coord
    dest = source

    if len(move.directions) == 1:
        # Simple move: move one cell in the given direction.
        d = move.directions[0]
        dest = source + d
    else:
        # Jump move: each direction moves the frog by 2 cells.
        for d in move.directions:
            dest = dest + (d * 2)

    # Update board: remove the red frog from the source.
    if source in new_board and new_board[source] == CellState.RED:
        new_board[source] = CellState.LILY_PAD

    # Place the red frog at the destination (consuming the lily pad).
    new_board[dest] = CellState.RED
    # (Optional: you may want to explicitly remove the lily pad, e.g. by deleting the key,
    # if that is required by the assignment. For example:
    # if dest in new_board and new_board[dest] == CellState.LILY_PAD:
    #     del new_board[dest]
    #     new_board[dest] = CellState.RED
    # )
    
    # For jump moves, also remove the blue frog jumped over.
    if len(move.directions) > 1 or (len(move.directions) == 1 and board.get(source + move.directions[0]) != CellState.LILY_PAD):
        mid_r = (source.r + dest.r) // 2
        mid_c = (source.c + dest.c) // 2
        mid_coord = Coord(mid_r, mid_c)
        new_board[mid_coord] = CellState.LILY_PAD

    return new_board

def get_possible_moves(board: dict[Coord, CellState]) -> list[MoveAction]:
    """
    Get all possible moves from the current board state.
    Returns a list of MoveAction objects.
    """
    possible_moves = []
    red_coord = None

    # Find the red frog's position.
    for coord, cellState in board.items():
        if cellState == CellState.RED:
            red_coord = coord
            break

    if red_coord is None:
        return possible_moves

    possible_directions = [
        Direction.Down,
        Direction.Left,
        Direction.Right,
        Direction.DownLeft,
        Direction.DownRight,
    ]

    for direction in possible_directions:
        # -- Simple Moves --
        new_r = red_coord.r + direction.r
        new_c = red_coord.c + direction.c
        if not (0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N):
            continue
        simple_dest = Coord(new_r, new_c)
        if board.get(simple_dest) == CellState.LILY_PAD:
            possible_moves.append(MoveAction(red_coord, [direction]))  # Pass as a list.

        # -- Jump Moves --
        jump_sequences = find_jump_sequences(board, red_coord, direction)
        for seq in jump_sequences:
            # Here, seq is a list of Directions representing a jump chain.
            possible_moves.append(MoveAction(red_coord, seq))

    return possible_moves

def bfs_search(board: dict[Coord, CellState]) -> list[MoveAction] | None:
    """
    Perform a breadth-first search to find a sequence of moves that moves the red frog to row 7.
    """
    nodes_created = 1
    nodes_explored = 0
    start_time = time.time()
    
    queue = deque([(board, [])])
    visited = set()
    visited.add(tuple(board.items()))

    while queue:
        current_board, path = queue.popleft()
        nodes_explored += 1

        if goal_test(current_board):
            end_time = time.time()
            print(f"Nodes created: {nodes_created}")
            print(f"Nodes explored: {nodes_explored}")
            print(f"Total time taken: {end_time - start_time:.4f} seconds")
            return path
        
        possible_moves = get_possible_moves(current_board)
        for move in possible_moves:
            new_board = apply_move(current_board, move)
            new_board_tuple = tuple(new_board.items())
            if new_board_tuple not in visited:
                visited.add(new_board_tuple)
                queue.append((new_board, path + [move]))
                nodes_created += 1

    end_time = time.time()
    print(f"Nodes created: {nodes_created}")
    print(f"Nodes explored: {nodes_explored}")
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    return None

def search(board: dict[Coord, CellState]) -> list[MoveAction] | None:
    """
    Entry point for the search. Finds a sequence of moves from the initial board
    state that moves the red frog to row 7. It prints the board state after each
    move in the solution sequence.
    """
    print("Initial board:")
    print(render_board(board, ansi=True))

    # Use BFS to get the solution path (a list of MoveActions).
    solution_path = bfs_search(board)
    if solution_path is None:
        print("No solution found.")
        return None

    # Apply each move from the solution path and print the board after each move.
    current_board = board
    for i, move in enumerate(solution_path, start=1):
        current_board = apply_move(current_board, move)
        print(f"\nBoard after move {i} ({move}):")
        print(render_board(current_board, ansi=True))

    return solution_path
