# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction
from .utils import render_board
from collections import deque
import time

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
            moves.append(MoveAction(red_coords, seq))   

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

    nodes_created = 1  # The initial board counts as a created node.
    nodes_explored = 0
    start_time = time.time()  # Record the starting time.
    
    queue = deque([(board, [])])
    visited = set()
    visited.add(board_to_tuple(board))

    while queue:
        current_board, current_path = queue.popleft()
        nodes_explored += 1  # We are exploring this node.

        # check if current state is the goal state
        if goal_test(current_board):
            end_time = time.time()
            print(f"Nodes created: {nodes_created}")
            print(f"Nodes explored: {nodes_explored}")
            print(f"Total time taken: {end_time - start_time:.4f} seconds")
            return current_path
        
        # generate all possible moves from the current state
        for move in generate_valid_moves(current_board):
            new_board = apply_move(current_board, move)
            new_board_tuple = board_to_tuple(new_board)

            # check if the new state has been visited
            if new_board_tuple not in visited:
                visited.add(new_board_tuple)
                queue.append((new_board, current_path + [move]))
                nodes_created += 1 
    
    # no solution found if all options are exhausted
    # If no solution is found, print the stats.
    end_time = time.time()
    print(f"Nodes created: {nodes_created}")
    print(f"Nodes explored: {nodes_explored}")
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    return None


# def search(
#     board: dict[Coord, CellState]
# ) -> list[MoveAction] | None:
#     """
#     This is the entry point for your submission. You should modify this
#     function to solve the search problem discussed in the Part A specification.
#     See `core.py` for information on the types being used here.

#     Parameters:
#         `board`: a dictionary representing the initial board state, mapping
#             coordinates to "player colours". The keys are `Coord` instances,
#             and the values are `CellState` instances which can be one of
#             `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
    
#     Returns:
#         A list of "move actions" as MoveAction instances, or `None` if no
#         solution is possible.
#     """

#     # The render_board() function is handy for debugging. It will print out a
#     # board state in a human-readable format. If your terminal supports ANSI
#     # codes, set the `ansi` flag to True to print a colour-coded version!
#     print(render_board(board, ansi=True))

#     # Do some impressive AI stuff here to find the solution...
#     # ...
#     # ... (your solution goes here!)
#     # ...
#     return bfs_search(board)

#     # Here we're returning "hardcoded" actions as an example of the expected
#     # output format. Of course, you should instead return the result of your
#     # search algorithm. Remember: if no solution is possible for a given input,
#     # return `None` instead of a list.
#     # return [
#     #     MoveAction(Coord(0, 5), [Direction.Down]),
#     #     MoveAction(Coord(1, 5), [Direction.DownLeft]),
#     #     MoveAction(Coord(3, 3), [Direction.Left]),
#     #     MoveAction(Coord(3, 2), [Direction.Down, Direction.Right]),
#     #     MoveAction(Coord(5, 4), [Direction.Down]),
#     #     MoveAction(Coord(6, 4), [Direction.Down]),
#     # ]
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

#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================
#==========================================================================================================================================================================


# import heapq
# import math
# import time
# import itertools
# from .core import CellState, Coord, Direction, MoveAction
# from .utils import render_board

# # Constants
# BOARD_N = 8

# def board_to_tuple(board: dict[Coord, CellState]):
#     """
#     Convert the board to a sorted tuple of tuples for hashing.
#     """
#     return tuple(sorted(((coord.r, coord.c), cellState.value) for coord, cellState in board.items()))

# def goal_test(board: dict[Coord, CellState]):
#     """
#     Check if the board is in the goal state.
#     """
#     for coord, cellState in board.items():
#         if cellState == CellState.RED and coord.r == 7:
#             return True
#     return False

# def heuristic(board: dict[Coord, CellState]) -> int:
#     """
#     An admissible heuristic based on the vertical distance divided by a fixed constant.
#     Assumes that in the best-case scenario a move covers 2 rows.
#     """
#     goal_row = 7
#     red_pos = None
#     for coord, state in board.items():
#         if state == CellState.RED:
#             red_pos = coord
#             break
#     if red_pos is None:
#         return float('inf')
    
#     vertical_distance = goal_row - red_pos.r
#     if vertical_distance <= 0:
#         return 0
    
#     constant = 2
#     return math.ceil(vertical_distance / constant)

# def find_jump_sequences(start: Coord, board: dict[Coord, CellState]) -> list[list[Direction]]:
#     """
#     Find all possible jump sequences starting from the given coordinate.
#     """
#     allowed_directions = [
#         Direction.Right,
#         Direction.Left,
#         Direction.Down,
#         Direction.DownRight,
#         Direction.DownLeft
#     ]
#     jump_sequences = []
    
#     for direction in allowed_directions:
#         adj_r = start.r + direction.r
#         adj_c = start.c + direction.c
#         if not (0 <= adj_r < BOARD_N and 0 <= adj_c < BOARD_N):
#             continue
#         adjacent = Coord(adj_r, adj_c)
#         if adjacent not in board or board[adjacent] != CellState.BLUE:
#             continue
#         jump_r = adj_r + direction.r
#         jump_c = adj_c + direction.c
#         if not (0 <= jump_r < BOARD_N and 0 <= jump_c < BOARD_N):
#             continue
#         jump_dest = Coord(jump_r, jump_c)
#         if jump_dest not in board or board[jump_dest] != CellState.LILY_PAD:
#             continue
        
#         jump_move = MoveAction(start, [direction, direction])
#         new_board = apply_move(board, jump_move)
        
#         subsequent_jump_sequences = find_jump_sequences(jump_dest, new_board)
#         if subsequent_jump_sequences:
#             for seq in subsequent_jump_sequences:
#                 jump_sequences.append([direction] + seq)
#         else:
#             jump_sequences.append([direction])
        
#     return jump_sequences

# def generate_valid_moves(board: dict[Coord, CellState]):
#     """
#     Generate all valid moves from the current board state.
#     """
#     moves = []
#     red_coord = None
#     for coord, cellState in board.items():
#         if cellState == CellState.RED:
#             red_coord = coord
#             break
#     if red_coord is None:
#         return moves
    
#     allowed_directions = [
#         Direction.Right,
#         Direction.Left,
#         Direction.Down,
#         Direction.DownRight,
#         Direction.DownLeft
#     ]
    
#     for direction in allowed_directions:
#         new_r = red_coord.r + direction.r
#         new_c = red_coord.c + direction.c
#         if not (0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N):
#             continue
#         simple_dest = Coord(new_r, new_c)
#         if simple_dest in board and board[simple_dest] == CellState.LILY_PAD:
#             moves.append(MoveAction(red_coord, [direction]))
        
#         jump_sequences = find_jump_sequences(red_coord, board)
#         for seq in jump_sequences:
#             expanded_seq = [d for d in seq for _ in (0, 1)]
#             moves.append(MoveAction(red_coord, expanded_seq))
    
#     return moves

# def apply_move(board: dict[Coord, CellState], move: MoveAction):
#     """
#     Apply the move to the board.
#     """
#     new_board = board.copy()
#     source = move.coord
#     dest = source
#     for direction in move.directions:
#         dest = dest + direction
#     if source in new_board and new_board[source] == CellState.RED:
#         del new_board[source]
#     if dest in new_board and new_board[dest] == CellState.LILY_PAD:
#         del new_board[dest]
#     new_board[dest] = CellState.RED
#     return new_board

# def a_star_search(board: dict[Coord, CellState]):
#     """
#     A* search implementation using the heuristic (vertical distance / constant).
#     """
#     heap = []
#     counter = itertools.count()  # Tie-breaker counter.
#     nodes_created = 1  # Start with the initial board.
#     nodes_explored = 0
    
#     initial_g = 0
#     initial_h = heuristic(board)
#     initial_f = initial_g + initial_h
#     heapq.heappush(heap, (initial_f, initial_g, next(counter), board, []))
    
#     best_cost = { board_to_tuple(board): initial_g }
    
#     while heap:
#         f, g, _, current_board, current_path = heapq.heappop(heap)
#         nodes_explored += 1
        
#         if goal_test(current_board):
#             print(f"A* Nodes created: {nodes_created}")
#             print(f"A* Nodes explored: {nodes_explored}")
#             return current_path
        
#         for move in generate_valid_moves(current_board):
#             new_board = apply_move(current_board, move)
#             new_board_key = board_to_tuple(new_board)
#             new_g = g + 1
#             if new_board_key not in best_cost or new_g < best_cost[new_board_key]:
#                 best_cost[new_board_key] = new_g
#                 new_h = heuristic(new_board)
#                 new_f = new_g + new_h
#                 heapq.heappush(heap, (new_f, new_g, next(counter), new_board, current_path + [move]))
#                 nodes_created += 1
    
#     print(f"A* Nodes created: {nodes_created}")
#     print(f"A* Nodes explored: {nodes_explored}")
#     return None


# def search(board: dict[Coord, CellState]) -> list[MoveAction] | None:
#     """
#     Entry point for the search. Finds a sequence of moves that moves the red frog to row 7.
#     """
#     print("Initial board:")
#     print(render_board(board, ansi=True))
    
#     start_time = time.time()
#     solution_path = a_star_search(board)
#     end_time = time.time()
    
#     if solution_path is None:
#         print("No solution found.")
#     else:
#         print(f"Solution found in {end_time - start_time:.4f} seconds.")
    
#     current_board = board
#     for i, move in enumerate(solution_path or [], start=1):
#         current_board = apply_move(current_board, move)
#         print(f"\nBoard after move {i} ({move}):")
#         print(render_board(current_board, ansi=True))
    
#     return solution_path
