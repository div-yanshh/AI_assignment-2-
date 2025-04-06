# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction
from .utils import render_board, bfs_search

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

    return solution_path