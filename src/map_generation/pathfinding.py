import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import random

class Node:
    """Lightweight A* node used for the heap only."""
    __slots__ = ("parent", "position", "g", "h", "f")
    def __init__(self, parent=None, position=None, g=0.0, h=0.0):
        self.parent = parent
        self.position = position  # (row, col)
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        # break ties by h (prefer nodes closer to goal)
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f

def heuristic(a, b):
    """Euclidean distance heuristic (admissible for 8-connect with sqrt(2) diag)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def astar(maze, start, goal, allow_diagonal=True, prevent_corner_cutting=True):
    """
    A* on a 2D grid.
    maze: 2D numpy array (0 = free, 1 = obstacle)
    start, goal: (row, col)
    prevent_corner_cutting: if True, disallow diagonal moves when orthogonal neighbors are blocked.
    Returns list of (row, col) or None if no path.
    """

    rows, cols = maze.shape

    # Validate start/goal are inside bounds
    def in_bounds(pos):
        return 0 <= pos[0] < rows and 0 <= pos[1] < cols

    if not in_bounds(start) or not in_bounds(goal):
        raise ValueError("Start/goal out of grid bounds")

    if maze[start[0], start[1]] != 0:
        raise ValueError("Start position is not free (obstacle).")
    if maze[goal[0], goal[1]] != 0:
        raise ValueError("Goal position is not free (obstacle).")

    if start == goal:
        return [start]

    # Movements: (dr, dc, cost)
    diag_cost = math.sqrt(2)
    moves = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)
    ]
    if allow_diagonal:
        moves += [(-1, -1, diag_cost), (-1, 1, diag_cost), (1, -1, diag_cost), (1, 1, diag_cost)]

    open_heap = []
    best_g = {}  # position -> best g so far
    closed = set()

    h0 = heuristic(start, goal)
    start_node = Node(parent=None, position=start, g=0.0, h=h0)
    heapq.heappush(open_heap, start_node)
    best_g[start] = 0.0

    while open_heap:
        current = heapq.heappop(open_heap)

        # Skip stale entry if we already have a better g for this position
        if best_g.get(current.position, float("inf")) < current.g:
            continue

        if current.position == goal:
            # reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node.position)
                node = node.parent
            return path[::-1]

        closed.add(current.position)

        for dr, dc, cost in moves:
            nbr = (current.position[0] + dr, current.position[1] + dc)

            if not in_bounds(nbr):
                continue
            if maze[nbr[0], nbr[1]] != 0:
                continue
            if nbr in closed:
                continue

            # Prevent corner cutting: if moving diagonally, ensure both orthogonal neighbors are free
            if prevent_corner_cutting and (dr != 0 and dc != 0):
                orth1 = (current.position[0] + dr, current.position[1])
                orth2 = (current.position[0], current.position[1] + dc)
                if (not in_bounds(orth1) or not in_bounds(orth2)) or maze[orth1[0], orth1[1]] != 0 or maze[orth2[0], orth2[1]] != 0:
                    continue

            tentative_g = current.g + cost

            # If we already found a better path to nbr, skip
            if tentative_g >= best_g.get(nbr, float("inf")):
                continue

            h = heuristic(nbr, goal)
            child = Node(parent=current, position=nbr, g=tentative_g, h=h)

            best_g[nbr] = tentative_g
            heapq.heappush(open_heap, child)

    return None  # no path found

def main():
    # 1. Load the Map (Raw 0/1 grid)
    try:
        grid = np.load('clean_map.npy')
    except FileNotFoundError:
        print("File 'clean_map.npy' not found. Make sure the file exists in the working directory.")
        return

    print(f"Loaded Map: {grid.shape}")

    free_indices = np.argwhere(grid == 0)
    if len(free_indices) < 2:
        print("Map is full of obstacles!")
        return

    # Option A: random start/end
    # start = tuple(free_indices[np.random.choice(len(free_indices))])
    # end = tuple(free_indices[np.random.choice(len(free_indices))])

    # Option B: manual override (only if those coordinates are free)
    # If you want to override, change these numbers; otherwise comment them out
    manual_start = (10, 10)
    manual_end   = (10, 75)

    if (0 <= manual_start[0] < grid.shape[0] and 0 <= manual_start[1] < grid.shape[1] and grid[manual_start] == 0):
        start = manual_start
    else:
        start = tuple(free_indices[random.randrange(len(free_indices))])
        print(f"Manual start invalid or blocked; picked random start {start}")

    if (0 <= manual_end[0] < grid.shape[0] and 0 <= manual_end[1] < grid.shape[1] and grid[manual_end] == 0):
        end = manual_end
    else:
        end = tuple(free_indices[random.randrange(len(free_indices))])
        print(f"Manual end invalid or blocked; picked random end {end}")

    print(f"Calculating path from {start} to {end}...")

    path = astar(grid, start, end, allow_diagonal=True, prevent_corner_cutting=True)

    if path is None:
        print("No path found! (Are start/end enclosed by walls?)")
        return

    print(f"Path found! Length: {len(path)} steps")

    # Visualization
    path_y = [p[0] for p in path]  # rows -> y
    path_x = [p[1] for p in path]  # cols -> x

    plt.figure(figsize=(8, 8))
    plt.imshow(1 - grid, cmap='gray', origin='lower')  # free=white, obstacle=black
    plt.plot(path_x, path_y, linewidth=2, label='A* Path')
    plt.scatter(start[1], start[0], marker='o', s=80, label='Start', zorder=5)
    plt.scatter(end[1], end[0], marker='X', s=80, label='Goal', zorder=5)
    plt.title(f"Navigation Path ({len(path)} steps)")
    plt.legend(loc='center right')
    plt.show()

if __name__ == "__main__":
    main()
