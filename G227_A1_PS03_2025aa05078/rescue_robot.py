"""
Rescue Robot - Greedy Best-First Search (GBFS) vs A*
Assignment 1 - PS3 | AIMLCZG557/AECLZG557
BITS Pilani, Work-Integrated Learning Programmes Division

PEAS Description:
  Performance : Maximize survivors rescued; minimize cost, toxic zones crossed, time, nodes explored.
  Environment : 8x8 discrete, static, deterministic, fully observable grid.
                Cells: Safe (.), Toxic (T), Blocked (#), Start (S), Survivor (P).
  Actuators   : Orthogonal movement (Up/Down/Left/Right); rescue tool (marks node as safe).
  Sensors     : Position tracker (row, col); environment scanner (adjacent cell types); survivor locator.

Usage:
  python rescue_robot.py
  Input  : inputPS3.txt  (grid file, optionally with leading row numbers)
  Output : outputPS3.txt (text log of all search steps)
"""

import copy
import heapq
import os
import time
import tracemalloc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

def read_input_file(filepath):
    """
    Reads the environment grid from a text file.
    Each line may optionally start with a row number (e.g. "1 S . . #..."),
    which is stripped before parsing.

    Args:
        filepath (str): Path to the text file containing the grid layout.

    Returns:
        list: A 2D list (matrix) representing the grid environment.
    """
    grid = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ',' in line:
                    parts = line.split(',')
                elif ' ' in line:
                    parts = line.split(' ')
                else:
                    parts = list(line)

                # Strip leading numeric row index if present
                if parts and parts[0].isdigit():
                    parts = parts[1:]

                grid.append(parts)
        return grid
    else:
        print(f"File '{filepath}' not found. Using built-in default grid.")
        return [
            ['S', '.', '.', '#', '.', 'T', '.', '.'],
            ['#', '.', 'T', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'T', '.', '#', '.'],
            ['.', '#', '#', '.', '.', '.', '.', '.'],
            ['T', '.', '.', '.', '#', '#', '.', 'P'],
            ['.', '.', '.', 'T', '.', '.', '.', '.'],
            ['.', '#', '.', '.', 'P', '#', 'T', '.'],
            ['.', '.', '.', 'T', '.', 'P', '.', '.'],
        ]


def print_grid(grid):
    """
    Prints the grid as text to the terminal.

    Args:
        grid (list): 2D list representing the grid.
    """
    for row in grid:
        print(" ".join(row))


def visualize_grid(grid, title="Rescue Robot Environment"):
    """
    Visualizes the grid graphically using Matplotlib.
    Color scheme:
      . -> white    # -> black    T -> orange
      P -> green    S/A -> blue   path -> lightblue

    Args:
        grid (list): 2D list representing the grid.
        title (str): Plot title.
    """
    from matplotlib.colors import ListedColormap

    color_map = {'.': 0, '#': 1, 'T': 2, 'P': 3, 'S': 4, 'A': 4, 'path': 5}
    rows, cols = len(grid), len(grid[0])
    grid_numeric = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            grid_numeric[r][c] = color_map.get(grid[r][c], 0)

    cmap = ListedColormap(['white', 'black', 'orange', 'green', 'blue', 'lightblue'])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(grid_numeric, cmap=cmap, vmin=0, vmax=5)

    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, grid[r][c], va='center', ha='center', fontweight='bold',
                    fontsize=12, color='white' if grid[r][c] in ['#', 'S'] else 'black')

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticklabels(np.arange(rows))
    ax.set_title(title)
    ax.grid(color='gray', linestyle='-', linewidth=1)

    legend_elements = [
        mpatches.Patch(color='blue',      label='S - Start/Agent'),
        mpatches.Patch(color='green',     label='P - Survivor'),
        mpatches.Patch(color='black',     label='# - Blocked'),
        mpatches.Patch(color='orange',    label='T - Toxic Zone'),
        mpatches.Patch(color='lightblue', label='path - Traversed'),
        mpatches.Patch(color='white',     label='. - Safe Passage'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1.0), fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. HEURISTICS & SEARCH HELPERS
# ---------------------------------------------------------------------------

def find_targets(grid):
    """
    Scans the grid to locate the Start position and all Survivor targets.

    Args:
        grid (list): The 2D grid matrix.

    Returns:
        tuple: (start_position, list_of_survivor_positions)
    """
    start = None
    survivors = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'P':
                survivors.append((r, c))
    return start, survivors


def get_adjacent_toxics(r, c, grid):
    """
    Counts orthogonally adjacent Toxic zones (T) around a cell.
    T(n) in {0, 1, 2, 3, 4}.

    Args:
        r (int): Row index.
        c (int): Column index.
        grid (list): The grid matrix.

    Returns:
        int: Number of toxic cells immediately adjacent to (r, c).
    """
    toxics = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] == 'T':
                toxics += 1
    return toxics


def manhattan(pos1, pos2):
    """Returns Manhattan distance between two (row, col) positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def heuristic_1(current, goal, grid=None):
    """
    Heuristic 1: Pure Manhattan Distance.
    h1(n) = |x_goal - x_current| + |y_goal - y_current|
    Admissible but ignores environmental hazards.
    """
    return manhattan(current, goal)


def heuristic_2(current, goal, grid):
    """
    Heuristic 2: Risk-Aware Heuristic.
    h2(n) = h1(n) + alpha * T(n)
    where alpha=2 and T(n) = number of orthogonally adjacent toxic cells.
    Penalizes proximity to toxic zones to encourage safer routing.
    """
    return manhattan(current, goal) + 2 * get_adjacent_toxics(current[0], current[1], grid)


def get_neighbors(r, c, grid):
    """
    Returns valid orthogonal neighbors (Up, Down, Left, Right).
    Excludes out-of-bounds and Blocked (#) cells.

    Args:
        r (int): Row index.
        c (int): Column index.
        grid (list): The grid matrix.

    Returns:
        list: Valid (row, col) neighbor coordinates.
    """
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] != '#':
                neighbors.append((nr, nc))
    return neighbors


def get_closest_target(current, targets, heuristic_fn, grid):
    """
    Selects the closest remaining survivor using the given heuristic.
    Prints heuristic estimates for all remaining survivors.

    Args:
        current (tuple): Current agent position.
        targets (list): Remaining survivor positions.
        heuristic_fn (function): Heuristic function to use.
        grid (list): The grid matrix.

    Returns:
        tuple: Position of the closest survivor.
    """
    best_target = None
    min_h = float('inf')

    print(f"\n--- Re-evaluating closest target from {current} ---")
    for t in targets:
        h_val = heuristic_fn(current, t, grid)
        print(f"  Heuristic estimate from {current} to Survivor {t}: {h_val}")
        if h_val < min_h:
            min_h = h_val
            best_target = t

    print(f">> Selected Target: {best_target} with h(n) = {min_h}")
    return best_target


# ---------------------------------------------------------------------------
# 3. GREEDY BEST-FIRST SEARCH (GBFS)
# ---------------------------------------------------------------------------

def gbfs(grid, start, target, heuristic_fn):
    """
    Executes Greedy Best-First Search from start to target.
    Uses a min-heap (priority queue) keyed by heuristic value h(n).

    Tracks at every iteration:
      - Full frontier node list with heuristic values
      - Explored nodes set
      - Selected node and its heuristic value
      - Toxic zones crossed along the path

    Memory measured via tracemalloc (true peak heap allocation).

    Args:
        grid (list): Environment grid matrix.
        start (tuple): Starting position.
        target (tuple): Goal/Survivor position.
        heuristic_fn (function): Heuristic function to guide search.

    Returns:
        dict: {success, path, toxic_crossed, explored_count, search_log,
               tries, exec_time, mem_peak_bytes}
    """
    tracemalloc.start()
    start_time = time.perf_counter()

    # Heap entries: (heuristic_val, insertion_order, position, path, toxics_crossed)
    pq = []
    insertion_order = 0
    initial_h = heuristic_fn(start, target, grid)
    heapq.heappush(pq, (initial_h, insertion_order, start, [start], 0))

    explored = set()
    search_log = []
    tries = 0

    while pq:
        tries += 1
        # Capture full frontier BEFORE popping: list of (node_pos, h_value)
        frontier_nodes = [(node[2], node[0]) for node in pq]

        eval_h, _, current, path, toks = heapq.heappop(pq)

        search_log.append({
            'iteration':      tries,
            'frontier':       frontier_nodes,
            'selected_node':  current,
            'heuristic_val':  eval_h,
            'explored_so_far': list(explored),
        })

        if current == target:
            end_time = time.perf_counter()
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {
                'success':        True,
                'path':           path,
                'toxic_crossed':  toks,
                'explored_count': len(explored),
                'search_log':     search_log,
                'tries':          tries,
                'exec_time':      end_time - start_time,
                'mem_peak_bytes': mem_peak,
            }

        if current in explored:
            continue

        explored.add(current)

        for nx, ny in get_neighbors(current[0], current[1], grid):
            if (nx, ny) not in explored:
                toks_added = 1 if grid[nx][ny] == 'T' else 0
                insertion_order += 1
                heapq.heappush(pq, (
                    heuristic_fn((nx, ny), target, grid),
                    insertion_order,
                    (nx, ny),
                    path + [(nx, ny)],
                    toks + toks_added,
                ))

    end_time = time.perf_counter()
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'success':        False,
        'explored_count': len(explored),
        'search_log':     search_log,
        'tries':          tries,
        'exec_time':      end_time - start_time,
        'mem_peak_bytes': mem_peak,
    }


# ---------------------------------------------------------------------------
# 4. MISSION EXECUTION (GBFS)
# ---------------------------------------------------------------------------

def execute_rescue_mission(initial_grid, heuristic_fn, heuristic_name, output_file=None):
    """
    Runs the full sequential rescue mission using GBFS with the given heuristic.

    Loop:
      1. Select closest remaining survivor via heuristic.
      2. Run GBFS from current position to that survivor.
      3. Print/log: frontier, explored, heuristic values, path, metrics.
      4. Mark survivor node as safe (.), update agent position.
      5. Repeat until all survivors rescued or one is unreachable.

    Args:
        initial_grid (list): The starting grid (not mutated).
        heuristic_fn (function): Heuristic function to use.
        heuristic_name (str): Display name for logging.
        output_file (str, optional): Path to append text output log.

    Returns:
        list: Per-rescue result dicts with path, metrics, and grid snapshots.
    """
    grid = copy.deepcopy(initial_grid)
    current_pos, remaining_survivors = find_targets(grid)

    header = f"\n{'='*60}\n COMMENCING MISSION USING {heuristic_name.upper()}\n{'='*60}"
    print(header)
    output_lines = [header]
    mission_log = []

    while remaining_survivors:
        target = get_closest_target(current_pos, remaining_survivors, heuristic_fn, grid)

        start_msg = f"\n---> Starting GBFS from {current_pos} to {target}..."
        print(start_msg)
        output_lines.append(start_msg)

        result = gbfs(grid, current_pos, target, heuristic_fn)

        if result['success']:
            msgs = [
                f"SUCCESS: Reached Survivor {target} in {len(result['path'])-1} steps!",
                f" - Toxic Zones Crossed : {result['toxic_crossed']}",
                f" - Explored Nodes      : {result['explored_count']}",
                f" - Total Loop Tries    : {result['tries']}",
                f" - Exec Time           : {result['exec_time'] * 1000:.4f} ms",
                f" - Peak Memory Usage   : {result['mem_peak_bytes']} bytes",
                f" - Path: {' -> '.join(str(p) for p in result['path'])}",
            ]
            for m in msgs:
                print(m)
                output_lines.append(m)

            # Print full frontier + explored at each iteration (Deliverable 3d/3e)
            for log in result['search_log']:
                frontier_str = ', '.join(f"{n}(h={h})" for n, h in log['frontier'])
                explored_str = ', '.join(str(n) for n in log['explored_so_far']) or 'none'
                log_msg = (
                    f"   [Iter {log['iteration']:>2}] Selected: {log['selected_node']}  "
                    f"h={log['heuristic_val']}\n"
                    f"             Frontier ({len(log['frontier'])}): [{frontier_str}]\n"
                    f"             Explored ({len(log['explored_so_far'])}): [{explored_str}]"
                )
                print(log_msg)
                output_lines.append(log_msg)

            # Grid snapshot BEFORE rescue (for path visualization)
            grid_before = copy.deepcopy(grid)

            mission_log.append({
                'target':          target,
                'path':            result['path'],
                'toxic_crossed':   result['toxic_crossed'],
                'explored_count':  result['explored_count'],
                'tries':           result['tries'],
                'exec_time':       result['exec_time'],
                'mem_peak_bytes':  result['mem_peak_bytes'],
                'search_log':      result['search_log'],
                'grid_snapshot':   grid_before,
            })

            # Mark survivor node as safe passage and advance agent
            current_pos = target
            remaining_survivors.remove(target)
            grid[target[0]][target[1]] = '.'

            # Grid snapshot AFTER rescue (for updated-grid visual, Deliverable 9b)
            mission_log[-1]['grid_after_rescue'] = copy.deepcopy(grid)

        else:
            fail_msg = f"FAILURE: Agent trapped! Couldn't reach Survivor {target}."
            print(fail_msg)
            output_lines.append(fail_msg)
            remaining_survivors.remove(target)

    end_msg = f"\nMission '{heuristic_name}' Ended."
    print(end_msg)
    output_lines.append(end_msg)

    if output_file:
        mode = 'a' if os.path.exists(output_file) else 'w'
        with open(output_file, mode) as f:
            f.write('\n'.join(output_lines) + '\n')

    return mission_log


# ---------------------------------------------------------------------------
# 5. VISUALIZATIONS (Deliverables 9a, 9b, 4/10, 5b, 5c)
# ---------------------------------------------------------------------------

def visualize_mission(mission_results, title="Mission Analysis"):
    """
    Deliverables 9a & 9b:
      - For each rescue: plots the path taken on the grid.
      - Between rescues: plots the updated grid (survivor → safe '.').

    Args:
        mission_results (list): Output from execute_rescue_mission.
        title (str): Plot title prefix.
    """
    for idx, res in enumerate(mission_results):
        grid_state = copy.deepcopy(res['grid_snapshot'])
        for r, c in res['path']:
            if grid_state[r][c] not in ['S', 'P', '#', 'T']:
                grid_state[r][c] = 'path'
        grid_state[res['path'][0][0]][res['path'][0][1]] = 'S'
        grid_state[res['path'][-1][0]][res['path'][-1][1]] = 'P'
        visualize_grid(grid_state, f"{title}: Rescue {idx+1} — Path to Survivor {res['target']}")

        # Show updated grid before the next rescue begins
        if idx < len(mission_results) - 1:
            visualize_grid(
                copy.deepcopy(res['grid_after_rescue']),
                f"{title}: Grid After Rescue {idx+1} (Survivor {res['target']} → safe '.')",
            )


def print_complexity_table(results_H1, results_H2):
    """
    Deliverables 4 & 10: Prints complexity metrics table for H1 vs H2.

    Args:
        results_H1 (list): Mission log for Heuristic 1.
        results_H2 (list): Mission log for Heuristic 2.
    """
    rows = []
    for idx in range(len(results_H1)):
        rows.append({
            'Rescue #':            idx + 1,
            'H1 Target':           results_H1[idx]['target'],
            'H1 Path Length':      len(results_H1[idx]['path']) - 1,
            'H1 Explored Nodes':   results_H1[idx]['explored_count'],
            'H1 Exec Time (ms)':   round(results_H1[idx]['exec_time'] * 1000, 3),
            'H1 Peak Mem (bytes)': results_H1[idx]['mem_peak_bytes'],
            'H1 Toxics Crossed':   results_H1[idx]['toxic_crossed'],
            'H2 Target':           results_H2[idx]['target'],
            'H2 Path Length':      len(results_H2[idx]['path']) - 1,
            'H2 Explored Nodes':   results_H2[idx]['explored_count'],
            'H2 Exec Time (ms)':   round(results_H2[idx]['exec_time'] * 1000, 3),
            'H2 Peak Mem (bytes)': results_H2[idx]['mem_peak_bytes'],
            'H2 Toxics Crossed':   results_H2[idx]['toxic_crossed'],
        })
    df = pd.DataFrame(rows)
    print("\n--- Deliverable 4/10: Mission Summary & Complexity Metrics ---")
    print(df.to_string(index=False))


def plot_heuristic_heatmap(grid, heuristic_fn, target, heuristic_name):
    """
    Deliverable 5b: Spatial heatmap of h(n) values across the grid for a given target.
    Blocked cells are shown in black; each passable cell shows its cell type and h(n).

    Args:
        grid (list): The grid matrix.
        heuristic_fn (function): Heuristic function.
        target (tuple): Survivor position.
        heuristic_name (str): Label for plot title.
    """
    rows, cols = len(grid), len(grid[0])
    h_grid = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != '#':
                h_grid[r][c] = heuristic_fn((r, c), target, grid)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(h_grid, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='h(n) value')

    max_h = np.nanmax(h_grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '#':
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
            else:
                val = h_grid[r][c]
                text_color = 'white' if val >= max_h * 0.7 else 'black'
                ax.text(c, r, f"{grid[r][c]}\n{int(val)}", ha='center', va='center',
                        fontsize=7, color=text_color)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title(f"{heuristic_name} — Heuristic Values toward Survivor {target}")
    plt.tight_layout()
    plt.show()


def plot_heuristic_convergence(results_H1, results_H2):
    """
    Deliverable 5c: Line chart of h(n) values vs iteration step for all rescues,
    both heuristics. Shows convergence speed and trap spikes.

    Args:
        results_H1 (list): Mission log for Heuristic 1.
        results_H2 (list): Mission log for Heuristic 2.
    """
    plt.figure(figsize=(12, 5))
    for i, res in enumerate(results_H1):
        h_vals = [log['heuristic_val'] for log in res['search_log']]
        plt.plot(range(1, len(h_vals) + 1), h_vals,
                 label=f"H1 - Rescue {i+1} → {res['target']}", marker='o', linestyle='dashed')

    for i, res in enumerate(results_H2):
        h_vals = [log['heuristic_val'] for log in res['search_log']]
        plt.plot(range(1, len(h_vals) + 1), h_vals,
                 label=f"H2 - Rescue {i+1} → {res['target']}", marker='x', linestyle='dotted')

    plt.title('Deliverable 5c: Heuristic Value h(n) vs Iteration Step')
    plt.xlabel('Iteration Step')
    plt.ylabel('Heuristic Value h(n)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. TRAP ANALYSIS (Deliverable 6)
# ---------------------------------------------------------------------------

def analyze_traps(mission_results, heuristic_id):
    """
    Deliverable 6: Identifies trap situations — iterations where GBFS was forced
    to expand a node with a HIGHER h(n) than the best seen so far, indicating a
    local minimum (dead-end or sub-optimal pocket).

    Each trap record includes:
      - Heuristic label (H1/H2)
      - Heuristic value at trap entry
      - Best h(n) seen before the trap
      - Node where the regression started (trapped_at)
      - Iteration number of trap entry
      - Iteration number when h(n) improved again (escape)
      - Target survivor

    Args:
        mission_results (list): Output from execute_rescue_mission.
        heuristic_id (str): 'H1' or 'H2'.

    Returns:
        list: List of trap dictionaries.
    """
    traps = []
    for run in mission_results:
        logs = run['search_log']
        target = run['target']
        best_h = float('inf')
        trap_entry = None

        for i, step in enumerate(logs):
            curr_h = step['heuristic_val']

            if curr_h < best_h:
                if trap_entry is not None:
                    trap_entry['Next Iteration Out'] = step['iteration']
                    traps.append(trap_entry)
                    trap_entry = None
                best_h = curr_h

            elif curr_h > best_h and trap_entry is None:
                trapped_node = logs[i - 1]['selected_node'] if i > 0 else step['selected_node']
                trap_entry = {
                    'Heuristic':              heuristic_id,
                    'Heuristic Value (trap)': curr_h,
                    'Best h(n) Before Trap':  best_h,
                    'Trapped At Node':        trapped_node,
                    'Trap Iteration':         step['iteration'],
                    'Next Iteration Out':     'still in trap',
                    'Target':                 target,
                }

        if trap_entry is not None:
            trap_entry['Next Iteration Out'] = logs[-1]['iteration']
            traps.append(trap_entry)

    return traps


def print_trap_analysis(results_H1, results_H2):
    """
    Prints the Deliverable 6 trap table for both heuristics.

    Args:
        results_H1 (list): Mission log for Heuristic 1.
        results_H2 (list): Mission log for Heuristic 2.
    """
    trap_h1 = analyze_traps(results_H1, "H1")
    trap_h2 = analyze_traps(results_H2, "H2")
    all_traps = trap_h1 + trap_h2

    print(f"\nH1 traps found: {len(trap_h1)}")
    print(f"H2 traps found: {len(trap_h2)}")

    if not all_traps:
        print("\nNo traps observed: GBFS expanded nodes monotonically for both heuristics on this map.")
        print("Neither heuristic encountered a local minimum requiring backtracking.\n")
    else:
        df = pd.DataFrame(all_traps)
        print("\n--- Deliverable 6: Trap / Local Minima Table ---")
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# 7. A* SEARCH & COMPARISON (Deliverable 7)
# ---------------------------------------------------------------------------

def a_star(grid, start, target, heuristic_fn):
    """
    Executes A* Search using f(n) = g(n) + h(n).
    Guarantees optimal paths on unit-cost grids with an admissible heuristic.

    Args:
        grid (list): Environment grid matrix.
        start (tuple): Starting position.
        target (tuple): Goal position.
        heuristic_fn (function): Admissible heuristic function (h).

    Returns:
        dict: {success, path, toxic_crossed, explored_count, cost}
    """
    pq = []
    insertion_order = 0
    initial_h = heuristic_fn(start, target, grid)
    # Heap entries: (f_val, insertion_order, position, path, g_cost, toxics_crossed)
    heapq.heappush(pq, (initial_h, insertion_order, start, [start], 0, 0))

    explored = set()

    while pq:
        f_val, _, current, path, g_cost, toks = heapq.heappop(pq)

        if current == target:
            return {
                'success':        True,
                'path':           path,
                'toxic_crossed':  toks,
                'explored_count': len(explored),
                'cost':           g_cost,
            }

        if current in explored:
            continue

        explored.add(current)

        for nx, ny in get_neighbors(current[0], current[1], grid):
            if (nx, ny) not in explored:
                toks_added = 1 if grid[nx][ny] == 'T' else 0
                new_g = g_cost + 1
                insertion_order += 1
                heapq.heappush(pq, (
                    new_g + heuristic_fn((nx, ny), target, grid),
                    insertion_order,
                    (nx, ny),
                    path + [(nx, ny)],
                    new_g,
                    toks + toks_added,
                ))

    return {'success': False, 'explored_count': len(explored), 'cost': float('inf')}


def execute_astar_mission(initial_grid):
    """
    Runs the full sequential rescue mission using A* with Heuristic 1.
    Uses the same greedy target-selection as GBFS for a fair comparison.

    Returns:
        list: Per-rescue result dicts.
    """
    grid = copy.deepcopy(initial_grid)
    current_pos, remaining_survivors = find_targets(grid)
    mission_log = []

    while remaining_survivors:
        target = get_closest_target(current_pos, remaining_survivors, heuristic_1, grid)
        result = a_star(grid, current_pos, target, heuristic_1)

        if result['success']:
            mission_log.append({
                'target':   target,
                'path_len': len(result['path']) - 1,
                'explored': result['explored_count'],
                'cost':     result['cost'],
                'success':  True,
            })
            current_pos = target
            remaining_survivors.remove(target)
            grid[target[0]][target[1]] = '.'
        else:
            mission_log.append({'target': target, 'success': False})
            remaining_survivors.remove(target)

    return mission_log


def print_gbfs_vs_astar(results_H1, astar_results):
    """
    Deliverable 7: Prints comparison table and bar chart of GBFS H1 vs A* H1.
    Includes completeness and optimality columns.

    Args:
        results_H1 (list): GBFS mission log (Heuristic 1).
        astar_results (list): A* mission log.
    """
    comp_data = []
    for i in range(len(results_H1)):
        comp_data.append({
            'Rescue #':             i + 1,
            'Target':               results_H1[i]['target'],
            'GBFS H1 Explored':     results_H1[i]['explored_count'],
            'A* H1 Explored':       astar_results[i].get('explored', '-'),
            'GBFS H1 Path Len':     len(results_H1[i]['path']) - 1,
            'A* H1 Path Len':       astar_results[i].get('path_len', '-'),
            'GBFS Complete?':       'Yes (finite + visited set)',
            'A* Complete?':         'Yes (finite + visited set)',
            'GBFS Optimal?':        'Not guaranteed (greedy)',
            'A* Optimal?':          'Yes (f = g + h, admissible h)',
        })

    df = pd.DataFrame(comp_data)
    print("\n--- Deliverable 7: GBFS vs A* Comparison ---")
    print(df.to_string(index=False))

    # Bar chart: nodes explored per rescue
    labels      = [f"Rescue {r['Rescue #']}\n→{r['Target']}" for r in comp_data]
    gbfs_nodes  = [r['GBFS H1 Explored'] for r in comp_data]
    astar_nodes = [r['A* H1 Explored']   for r in comp_data]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, gbfs_nodes,  width, label='GBFS H1', color='steelblue')
    bars2 = ax.bar(x + width / 2, astar_nodes, width, label='A* H1',   color='darkorange')
    ax.set_xlabel('Rescue Sequence')
    ax.set_ylabel('Nodes Explored')
    ax.set_title('Deliverable 7: GBFS vs A* — Nodes Explored per Rescue')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def main():
    input_file  = "inputPS3.txt"
    output_file = input_file.replace('input', 'output')

    # Clear previous output
    if os.path.exists(output_file):
        os.remove(output_file)

    # --- Load and display initial grid ---
    initial_grid = read_input_file(input_file)
    print("Initial Text Grid:")
    print_grid(initial_grid)
    visualize_grid(initial_grid, title="Initial Earthquake Environment")

    # --- GBFS with Heuristic 1 (Manhattan) ---
    results_H1 = execute_rescue_mission(
        initial_grid, heuristic_1, "Heuristic 1 (Manhattan)", output_file
    )

    # --- GBFS with Heuristic 2 (Risk-Aware) ---
    results_H2 = execute_rescue_mission(
        initial_grid, heuristic_2, "Heuristic 2 (Risk-Aware)", output_file
    )

    # --- Deliverables 9a & 9b: Path visualizations + updated grids ---
    print("\n--- H1: PATH VISUALIZATIONS ---")
    visualize_mission(results_H1, "H1 (Manhattan)")

    print("\n--- H2: PATH VISUALIZATIONS ---")
    visualize_mission(results_H2, "H2 (Risk-Aware)")

    # --- Deliverables 4 & 10: Complexity metrics table ---
    print_complexity_table(results_H1, results_H2)

    # --- Deliverable 5b: Heuristic value heatmaps ---
    print("\n--- Deliverable 5b: Heuristic Value Heatmaps ---")
    for target in [(6, 4), (7, 5), (4, 7)]:
        plot_heuristic_heatmap(initial_grid, heuristic_1, target, "H1 (Manhattan Distance)")
        plot_heuristic_heatmap(initial_grid, heuristic_2, target, "H2 (Risk-Aware)")

    # --- Deliverable 5c: Heuristic convergence graph ---
    plot_heuristic_convergence(results_H1, results_H2)

    # --- Deliverable 6: Trap analysis ---
    print_trap_analysis(results_H1, results_H2)

    # --- Deliverable 7: A* comparison ---
    print("\n--- Running A* with Heuristic 1 for comparison ---")
    astar_results = execute_astar_mission(initial_grid)
    print_gbfs_vs_astar(results_H1, astar_results)

    print(f"\nDone. Text output written to: {output_file}")


if __name__ == "__main__":
    main()
