import os
import sys
import time
import copy
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================
# Step 2: Environment Setup & IO
# ==========================================

def read_input_file(filepath):
    """
    Reads the environment grid from a text file.
    
    Args:
        filepath (str): The path to the text file containing the grid layout.
        
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
                    grid.append(line.split(','))
                elif ' ' in line:
                    grid.append(line.split(' '))
                else:
                    grid.append(list(line))
        return grid
    else:
        print(f"File {filepath} not found. Returning a default grid.")
        return [
            ['S', '.', '.', '#', '.', 'T', '.', '.'],
            ['#', '.', 'T', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', 'T', '.', '#', '.'],
            ['.', '#', '#', '.', '.', '.', '.', '.'],
            ['T', '.', '.', '.', '#', '#', '.', 'P'],
            ['.', '.', '.', 'T', '.', '.', '.', '.'],
            ['.', '#', '.', '.', 'P', '#', 'T', '.'],
            ['.', '.', '.', 'T', '.', 'P', '.', '.']
        ]

def print_grid(grid):
    """
    Prints the given grid continuously to the terminal.
    
    Args:
        grid (list): 2D list representing the grid to be printed.
    """
    for row in grid:
        print(" ".join(row))

def visualize_grid(grid, title="Rescue Robot Environment"):
    """
    Visualizes the grid environment graphically using Matplotlib.
    
    Args:
        grid (list): 2D list representing the grid.
        title (str, optional): Title of the plot.
    """
    color_map = {'.': 0, '#': 1, 'T': 2, 'P': 3, 'S': 4, 'A': 4, 'path': 5}
    grid_numeric = np.zeros((len(grid), len(grid[0])))
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            val = grid[r][c]
            grid_numeric[r][c] = color_map.get(val, 0)
    
    cmap = ListedColormap(['white', 'black', 'orange', 'green', 'blue', 'lightblue'])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(grid_numeric, cmap=cmap, vmin=0, vmax=5)
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            ax.text(c, r, grid[r][c], va='center', ha='center', fontweight='bold', fontsize=12,
                    color='white' if grid[r][c] in ['#', 'S'] else 'black')
            
    ax.set_xticks(np.arange(len(grid[0])))
    ax.set_yticks(np.arange(len(grid)))
    ax.set_xticklabels(np.arange(len(grid[0])))
    ax.set_yticklabels(np.arange(len(grid)))
    ax.set_title(title)
    ax.grid(color='gray', linestyle='-', linewidth=1)
    plt.show()

# ==========================================
# Step 3: Heuristics & Core Search Setup
# ==========================================

def find_targets(grid):
    """
    Scans the grid to locate the Start agent and all Survivor targets.
    
    Args:
        grid (list): The 2D tracking matrix.
        
    Returns:
        tuple: (start_position_tuple, list_of_survivor_tuples)
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
    Calculates orthogonally adjacent Toxic zones.
    """
    toxics = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] == 'T':
                toxics += 1
    return toxics

def manhattan(pos1, pos2):
    """Computes Manhattan Distance."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def heuristic_1(current, goal, grid=None):
    """Heuristic 1: Simple Manhattan Distance."""
    return manhattan(current, goal)

def heuristic_2(current, goal, grid):
    """Heuristic 2: Risk-Aware calculation modifying Pure Distance."""
    h1 = manhattan(current, goal)
    t_n = get_adjacent_toxics(current[0], current[1], grid)
    return h1 + 2 * t_n

def get_neighbors(r, c, grid):
    """Retrieves valid orthogonal neighbors."""
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if grid[nr][nc] != '#':
                neighbors.append((nr, nc))
    return neighbors

def get_closest_target(current, targets, heuristic_fn, grid):
    """Locates the optimal survivor based on the chosen heuristic."""
    best_target = None
    min_h = float('inf')
    
    print(f"\n--- Re-evaluating closest target from {current} ---")
    for t in targets:
        h_val = heuristic_fn(current, t, grid)
        print(f"Heuristic estimate from {current} to Survivor {t}: {h_val}")
        if h_val < min_h:
            min_h = h_val
            best_target = t
            
    print(f">> Selected Target: {best_target} with h(n) = {min_h}")
    return best_target

def gbfs(grid, start, target, heuristic_fn):
    """
    Executes Greedy Best-First Search tracking times and nodes logically.
    """
    start_time = time.perf_counter()
    pq = []
    insertion_order = 0
    initial_h = heuristic_fn(start, target, grid)
    heapq.heappush(pq, (initial_h, insertion_order, start, [start], 0))
    
    explored = set()
    search_log = []
    tries = 0 
    
    while pq:
        tries += 1
        frontier_repr = [((node[2]), f"h={node[0]}") for node in pq]
        eval_h, _, current, path, toks = heapq.heappop(pq)
        
        search_log.append({
            'iteration': tries,
            'frontier': frontier_repr,
            'selected_node': current,
            'heuristic_val': eval_h,
            'explored_so_far': tuple(explored)
        })
        
        if current == target:
            end_time = time.perf_counter()
            mem_usage = sys.getsizeof(explored) + sys.getsizeof(pq)
            return {
                'success': True,
                'path': path,
                'toxic_crossed': toks,
                'explored_count': len(explored),
                'search_log': search_log,
                'tries': tries,
                'exec_time': end_time - start_time,
                'mem_usage': mem_usage
            }
            
        if current in explored:
            continue
            
        explored.add(current)
        
        for nx, ny in get_neighbors(current[0], current[1], grid):
            if (nx, ny) not in explored:
                toks_added = 1 if grid[nx][ny] == 'T' else 0
                new_path = path + [(nx, ny)]
                new_h = heuristic_fn((nx, ny), target, grid)
                
                insertion_order += 1
                heapq.heappush(pq, (new_h, insertion_order, (nx, ny), new_path, toks + toks_added))
                
    end_time = time.perf_counter()
    mem_usage = sys.getsizeof(explored) + sys.getsizeof(pq)
    return {
        'success': False,
        'explored_count': len(explored),
        'search_log': search_log,
        'tries': tries,
        'exec_time': end_time - start_time,
        'mem_usage': mem_usage
    }

# ==========================================
# Step 4: Mission Execution
# ==========================================

def execute_rescue_mission(initial_grid, heuristic_fn, heuristic_name, output_file=None):
    """
    Simulates the global target gathering mission iteratively over valid survivors.
    """
    import copy
    grid = copy.deepcopy(initial_grid)
    current_pos, remaining_survivors = find_targets(grid)
    
    header = f"\n" + "="*60 + f"\n COMMENCING MISSION USING {heuristic_name.upper()}\n" + "="*60
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
            succ_msgs = [
                f"SUCCESS: Reached Survivor {target} in {len(result['path'])-1} steps!",
                f" - Toxic Zones Crossed: {result['toxic_crossed']}",
                f" - Explored Nodes: {result['explored_count']}",
                f" - Total Loop Tries: {result['tries']}",
                f" - Exec Time: {result['exec_time'] * 1000:.4f} ms | Mem Space: {result['mem_usage']} bytes",
                f" - Path: {' -> '.join([str(p) for p in result['path']])}"
            ]
            for msg in succ_msgs:
                print(msg)
                output_lines.append(msg)
            
            for log in result['search_log']:
                log_msg = f"   [Iter {log['iteration']}] Selected: {log['selected_node']}, h={log['heuristic_val']} | Frontier Size: {len(log['frontier'])}"
                output_lines.append(log_msg)
                
            mission_log.append({
                'target': target,
                'path': result['path'],
                'toxic_crossed': result['toxic_crossed'],
                'explored_count': result['explored_count'],
                'tries': result['tries'],
                'exec_time': result['exec_time'],
                'mem_usage': result['mem_usage'],
                'search_log': result['search_log'],
                'grid_snapshot': copy.deepcopy(grid)
            })
            
            current_pos = target
            remaining_survivors.remove(target)
            grid[target[0]][target[1]] = '.' 
            
        else:
            fail_msg = f"FAILURE: Agent trapped! Couldn't reach Survivor {target}."
            print(fail_msg)
            output_lines.append(fail_msg)
            break
            
    end_msg = f"\nMission '{heuristic_name}' Ended."
    print(end_msg)
    output_lines.append(end_msg)
    
    if output_file:
        mode = 'a' if os.path.exists(output_file) else 'w'
        with open(output_file, mode) as f:
            f.write('\n'.join(output_lines) + '\n')
            
    return mission_log

def visualize_mission(mission_results, title="Mission Analysis"):
    """ Visualizes path for all tracked valid missions. """
    for idx, res in enumerate(mission_results):
        grid_state = res['grid_snapshot']
        for p in res['path']:
            r, c = p
            if grid_state[r][c] not in ['S', 'P', '#']:
                grid_state[r][c] = 'path'
        grid_state[res['path'][0][0]][res['path'][0][1]] = 'S' 
        grid_state[res['path'][-1][0]][res['path'][-1][1]] = 'P' 
        visualize_grid(grid_state, f"{title}: Rescue {idx+1} [Survivor {res['target']}]")

def analyze_traps(mission_results, heuristic_id):
    """ Determines Traps (deadends where greedy logic reverses geometrically). """
    traps = []
    for run in mission_results:
        target = run['target']
        logs = run['search_log']
        min_h_seen = float('inf')
        for i, step in enumerate(logs):
            curr_h = step['heuristic_val']
            if curr_h > min_h_seen and curr_h != min_h_seen:
                traps.append({
                    'Heuristic': heuristic_id,
                    'Heuristic_Value': curr_h,
                    'Trapped_at_Node': logs[i-1]['selected_node'] if i > 0 else None,
                    'Next_Iteration_to_Come_Out': i + 1,
                    'Target': target
                })
            if curr_h < min_h_seen:
                min_h_seen = curr_h
    return traps

def a_star(grid, start, target, heuristic_fn):
    """Executes robust optimal A* Pathfinding logic."""
    pq = []
    insertion_order = 0
    initial_h = heuristic_fn(start, target, grid)
    heapq.heappush(pq, (initial_h, insertion_order, start, [start], 0, 0))
    explored = set()
    tries = 0
    
    while pq:
        tries += 1
        f_val, _, current, path, g_cost, toks = heapq.heappop(pq)
        
        if current == target:
            return {
                'success': True,
                'path': path,
                'toxic_crossed': toks,
                'explored_count': len(explored),
                'cost': g_cost
            }
            
        if current in explored:
            continue
            
        explored.add(current)
        
        for nx, ny in get_neighbors(current[0], current[1], grid):
            if (nx, ny) not in explored:
                toks_added = 1 if grid[nx][ny] == 'T' else 0
                new_path = path + [(nx, ny)]
                new_g = g_cost + 1
                new_f = new_g + heuristic_fn((nx, ny), target, grid)
                insertion_order += 1
                heapq.heappush(pq, (new_f, insertion_order, (nx, ny), new_path, new_g, toks + toks_added))
                
    return {'success': False, 'explored_count': len(explored)}

def execute_astar_mission(initial_grid):
    """A* looping logic testing tracking."""
    grid = copy.deepcopy(initial_grid)
    current_pos, remaining_survivors = find_targets(grid)
    mission_log = []
    
    while remaining_survivors:
        target = get_closest_target(current_pos, remaining_survivors, heuristic_1, grid)
        result = a_star(grid, current_pos, target, heuristic_1)
        if result['success']:
            mission_log.append({
                'target': target,
                'path_len': len(result['path']) - 1,
                'explored': result['explored_count']
            })
            current_pos = target
            remaining_survivors.remove(target)
            grid[target[0]][target[1]] = '.'
        else:
            break
    return mission_log

# ==========================================
# Main Execution Flow
# ==========================================

def main():
    # 1. Read input and setup output
    input_file = "inputPS1.txt"
    output_file = input_file.replace('input', 'output')
    
    initial_grid = read_input_file(input_file)
    print("Initial Text Grid:")
    print_grid(initial_grid)
    visualize_grid(initial_grid, title="Initial Earthquake Environment")
    
    if os.path.exists(output_file):
        os.remove(output_file)

    # 2. Execute Missions
    results_H1 = execute_rescue_mission(initial_grid, heuristic_1, "Heuristic 1 (Manhattan)", output_file)
    results_H2 = execute_rescue_mission(initial_grid, heuristic_2, "Heuristic 2 (Risk-Aware)", output_file)

    # 3. Visualizations
    print("--- HEURISTIC 1 PATHS ---")
    visualize_mission(results_H1, "H1 (Manhattan)")

    print("--- HEURISTIC 2 PATHS ---")
    visualize_mission(results_H2, "H2 (Risk-Aware)")

    # 4. Comparative Analysis
    df_rows = []
    runs = min(len(results_H1), len(results_H2))
    for idx in range(runs):
        df_rows.append({
            'Rescue Sequence': idx + 1,
            'H1_Target': results_H1[idx]['target'],
            'H1_Path_Length': len(results_H1[idx]['path']) - 1,
            'H1_Explored_Nodes': results_H1[idx]['explored_count'],
            'H1_Exec_Time(ms)': round(results_H1[idx]['exec_time'] * 1000, 3),
            'H1_Mem(bytes)': results_H1[idx]['mem_usage'],
            'H1_Toxics': results_H1[idx]['toxic_crossed'],
            
            'H2_Target': results_H2[idx]['target'],
            'H2_Path_Length': len(results_H2[idx]['path']) - 1,
            'H2_Explored_Nodes': results_H2[idx]['explored_count'],
            'H2_Exec_Time(ms)': round(results_H2[idx]['exec_time'] * 1000, 3),
            'H2_Mem(bytes)': results_H2[idx]['mem_usage'],
            'H2_Toxics': results_H2[idx]['toxic_crossed'],
        })

    comparison_df = pd.DataFrame(df_rows)
    print("\n--- Mission Summary & Complexity Metrics ---")
    print(comparison_df.to_string())

    # 5. Iteration Convergence Plotting
    plt.figure(figsize=(12, 5))
    for i, res in enumerate(results_H1):
        h_vals = [log['heuristic_val'] for log in res['search_log']]
        plt.plot(range(len(h_vals)), h_vals, label=f"H1 - Rescue {i+1} to {res['target']}", marker='o', linestyle='dashed')

    for i, res in enumerate(results_H2):
        h_vals = [log['heuristic_val'] for log in res['search_log']]
        plt.plot(range(len(h_vals)), h_vals, label=f"H2 - Rescue {i+1} to {res['target']}", marker='x', linestyle='dotted')

    plt.title('Heuristic Value Convergence vs Iteration Steps (Time)')
    plt.xlabel('Iteration Step')
    plt.ylabel('Heuristic Value h(n)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. Traps Detection
    trap_data = analyze_traps(results_H1, "H1") + analyze_traps(results_H2, "H2")
    if not trap_data:
        print("No dead-end traps observed according to strict heuristic regression.")
    else:
        trap_df = pd.DataFrame(trap_data)
        trap_df = trap_df.drop_duplicates(subset=['Heuristic', 'Trapped_at_Node', 'Target'])
        print("\n--- TRAPS (Local Minima Explored) ---")
        print(trap_df.to_string())

    # 7. A* Comparison
    astar_results = execute_astar_mission(initial_grid)
    comp_data = []
    runs_astar = min(len(results_H1), len(astar_results))
    for i in range(runs_astar):
        comp_data.append({
            'Survivor Sequence': i + 1,
            'GBFS_H1_Explored': results_H1[i]['explored_count'],
            'AStar_H1_Explored': astar_results[i]['explored'],
            'GBFS_H1_Path_Len': len(results_H1[i]['path']) - 1,
            'AStar_H1_Path_Len': astar_results[i]['path_len']
        })
    print("\n--- GBFS vs A* Comparison ---")
    print(pd.DataFrame(comp_data).to_string())


if __name__ == "__main__":
    main()
