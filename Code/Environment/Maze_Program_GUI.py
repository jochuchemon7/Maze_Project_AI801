import tkinter as tk
import sys

window = tk.Tk()

window.rowconfigure(0, minsize=400, weight=1)
window.columnconfigure([0, 1, 2, 3, 4], minsize=200, weight=1)

def run_dfs():
    import DFS_Maze
    DFS_Maze
    sys.modules.pop('DFS_Maze')

def run_bfs():
    import BFS_Maze
    BFS_Maze
    sys.modules.pop('BFS_Maze')
    
def run_astar():
    import ASTAR_Maze
    ASTAR_Maze
    sys.modules.pop('ASTAR_Maze')

def run_dfs_heur():
    import DFS_with_Heuristic
    DFS_with_Heuristic
    sys.modules.pop('DFS_with_Heuristic')
        
def run_rl():
    import RL_Maze
    RL_Maze
    sys.modules.pop('RL_Maze')

btn_dfs = tk.Button(master=window, text="Run Depth-First Search", command=run_dfs)
btn_dfs.grid(row=0, column=0, sticky="nsew")

btn_bfs = tk.Button(master=window, text="Run Breadth-First Search", command=run_bfs)
btn_bfs.grid(row=0, column=1, sticky="nsew")

btn_dfs = tk.Button(master=window, text="Run A* Search", command=run_astar)
btn_dfs.grid(row=0, column=2, sticky="nsew")

btn_bfs = tk.Button(master=window, text="Run Depth-First Search with Heuristic", command=run_dfs_heur)
btn_bfs.grid(row=0, column=3, sticky="nsew")

btn_dfs = tk.Button(master=window, text="Run Reinforcement Learning", command=run_rl)
btn_dfs.grid(row=0, column=4, sticky="nsew")

# Run the event loop
window.mainloop()