Requirements:

PyTorch is required to run the Reinforcement Learning script RL_Maze.py


Files:

MazeEnv.py - Creates the maze environment for the following search algorithm scripts:
- DFS_Maze.py
- BFS_Maze.py
- ASTAR_Maze.py
- DFS_with_Heuristic.py


MazeEnvRL.py - Creates the maze environment for the following Reinforcement Learning script:
- MazeEnvRL.py


DFS_Maze.py - Implements Depth-First Search algorithm to solve maze and displays animation in figure; default maze dimension is 30 x 30 (update env.new_maze(30, 30) to alter maze dimensions)


BFS_Maze.py - Implements Breadth-First Search algorithm to solve maze and displays animation in figure; default maze dimension is 30 x 30 (update env.new_maze(30, 30) to alter maze dimensions)


ASTAR_Maze.py - Implements A* Search algorithm to with Manhattan distance heuristic to solve maze and displays animation in figure; default maze dimension is 30 x 30 (update env.new_maze(30, 30) to alter maze dimensions)


DFS_with_Heuristic.py - Implements Depth-First Search algorithm with Manhattan distance heuristic to solve maze and displays animation in figure; default maze dimension is 30 x 30 (update env.new_maze(30, 30) to alter maze dimensions)


RL_Maze.py - Implements Reinforcement Learning algorithm (Deep Q-Learning Network model) to solve maze and displays animation in figure; default maze dimension is 9 x 9 (update env.new_maze(nrow=9, ncol=9) to alter maze dimensions - note: only works to approx 11 x 11 size)


Maze_Program_GUI.py - Optional GUI to run each of the five maze-solving scripts (4 search algorithms, 1 Reinforcement Learning; listed above)


To run:

Option 1: Run Maze_Program_GUI.py from the Python shell and make selections from the GUI


Option 2: Run individual algorithm and RL scripts via direct call in the Python shell