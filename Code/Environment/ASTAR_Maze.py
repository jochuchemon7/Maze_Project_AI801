import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MazeEnv import CreateMaze
import numpy as np
import heapq
import math

# - MAZE CREATION -
env = CreateMaze()
nrow, ncol = np.random.randint(20, 35),  np.random.randint(20, 35)
env.new_maze(30, 30)
env.make()
env.render()

# - VISITED DICTIONARY CREATION -
wall = 'w'
visited_astr = dict()
for i in range(env.get_maze().shape[0]):
    for j in range(env.get_maze().shape[1]):
        visited_astr[(i, j)] = False

# - ASTAR ALGORITHM -
astar_queue = []
heapq.heapify(astar_queue)

def astar(maze, visited_astr, node, target_node, order, path_cost):
    if node[0] == target_node[0] and node[1] == target_node[1]:
        return True
    else:
        visited_astr[(node[0], node[1])] = True
        order.append([node[0], node[1]])

    new_node = node
    
    #Identify next best node to visit from current
    node_h_cost = (float(math.inf),float(math.inf))
    for adjacent_x, adjacent_y in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # Top, right, bottom and left neighbors
        frontier_node = node[0] + adjacent_x, node[1] + adjacent_y

        #Increment path cost to follow-on nodes (+1 for each node opening - do only once)
        if adjacent_x == -1 and adjacent_y == 0:
            path_cost += 1
            
        #Manhattan distance heuristic - distance in nodes to target
        h = abs(target_node[0]-frontier_node[0])+abs(target_node[1]-frontier_node[1])
        #Uncomment the following to see behavior with no heuristic (becomes BFS)
        #h = 0
        frontier_node_cost = path_cost + h
        frontier_node_h_cost = (frontier_node_cost,h)

        #Push frontier node location and costs to priority queue data structure - lowest cost node first       
        if maze[frontier_node[0], frontier_node[1]] != wall and not visited_astr[(frontier_node[0], frontier_node[1])]:
            heapq.heappush(astar_queue, ((frontier_node_cost,h),[frontier_node[0],frontier_node[1]],path_cost))
        
        if frontier_node_h_cost <= node_h_cost and maze[frontier_node[0], frontier_node[1]] != wall and not visited_astr[(frontier_node[0], frontier_node[1])]:
            node_h_cost = frontier_node_h_cost
            new_node = frontier_node

    #Pop node from priority queue if it is visited - only want to visit unexplored nodes on the frontier if necessary
    for data in astar_queue:
        if visited_astr[data[1][0],data[1][1]]:
            astar_queue.remove(data)
            heapq.heapify(astar_queue)
    
    #Get lowest cost (test) node from queue - move back to the test node if cost is lower than current
    test_node = astar_queue[0]
    test_node_cost = test_node[0]
    test_node_coord = test_node[1]

    if test_node_cost < node_h_cost or node_h_cost == (float(math.inf),float(math.inf)):
        new_node = test_node_coord
        #Reset path cost to test node cost
        path_cost = test_node[2]
        
    if 0 <= new_node[0] < len(maze) and 0 <= new_node[1] < len(maze[0]) and not visited_astr[(new_node[0], new_node[1])] and maze[new_node[0], new_node[1]] != wall:
        #Move on to next node - no dead end
        if astar(maze, visited_astr, new_node, target_node, order, path_cost):
            return True, visited_astr, order
        #Move on to next node - back track from dead end
        elif astar(maze, visited_astr, astar_queue[0][1], target_node, order, astar_queue[0][2]):
            return True, visited_astr, order
        
    return False


# - ASTAR CALL -
start_node = env.get_start_node()
target_node = env.get_target_node()
order = []
demo_found, demo_result, new_order = astar(env.maze, visited_astr, start_node, target_node, order, 0)


# - ANIMATION VIEW -
if demo_found:
    demo_solution = env.get_maze()
    demo_solution[env.get_target_node()[0], env.get_target_node()[1]] = 2
    fig = plt.figure('A* Search')
    img = []
    for cell in new_order:
        if demo_result[(cell[0], cell[1])]:
            demo_solution[cell[0], cell[1]] = 3
            img.append([plt.imshow(demo_solution)])

    ani = animation.ArtistAnimation(fig, img, interval=20, blit=True, repeat_delay=0)
    plt.show()

plt.imshow(demo_solution)