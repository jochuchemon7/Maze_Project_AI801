import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Randomized Prim's Algorithm https://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_Kruskal's_algorithm

class CreateMaze:
    def __init__(self, row=20, col=20):
        self.nrow, self.ncol = row, col
        self.target_node = [0, 0]
        self.start_node = [0, 0]
        self.new_maze(self.nrow, self.ncol)

    def new_maze(self, nrow, ncol):
        self.nrow, self.ncol = nrow, ncol
        self.maze = np.array(['' for _ in range(self.nrow * self.ncol)], dtype='str').reshape(self.nrow, self.ncol)
        self.wall = 'w'  # Characters for ease when dealing with empty cells
        self.passage = 'p'
        self.empty = ''


        # PART II (Pick a cell and mark it as part of the maze)
        start_x = np.random.randint(1, self.ncol)  # Not 0 bc of edge case
        start_y = np.random.randint(1, self.nrow)

        # Edge case
        if start_y == (self.nrow - 1):
            start_y -= 1
        if start_x == (self.ncol - 1):
            start_x -= 1

        self.maze[start_y, start_x] = self.passage
        # PART II (Add the walls of the cell to the wall list and wall to the maze-grid)

        self.walls = list()
        self.walls.append([start_y - 1, start_x])  # Append top block
        self.walls.append([start_y, start_x + 1])  # Append right block
        self.walls.append([start_y + 1, start_x])  # Append bottom block
        self.walls.append([start_y, start_x - 1])  # Append left block

        self.maze[self.walls[0][0], self.walls[0][1]] = self.wall
        self.maze[self.walls[1][0], self.walls[1][1]] = self.wall
        self.maze[self.walls[2][0], self.walls[2][1]] = self.wall
        self.maze[self.walls[3][0], self.walls[3][1]] = self.wall

    # PART III (While there are walls in the walls list)
    # PART III - I (Pick a random wall from the list, if only one of the cells that the wall divides is visited then):
    # PART III - I - I (Make the wall a passage and mark the unvisited cell as part of the maze)
    # PART III - I - II (Add the neighboring walls of the cell to the wall list)
    # PART III - II (Remove the wall from the list)

    def surrounding_passages(self, rand_wall):
        npassages = 0
        if self.maze[rand_wall[0]-1, rand_wall[1]] == self.passage:  # If top cell is a passage
            npassages += 1
        if self.maze[rand_wall[0], rand_wall[1]+1] == self.passage:  # If right cell is a passage
            npassages += 1
        if self.maze[rand_wall[0]+1, rand_wall[1]] == self.passage:  # If bottom cell is a passage
            npassages += 1
        if self.maze[rand_wall[0], rand_wall[1]-1] == self.passage:  # If left cell is a passage
            npassages += 1
        return npassages

    def remove_wall(self, rand_wall):
        for wall in self.walls:
            if wall[0] == rand_wall[0] and wall[1] == rand_wall[1]:
                self.walls.remove(wall)

    def make_walls(self, nrow, ncol):  # Make unvisited blocks into walls.
        for i in range(0, nrow):
            for j in range(0, ncol):
                if self.maze[i][j] == self.empty:
                    self.maze[i][j] = self.wall

    def create_entrance_exit(self, nrow, ncol):
        for i in range(ncol-1, 0, -1):
            if self.maze[nrow-2][i] == self.passage:
                self.maze[nrow-1][i] = self.passage
                self.target_node = [nrow-1, i]
                break

        for j in range(0, ncol):
            if self.maze[1][j] == self.passage:
                self.maze[0][j] = self.passage
                self.start_node = [0, j]
                break


    def print_maze(self, maze):
        print(maze)

    def get_maze_(self):
        return self.maze

    def get_start_node(self):
        return self.start_node

    def get_target_node(self):
        return self.target_node

    def get_maze(self):
        final_maze = pd.DataFrame(self.maze)
        final_maze[final_maze == self.passage] = 1
        final_maze[final_maze == self.wall] = 0
        final_maze = np.array(final_maze, dtype=np.int32)
        return final_maze

    def upper_wall(self, rand_wall):
        if rand_wall[0] != 0:  # If wall is not on top border  (Upper Cell)
            if self.maze[rand_wall[0] - 1, rand_wall[1]] != self.passage:  # If top side is a cell
                self.maze[rand_wall[0] - 1, rand_wall[1]] = self.wall  # Change top side to a wall
            if [rand_wall[0] - 1, rand_wall[1]] not in self.walls:  # If top side is not in walls list
                self.walls.append([rand_wall[0] - 1, rand_wall[1]])  # Add top side to the wall list

    def right_wall(self, rand_wall):
        if rand_wall[1] != self.ncol - 1:  # If wall not on the right border
            if self.maze[rand_wall[0], rand_wall[1] + 1] != self.passage:
                self.maze[rand_wall[0], rand_wall[1] + 1] = self.wall
            if [rand_wall[0], rand_wall[1] + 1] not in self.walls:
                self.walls.append([rand_wall[0], rand_wall[1] + 1])

    def bottom_wall(self, rand_wall):
        if rand_wall[0] != self.nrow - 1:  # If wall is not bottom border
            if self.maze[rand_wall[0] + 1, rand_wall[1]] != self.passage:
                self.maze[rand_wall[0] + 1, rand_wall[1]] = self.wall
            if [rand_wall[0] + 1, rand_wall[1]] not in self.walls:
                self.walls.append([rand_wall[0] + 1, rand_wall[1]])

    def left_wall(self, rand_wall):
        if rand_wall[1] != 0:  # If wall is not left border
            if self.maze[rand_wall[0], rand_wall[1] - 1] != self.passage:
                self.maze[rand_wall[0], rand_wall[1] - 1] = self.wall
            if [rand_wall[0], rand_wall[1] - 1] not in self.walls:
                self.walls.append([rand_wall[0], rand_wall[1] - 1])

    def create_maze(self):
        while self.walls:
            rand_wall = self.walls[np.random.randint(0, len(self.walls))]  # Random pick
            # self.print_maze(self.maze)

            if rand_wall[1] != 0:  # Check if not on the left border
                if self.maze[rand_wall[0], rand_wall[1] - 1] == self.empty and self.maze[rand_wall[0], rand_wall[1] + 1] == self.passage:  # Divides left side empty with right side cell  (If left side is empty and right side cell)
                    npassages = self.surrounding_passages(rand_wall)
                    if npassages <= 1:  # If one or fewer passages are present on
                        self.maze[rand_wall[0], rand_wall[1]] = self.passage  # Change current random wall to a passage

                        self.upper_wall(rand_wall)
                        self.bottom_wall(rand_wall)
                        self.left_wall(rand_wall)

                    self.remove_wall(rand_wall)  # PART III (Remove wall from the list)
                    continue

            if rand_wall[0] != 0:  # Check if not on the top border
                if self.maze[rand_wall[0]-1, rand_wall[1]] == self.empty and self.maze[rand_wall[0]+1, rand_wall[1]] == self.passage:  # If above the wall is empty and below is a cell
                    npassages = self.surrounding_passages(rand_wall)
                    if npassages <= 1:
                        self.maze[rand_wall[0], rand_wall[1]] = self.passage

                        self.upper_wall(rand_wall)
                        self.right_wall(rand_wall)
                        self.left_wall(rand_wall)

                    self.remove_wall(rand_wall)
                    continue

            if rand_wall[0] != self.nrow-1:  # Check if wall is not on bottom border
                if self.maze[rand_wall[0]+1, rand_wall[1]] == self.empty and self.maze[rand_wall[0]-1, rand_wall[1]] == self.passage:  # If below wall is empty and above the wall is a cell
                    npassages = self.surrounding_passages(rand_wall)
                    if npassages < 2:
                        self.maze[rand_wall[0], rand_wall[1]] = self.passage

                        self.right_wall(rand_wall)
                        self.bottom_wall(rand_wall)
                        self.left_wall(rand_wall)

                    self.remove_wall(rand_wall)
                    continue

            if rand_wall[1] != self.ncol-1:  # Check if wall is not on right border
                if self.maze[rand_wall[0], rand_wall[1]+1] == self.empty and self.maze[rand_wall[0], rand_wall[1]-1] == self.passage:  # If right to the wall is empty and left to the wall is a cell
                    npassages = self.surrounding_passages(rand_wall)
                    if npassages < 2:

                        self.maze[rand_wall[0], rand_wall[1]] = self.passage

                        self.upper_wall(rand_wall)
                        self.right_wall(rand_wall)
                        self.bottom_wall(rand_wall)

                    self.remove_wall(rand_wall)
                    continue

            self.remove_wall(rand_wall)

    def make(self):
        self.create_maze()
        self.make_walls(self.maze.shape[0], self.maze.shape[1])
        self.create_entrance_exit(self.maze.shape[0], self.maze.shape[1])
        self.print_maze(self.maze)

    def render(self):
        new_maze = np.array(self.maze)
        df = pd.DataFrame(new_maze)
        df[df == self.wall] = 0
        df[df == self.passage] = 1
        new_maze = np.array(df, dtype='int')
        plt.imshow(new_maze)


