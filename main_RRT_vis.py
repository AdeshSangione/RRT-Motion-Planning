# Imports
import numpy as np
import pygame
import matplotlib.pyplot as plt

# Once name for classes figured out, use the below lines to import them
from helper_RRT import RRT_map as env
from helper_RRT import RRT_graph

# Initialize pygame to be able to have a continuous screen
pygame.init()

# Create the screen
MapDim = (10,10) # metres
W, H = MapDim
m2p = 100 # Conversion to get pixels
screen = pygame.display.set_mode((int(W*m2p), int(H*m2p)))

# Title
pygame.display.set_caption("Path Planning algorithm")

# Initialize Vars
obj_list = [(2,4,1,2), (4,4,1,1), (5,8,1,1)]
s_pos = (7,1)
g_pos = (1,9)

# Create Map object
path_map = env(MapDim, s_pos, g_pos, obj_list, m2p, screen)

# Create Graph object
RRT_path = RRT_graph(MapDim, s_pos, g_pos, obj_list)

# Max iterations for RRT*
max_iter = 10000

# Game loop
running = True
finding_path = True

# Iter
i = 0

# Fill Screen
screen.fill((255, 255, 255))

# Plot objects, goal, and start
path_map.plot_obs()
pygame.draw.circle(screen, (255,0,0), (s_pos[0]*m2p, (MapDim[1] - s_pos[1])*m2p), 20)
pygame.draw.circle(screen, (0,255,0), (g_pos[0]*m2p, (MapDim[1] - g_pos[1])*m2p), 20)

# Main Loop!
while running:

    # Quit if exit button pressed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Main RRT* loop
    if i < max_iter:
        
        # Randomly sample a node
        x,y = RRT_path.rand_sample(prob_bool=False, tol=0.05)

        # Check if node intersects objects
        if RRT_path.in_region(x, y):
            i += 1
            continue

        # Create node instance
        new_node = RRT_path.Node(x,y)
        
        # Find nearest node!
        near_node, near_idx = RRT_path.nearest(new_node)

        # If node is pretty much the nearest node, move on
        if new_node.is_state_identical(near_node):
            i += 1
            continue

        # Create an interpolated path between nearest and new nodes
        path_x, path_y = RRT_path.steer(near_node, new_node)

        # Check if any part of the path intersects an obstacle
        if RRT_path.in_region_path(path_x, path_y):
            i+= 1
            continue
        
        # Populate new_node's attributes (path_x, path_y, parent, cost to reach)
        new_node = RRT_path.propagate(near_idx,near_node, new_node, path_x, path_y)
        
        # Append new node to valid node list
        # RRT_path.node_list.append(new_node)

        # # Make new_node on of the children of near_node
        # RRT_path.node_list[near_idx].children.append(new_node)

        # Plot node in pygame
        path_map.plot_pt((new_node.x, new_node.y))

        # Plot branches
        pth = list(zip(new_node.path_x, new_node.path_y))
        path_map.plot_path(pth)

        # Update pygame figure with new information!
        pygame.display.update()

        # Wait 100 ms before updating screen!
        pygame.time.wait(100)

        # If node is within goal region, terminate branching
        if RRT_path.near_goal(new_node):
            i = max_iter
        else:
            i += 1
            continue

    # Find the valid path!
    if finding_path:

        valid_path = []
        curr_node = RRT_path.node_list[-1]
        Total_cost = curr_node.cost
        RRT_path.find_valid_path(curr_node, valid_path)
        finding_path = False
        print(Total_cost)

    path_map.plot_path(list(valid_path), branch=False)
    pygame.display.update()
    




