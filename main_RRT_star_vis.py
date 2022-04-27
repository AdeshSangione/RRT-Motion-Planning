import numpy as np
import pygame
import matplotlib.pyplot as plt
from helper_RRT import RRT_map as env
from helper_RRT import RRT_graph
import cv2


pygame.init()


# Change Scale to 1.0 for experiments without scalability requirements
scale = 0.5
# Add obstacles from the environment here
obj_list = [(0.5, 3.5, 2.7, 3.1), (1.8, 5.2, 1.5, 1.3), (5.1, 6.6, 1.2, 6.3), (3.6, 6.6, 1.2, 6.4), (2.0, 6.6, 1.3, 1.3), (0.5, 6.6, 1.3, 2.7)]
# Loop to scale obstacles if needed
obj_list_new = []
for tup in obj_list:
    obj_list_new.append(tuple([i*scale for i in tup]))

obj_list = obj_list_new

# Map dimensions, starting position, and goal postions
MapDim = (6.6*scale, 7.0*scale)

spos = [(0.2*scale, 2.2*scale)]

gpos = [(3.4*scale, 6.3*scale)]

W, H = MapDim

# img = pygame.image.load(r"C:\Users\Adesh\Documents\UofT Courses\AER1516\Final Project\Seattle_alpha.PNG")

m2p = 100 # Conversion to get pixels
screen = pygame.display.set_mode((int(W*m2p), int(H*m2p)))
pygame.display.set_caption("Path Planning algorithm")
# screen.blit(img, img.get_rect(topleft = (0,0)))

path_smooth_list = []
path_valid_list = []

for loc, goal in enumerate(gpos):

    start = spos[loc]
    print(start, goal)
    
    # Create Map object
    path_map = env(MapDim, start, goal, obj_list, m2p, screen)
    path_map.plot_obs()

    # Create Graph object
    RRT_path = RRT_graph(MapDim, start, goal, obj_list)

    # Initialize
    max_iter = 20000
    running = True
    finding_path = True
    i = 0

    # running_2 = True
    # while running_2:

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running_2 = False

    #     # Fill Screen and plot objects
    screen.fill((255, 255, 255))
        # screen.blit(img, img.get_rect(topleft = (0,0)))
    #     # path_map.plot_obs()
    pygame.draw.circle(screen, (255,0,0), (spos[0][0]*m2p, (MapDim[1] - spos[0][1])*m2p),10)
    #     pygame.draw.circle(screen, (255,0,0), (spos[1][0]*m2p, (MapDim[1] - spos[1][1])*m2p),10)
    #     pygame.draw.circle(screen, (255,0,0), (spos[2][0]*m2p, (MapDim[1] - spos[2][1])*m2p), 10)
    pygame.draw.circle(screen, (0,255,0), (goal[0]*m2p, (MapDim[1] - goal[1])*m2p), 10)
    #     pygame.display.update()

    # Main Loop!
    while running:
        path_map.plot_obs()

        # Quit if exit button pressed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Main RRT* loop
        if i < max_iter:

            x,y = RRT_path.rand_sample(prob_bool=True, tol = 0.05)

            # Check if node intersects objects
            if RRT_path.in_region(x, y):
                i += 1
                continue

            # Create node instance
            new_node = RRT_path.Node(x,y)
            
            # Find nearest neighbors!
            nearest, idx_nearest = RRT_path.nearest_neighbors(new_node)
            node_best_parent, best_idx, path_x, path_y = RRT_path.least_cost_parent(new_node, nearest, idx_nearest)

            if node_best_parent == None:
                i+= 1
                continue

            # If node is pretty much the nearest node, move on
            if new_node.is_state_identical(node_best_parent):
                i += 1
                continue

            # Create an interpolated path between best parent and new node
            # path_x, path_y = RRT_path.steer(node_best_parent, new_node)

            # Check if any part of the path intersects an obstacle
            # if RRT_path.in_region_path(path_x, path_y):
            #     i+= 1
            #     continue
            
            # Populate new_node's attributes (path_x, path_y, parent, cost to reach)
            new_node = RRT_path.propagate(best_idx, node_best_parent, new_node, path_x, path_y)

            # # Plot node in pygame
            # path_map.plot_pt((new_node.x, new_node.y))

            if RRT_path.near_goal(new_node):
                i = max_iter
                running = False
            else:
                RRT_path.rewire_graph(new_node, nearest, idx_nearest)
                i += 1
            # else:
            #     i += 1
            #     continue

            # Plot node in pygame
            path_map.plot_pt((new_node.x, new_node.y))

            ##### Time to Rewire Graph #####
            # if i < max_iter:
            #     RRT_path.rewire_graph(new_node, nearest, idx_nearest)

            # Plot branches
            pth = list(zip(new_node.path_x, new_node.path_y))
            path_map.plot_path(pth,2)

            # Update pygame figure and wait abit with new information!
            pygame.display.update()
            pygame.time.wait(10)

            # continue

            # i += 1

            # If node is within goal region, terminate branching
            # if RRT_path.near_goal(new_node):
            #     i = max_iter
            # else:
            #     i += 1
            #     continue

        # Find the valid path!
        # if finding_path:

    curr_node = RRT_path.node_list[-1]
    Total_cost = curr_node.cost
    valid_path, node_list = RRT_path.find_valid_path(curr_node)
    # finding_path = False
    # running = False

    path_smooth, Total_cost = RRT_path.bspline(valid_path, degree = 5)
    path_smooth_list.append(path_smooth)
    path_valid_list.append(valid_path)

    for j, pth in enumerate(path_smooth_list):
        path_map.plot_path(pth, width= 5, branch=0)
        path_map.plot_path(path_valid_list[j],width=3, branch=2)

    # path_map.plot_robot(path_smooth, 1)
    print(Total_cost)
    pygame.display.update()
    pygame.time.wait(5000)

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Fill Screen and plot objects
        # screen.fill((255, 255, 255))
    screen.blit(img, img.get_rect(topleft = (0,0)))
    # path_map.plot_obs()
    pygame.draw.circle(screen, (255,0,0), (spos[0][0]*m2p, (MapDim[1] - spos[0][1])*m2p),10)
    pygame.draw.circle(screen, (255,0,0), (spos[1][0]*m2p, (MapDim[1] - spos[1][1])*m2p),10)
    pygame.draw.circle(screen, (255,0,0), (spos[2][0]*m2p, (MapDim[1] - spos[2][1])*m2p), 10)
    # pygame.draw.circle(screen, (0,255,0), (goal[0]*m2p, (MapDim[1] - goal[1])*m2p), 10)

    for j, pth in enumerate(path_smooth_list):
        path_map.plot_path(pth, width= 5, branch=0)
        path_map.plot_path(path_valid_list[j],width=3, branch=2)

    pygame.display.update()

        
    




