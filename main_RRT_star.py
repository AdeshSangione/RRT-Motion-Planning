import numpy as np
import pygame
import matplotlib.pyplot as plt
from helper_RRT import RRT_graph
import time

def main(iters, MapDim, spos, gpos, obj_list):
    """
    iters: How many iterations want to be conducted
    MapDim: Dimensions of the environment
    s_pos: Starting Position for first trajectory
    g_pos: Target locations that must be visited
    obj_list: List of objects
    """

    # Initialize lists to hold execution time and path cost
    Total_time = []
    Path_cost = []

    # To complete numerous random trials
    for itr in range(0, iters):

        # Being timer
        start_time = time.time()  
        # Uncomment below for plotting
      
        # _, ax = plt.subplots()
        Tot_cost = 0

        # Iterate through goal positions
        for loc, goal in enumerate(gpos):

            # Find matching state location
            start = spos[loc]

            # Create Graph object
            RRT_path = RRT_graph(MapDim, start, goal, obj_list)

            # Initialize main loop params
            running = True
            max_iter = 10000
            i = 0
            check = 0

            # Main Loop!
            while running:

                # Main RRT* loop
                if i < max_iter:
                    x,y = RRT_path.rand_sample(prob_bool = True, tol = 0.05)

                    # Check if node intersects objects
                    if RRT_path.in_region(x, y):
                        i += 1
                        continue

                    # Create node instance
                    new_node = RRT_path.Node(x,y)
                    
                    # Find nearest neighbors!
                    nearest, idx_nearest = RRT_path.nearest_neighbors(new_node)
                    node_best_parent, best_idx, path_x, path_y = RRT_path.least_cost_parent(new_node, nearest, idx_nearest)
                    
                    # If there are no nodes available to be a parent, move on
                    if node_best_parent == None:
                        i+= 1
                        continue

                    # If node is pretty much the nearest node, move on
                    if new_node.is_state_identical(node_best_parent):
                        i += 1
                        continue
                   
                    # Populate new_node's attributes (path_x, path_y, parent, cost to reach)
                    new_node = RRT_path.propagate(best_idx, node_best_parent, new_node, path_x, path_y)
                    
                    # If new node is close to goal, terminate loop otherwise continue to next iteration
                    if RRT_path.near_goal(new_node):
                        i = max_iter
                        running = False
                    else:
                        RRT_path.rewire_graph(new_node, nearest, idx_nearest)
                        i += 1

                    # ax.plot(new_node.x, new_node.y, 'bo', markersize = 5)
                    # ax.plot(new_node.path_x, new_node.path_y, 'r-')
                    # plt.pause(0.05)

                # Check to see if loop went over maximum iterations, if so return None for path
                if i == max_iter and running == True:
                    running = False
                    check = 1

            if check == 0:
                curr_node = RRT_path.node_list[-1]
                valid_path, node_list = RRT_path.find_valid_path(curr_node)
                path_smooth, Total_cost = RRT_path.bspline(valid_path, degree = 5)
            else:
                Total_cost = int(10**6)

            # Add total cost if there were multiple goal locations within the same environment
            Tot_cost  += Total_cost
        
        # End timer
        t_comp = time.time() - start_time

        # Append results
        Total_time.append(t_comp)
        Path_cost.append(Tot_cost)
        print(t_comp, Tot_cost)
            
    return Total_time, Path_cost

if __name__== "__main__":

    # Change Scale to 1.0 for experiments without scalability requirements
    scale = 1.5
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

    t_iters, cost_iters = main(30, MapDim, spos, gpos, obj_list)
    np.savetxt("Execution Time.txt", t_iters)
    np.savetxt("Path Costs.txt", cost_iters)
    


