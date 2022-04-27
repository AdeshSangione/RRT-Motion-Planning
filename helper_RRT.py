# Import what is needed
import pygame
import numpy as np
import math
from scipy import interpolate

class RRT_map():
    def __init__(self, map_dim, start, goal, obs_list, m2p, screen):
        #Inputs:
        # map_dim: (W, H) in metres
        # start: (SX, SY) in metres
        # goal: (GX, GY) in metres
        # obs_list: List with tuple items ie: [(x1,y1,w1,h1),...,(xn,yn,w1,h1)]. Position plus geometric info required
        # Pygame screen to draw on - initialized in main

        # To get pixel coordinates from metres, will let us plot in the right coords
        self.m2p = m2p

        # Declare inputs
        self.map_dim = map_dim
        self.W, self.H = self.map_dim
        self.Wp, self.Hp = (int(self.W*self.m2p), int(self.H*self.m2p))
        self.start = start
        self.sx, self.sy = start
        self.goal = goal
        self.gx, self.gy = goal
        self.obs = obs_list
        self.screen = screen
        self.img = pygame.image.load(r"C:\Users\Adesh\Documents\UofT Courses\AER1516\Final Project\robot.png")
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.sx*self.m2p, self.Hp - self.sy*self.m2p))

        # Declare colours for plotting
        self.black = (0,0,0) # For obs
        self.red = (255,0,0) # For start
        self.green = (0, 255, 0) # For goal
        self.blue = (0, 0, 255) # For nodes added
        self.purple = (221,160,221) # For path itself
        self.cyan = (0,255,255)

        # Check to see if start and goal are far enough apart
        self.rad_check = 1


    def plot_obs(self):

        for obj in self.obs:
            # Extract (x,y) coords
            ob_x, ob_y, ob_w, ob_h = tuple(i*self.m2p for i in obj)

            # !!!!! Remember to flip y coord since CSYS starts in top-left corner
            ob_y = self.Hp - ob_y
            rect_tup = pygame.Rect(ob_x, ob_y, ob_w, ob_h)
            
            # Plot rectangle at coordinates, use self.black when plotting rectangle
            pygame.draw.rect(self.screen, self.black, rect_tup)

        return

    def plot_pt(self,point):
        rad_node = 5 #pixels

        # Extract (x,y) coords
        x, y = tuple( i*self.m2p for i in point)
        # !!!!! Remember to flip y coord since CSYS starts in top-left corner
        y = self.Hp - y
        # Plot a small circle where the node is, use self.blue for nodes
        pygame.draw.circle(self.screen, self.blue, (x,y), rad_node)
        
        return


    def plot_path(self,node_list, width, branch = 1):
        # Initialize converted list
        node_list_conv = []

        # To convert points into pixel coordinates
        for tup in node_list:
            x_i, y_i = tuple(i*self.m2p for i in tup)

            # !!!!! Remember to flip y coord since CSYS starts in top-left corner
            y_i = self.Hp - y_i
            node_list_conv.append((x_i, y_i))
        
        # Plot branch or final path (different colours)
        if branch == 1:
            pygame.draw.lines(self.screen, self.purple, 0, node_list_conv, width)
        elif branch == 0:
            pygame.draw.lines(self.screen, self.cyan, 0, node_list_conv, width)
        else:
            pygame.draw.lines(self.screen, (176, 0, 150), 0, node_list_conv, width)

    def plot_robot(self, path, robot_speed):

        # Need to convert positons to pixel coords!
        for idx, tup in enumerate(path[1:]):
            self.screen.fill((255, 255, 255))
            self.plot_obs()
            self.plot_path(path,5,branch = 0)

            x, y = tup
            x_b, y_b = path[idx]

            X = x - x_b
            Y = y - y_b

            norm = np.linalg.norm((X,Y))
            phi = np.rad2deg(math.atan2(Y,X))

            img_new = pygame.transform.rotate(self.rotated, phi)
            pos = img_new.get_rect(center=(x*self.m2p,self.Hp - y*self.m2p))
            self.screen.blit(img_new, pos)
            pygame.time.wait(int(norm/robot_speed*1000))
            pygame.display.update()

        return


class RRT_graph():

    class Node():

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.children = []
            self.cost = 0

        def is_state_identical(self, node):
            """
                check if x, y, yaw are the identical
            """
            if abs(node.x - self.x) > 0.001:
                return False
            elif abs(node.y - self.y) > 0.001:
                return False

            return True

    def __init__(self, map_dim, start, goal, obs_list, max_iter = 10000):
        
        # Declare inputs
        self.map_dim = map_dim
        self.W, self.H = self.map_dim
        self.start = self.Node(start[0], start[1])
        self.sx, self.sy = start
        self.goal = self.Node(goal[0], goal[1])
        self.gx, self.gy = goal
        self.obs = obs_list
        self.step = 0.02
        self.search_rad = 1.5
        self.node_list = [self.start]

    def rand_sample(self, prob_bool, tol = 0.10):
        """
        Sample a random node or assign to parent if bias is desired
        """
        prob = np.random.uniform(0.0, 1.0)

        if prob_bool and prob < tol:
            x = self.gx
            y = self.gy
            return x, y
        else:
            x = np.random.uniform(0, self.W)
            y = np.random.uniform(0, self.H)
            return x, y


    def steer(self, near_node, new_node):
        """Create an interpolate path between nodes"""
        x_n, y_n = (near_node.x, near_node.y)
        x_new, y_new = (new_node.x, new_node.y)
        X = x_new - x_n
        Y = y_new - y_n

        # Get norm
        norm = np.linalg.norm((X, Y))

        # Determine incremental step along norm
        step_iter = np.floor(norm/self.step).astype(np.int)

        if step_iter < 1:
            step_iter = 1

        idx = np.array(list(range(0,step_iter)))
        path_bw_nodes_x = list(x_n + idx/step_iter*X)
        path_bw_nodes_y = list(y_n + idx/step_iter*Y)
        path_bw_nodes_x.append(x_new)
        path_bw_nodes_y.append(y_new)
        
        return path_bw_nodes_x, path_bw_nodes_y


    def in_region(self, x, y):
        """
        Check if node position intersects any obstacles!
        """
        for obj in self.obs:
            x_TL, y_TL, w, h = obj
            
            if (x > x_TL) and (x < x_TL + w):
                if (y < y_TL) and (y > y_TL - h):
                    return True
            else:
                continue

        return False

    def in_region_path(self, node_path_x, node_path_y):
        """Check if path intersects obstacles"""
        for x, y in list(zip(node_path_x, node_path_y)):
            if self.in_region(x,y):
                return True
            else:
                continue

        return False

    def calc_cost(self, nearest_node, new_node):
        """Determine new cost"""
        cost_to_new = np.linalg.norm((new_node.x-nearest_node.x, new_node.y-nearest_node.y))
        cost = nearest_node.cost + cost_to_new
        return cost

    def propagate(self, par_idx, parent, new_node, x_int_path, y_int_path):
        """Populate the new node's and parent's attributes!"""
        new_node.path_x = x_int_path
        new_node.path_y = y_int_path
        new_node.parent = parent
        new_node.cost = self.calc_cost(parent, new_node)
        self.node_list.append(new_node)
        self.node_list[par_idx].children.append(new_node)

        return new_node

    def near_goal(self, new_node):
        """Function to check if node is near goal"""
        goal_reg = 0.1
        if np.linalg.norm((self.gx - new_node.x, self.gy - new_node.y)) < goal_reg:
            return True
        else:
            return False

    def nearest_neighbors(self, new_node):
        """Find the nearest neighbors to new node within a set radius"""
        x_n, y_n = (new_node.x, new_node.y)

        # Find the nearest neighbors
        near_neighbors = []
        idx_neighbors = []
        for idx, nd in enumerate(self.node_list):
            dist = np.linalg.norm((x_n - nd.x, y_n - nd.y))
            if dist < self.search_rad:
                near_neighbors.append(nd)
                idx_neighbors.append(idx)            
            else:
                continue

        # Check if there are any nodes in the vicinity, otherwise make start node default neighbor
        if near_neighbors == []:
            near_neighbors.append(self.start)
            idx_neighbors.append(0)

        return near_neighbors, idx_neighbors

    def least_cost_parent(self, new_node, nearest_nodes, nearest_idx):
        """Find the least cost parent and assign it to new_node"""
        
        # If start node is the only node in the nearest (assigned automatically, if no near nodes)
        if nearest_nodes == [self.start]:
            pathx, pathy = self.steer(self.start, new_node)

            if self.in_region_path(pathx, pathy):
                return None, None, None, None

            return self.start, 0, pathx, pathy
        
        # Check to see what is best neighbor
        cost = np.inf
        node_best_parent = None

        for idx, node in enumerate(nearest_nodes):
            pathx, pathy = self.steer(node, new_node)

            # Collision detection!
            if self.in_region_path(pathx, pathy):
                continue

            # Find minimum cost parent
            cost_curr = self.calc_cost(node, new_node)

            if cost_curr < cost:
                node_best_parent = node
                pathx_best, pathy_best = (pathx, pathy)
                best_idx = nearest_idx[idx]
                cost = cost_curr
        
        # If absolutely nothing near (likely because of collisions, move onto next iteration)
        if node_best_parent == None:
            return None, None, None, None

        return node_best_parent, best_idx, pathx_best, pathy_best

    def rewire_graph(self, new_node, nearby_nodes, indices):
        """Rewires the graph and updates downstream children costs"""
        for idx, node in enumerate(nearby_nodes):

            if node == new_node.parent:
                continue

            node_int = self.Node(node.x, node.y)

            # Calculate cost to new_child
            cost_int = self.calc_cost(new_node, node_int)

            # If cost to node is less than before through new_node
            if cost_int < node.cost:
                # Steer
                xpath, ypath = self.steer(new_node, node_int)

                # Check collision
                if self.in_region_path(xpath, ypath):
                    continue

                # Propgate new values and replace in master list
                node_idx = indices[idx]
                node_int.path_x = xpath
                node_int.path_y = ypath
                node_int.cost = cost_int
                node_int.parent = new_node
                node_int.children = node.children
                self.node_list[node_idx] = node_int

                # Update children costs!
                parent_list = [node_int]
                new_pars = []

                # Iterate down the branches of the rewired node, and recalculate cost
                while True:

                    # Get list of children nodes
                    for j in range(0,len(parent_list)):
                        ch_list = parent_list[j].children

                        # No children yet for this node
                        if ch_list == []:
                            continue

                        # for each child in the child node list, determine its place
                        # in the master list, and then update its cost
                        for ch in ch_list:
                            
                            # Update child cost
                            ch.cost = self.calc_cost(parent_list[j], ch)

                            # Continue down the line of children and update their costs incrementally
                            new_pars.append(ch)
                    
                    # Current children become new parents, continue updating children costs
                    if new_pars == []:
                        break
                    else:
                        parent_list = new_pars
                        new_pars = []

            else:
                continue
            
        return

    def find_valid_path(self, curr_node, find_path = True):
        """Helper function to find final path"""
        node_list = []
        valid_list = []

        while find_path:

            node_list.append(curr_node)
            path_int = list(zip(curr_node.path_x, curr_node.path_y))
            path_int = path_int[::-1]
            
            for tup in path_int:
                valid_list.append(tup)

            if curr_node.parent == self.start:
                node_list.append(curr_node.parent)
                find_path = False
                # print('Done finding path!')
            else:
                curr_node = curr_node.parent
                continue
            
        # Reverse backwards node list to go beginning to end
        return valid_list, node_list[::-1]

    def bspline(self, ctrl_pts, degree = 5):
        """Smooth the output path from path planning algo, so the robot doesn't have harsh paths
        
        Inputs:
        ctrl_pts: List type with nested tuples, [(x1, y1), ..., (x2, y2)]

        Outputs:
        path: List type with nested tuples, [(x1, y1), ..., (x2, y2)]

        """
        # Don't want too many control points, so remove some (remove every 10, I linearly interpolated the path in the steer function for RRT*. In other words, if I had 2 points connected by an edge, I subdivided this edge a bunch to get numerous points connecting the 2 nodes, you might have done this differently! I did this to help create a nice plot of the path and for collision detection.)
        dat_pts = []
        x_b, y_b = (np.inf,np.inf)
        for idx, tup in enumerate(ctrl_pts):
            if idx == len(ctrl_pts)-1 or idx%10 == 0:
                # Get every 10th point and the last point (endpoint condition)
                x_i, y_i = tup
                if x_i == x_b and y_i == y_b:
                    continue
                dat_pts.append(tup)
                x_b, y_b = tup

        # Parametrize the spline with 100 points between 0 and 1
        n = 100
        # Convert to array and get number of data points
        cv = np.array(dat_pts)
        count = cv.shape[0]

        # Prevent degree from exceeding count-1, otherwise splev will crash
        degree = np.clip(degree,1,count-1)

        # Calculate knot vector
        kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')

        # Calculate query range
        u = np.linspace(0,(count-degree),n)

        # Evaluate spline and recombine in a list with nested tuples
        X_s, Y_s = interpolate.splev(u, (kv,cv.T,degree))
        path = list(zip(X_s,Y_s))

        # Calculate Length of spline (total cost of path):
        Total_dist = 0
        for idx, tup in enumerate(path[1:]):
            x_prev, y_prev = path[idx]
            x_now, y_now = tup
            dist = np.linalg.norm((x_now-x_prev, y_now - y_prev))
            Total_dist += dist

        return path[::-1], Total_dist

