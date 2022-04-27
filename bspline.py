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

    return path, Total_dist