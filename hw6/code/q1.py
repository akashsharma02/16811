import numpy as np
import matplotlib.pyplot as plt

def createPoints(range_min, range_max, no_points):
    """Create 2D integer points in a fixed size

    :min: TODO
    :max: TODO
    :no_points: TODO
    :returns: TODO

    """
    points = []
    points = range_max * np.random.randn(no_points, 2) + range_min
    # for i in range(no_points):
    #     points.append((np.random.randint(range_min, range_max), np.random.randint(range_min, range_max)))
    return np.asarray(points)

def crossProduct(point1, point2, point3):
    """find crossProduct of 2 vectors in 2D space, magnitude tells direction

    :point1: TODO
    :point2: TODO
    :point3: TODO
    :returns: boolean

    """
    return (point2[0]-point1[0])*(point3[1]-point1[1]) \
            -(point2[1]-point1[1])*(point3[0]-point1[0])

def findAnchorPoint(points):
    """TODO: Docstring for findAnchorPoint.

    :points: TODO
    :returns: TODO

    """
    # Find anchor point (smallest y value, if equal then smallest x value)
    points = np.asarray(points)
    min_idx = None
    for i, (x, y) in enumerate(points):
        if min_idx == None or y < points[min_idx, 1]:
            min_idx = i
        if y == points[min_idx, 1] and x < points[min_idx, 0]:
            min_idx = i
    return min_idx, points[min_idx, :]

def sortPoints(points, anchor_point):
    """Sort the give points in anticlockwise fashion about the anchor point

    :points: TODO
    :anchor_point: TODO
    :returns: TODO

    """
    def polarAngle(point):
        """Find the polar angle between anchor and point
        :returns: angle value


        """
        y_len = point[1] - anchor_point[1]
        x_len = point[0] - anchor_point[0]
        return np.arctan2(y_len, x_len)

    point_list = points.tolist()
    point_list.sort(key = lambda p: polarAngle(p))
    return np.asarray(point_list)

def convexHull(points):
    """Find the convex hull for given set of 2d points

    :points: numpy array of 2D points (x, y) pairs
    :returns: a list of points forming the convex hull of the input

    """
    # Handle trivial case and return
    no_points = points.shape[0]
    if no_points <= 3:
        return points

    anchor_idx, anchor_pt = findAnchorPoint(points)
    points = np.delete(points, (anchor_idx), axis=0)

    points = sortPoints(points, anchor_pt)

    hull = [anchor_pt, points[0]]
    for p in points[1:]:
        while crossProduct(hull[-2], hull[-1], p) <= 0:
            del hull[-1]
            if len(hull) < 2:
                break
        hull.append(p)

    return np.asarray(hull)

def main():
    """Main function
    :returns:

    """
    no_points = 10000
    low, high = 0, 100
    # create points randomly
    points = createPoints(low, high, no_points)
    plt.scatter(points[:, 0], points[:, 1], color='b', marker='.')

    convex_hull_points = convexHull(points)

    convex_hull_points = np.append(convex_hull_points, convex_hull_points[0, :][None, :], axis=0)
    plt.plot(convex_hull_points[:, 0], convex_hull_points[:, 1], 'r')
    plt.show()

if __name__ == "__main__":
    main()
