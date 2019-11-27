import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import q1
import q2


def readFile(filename):
    """parse the input file

    :filename: TODO
    :returns: TODO

    """
    polygons = []
    start, end = None, None
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('start') or line.startswith('end'):
                tokens = line.strip('\n').replace(' ', '').split(':')[-1][1:-1].split('),(')
                points = []
                for value in tokens:
                    x, y = value.split(',')
                    points.append([float(x), float(y)])
                if(line.startswith('start')):
                    start_points = np.asarray(points)
                    _, start = q1.findAnchorPoint(start_points)
                    start_points = q1.sortPoints(start_points, start)
                else:
                    end_points = np.asarray(points)
                    _, end = q1.findAnchorPoint(end_points)
                    end_points = q1.sortPoints(end_points, end)
                continue
            else:
                polygon=[]
                poly_line = line.strip('\n').replace(' ','').split('),(')
                for idx, item in enumerate(poly_line):
                    x, y = item.replace('(', '').replace(')', '').split(',')
                    polygon.append([float(x), float(y)])

                # TODO: check whether we need to take random points and obtain convexHull polygon for the points
                polygon = np.array(polygon)
                _, anchor_pt = q1.findAnchorPoint(polygon)
                q1.sortPoints(polygon, anchor_pt)
            polygons.append(polygon)

    return start, start_points, end, end_points, polygons

def plotPolygons(polygons, sum_polygons, start_point, end_point, start_config, end_config, graph, id_to_point, path):
    """TODO: Docstring for poltPolygon.

    :polygon: TODO
    :returns: TODO

    """
    # Plot start and end points
    plt.figure()
    plt.scatter(start_point[0], start_point[1], color='b', marker='o')
    plt.text(start_point[0], start_point[1], 'Start', fontsize=15)
    plt.scatter(start_config[:, 0], start_config[:, 1], color='g', marker='.')
    start_config = np.append(start_config, start_config[0, :][None, :], axis=0)
    plt.plot(start_config[:, 0], start_config[:, 1], color='r', linewidth=1)

    start_config = -start_config+start_point
    plt.scatter(start_config[:, 0], start_config[:, 1], color='g', marker='.')
    start_config = np.append(start_config, start_config[0, :][None, :], axis=0)
    plt.plot(start_config[:, 0], start_config[:, 1], color='r', linestyle=':', linewidth=1)

    plt.scatter(end_point[0], end_point[1], color='b', marker='o')
    plt.text(end_point[0], end_point[1], 'End', fontsize=15)
    plt.scatter(end_config[:, 0], end_config[:, 1], color='g', marker='.')
    end_config = np.append(end_config, end_config[0, :][None, :], axis=0)
    plt.plot(end_config[:, 0], end_config[:, 1], color='r', linewidth=1)

    # Plot the polygons
    for polygon in polygons:
        plt.scatter(polygon[:, 0], polygon[:, 1], color='g', marker='.')
        polygon = np.append(polygon, polygon[0, :][None, :], axis=0)
        plt.plot(polygon[:, 0], polygon[:, 1], color='k', linestyle='--', linewidth=1)

    for polygon in sum_polygons:
        plt.scatter(polygon[:, 0], polygon[:, 1], color='g', marker='.')
        polygon = np.append(polygon, polygon[0, :][None, :], axis=0)
        plt.plot(polygon[:, 0], polygon[:, 1], color='r', linewidth=1)

    # plot the visibility graph
    for idx, neighs in graph.items():
        for vertex, dist in neighs:
            x, y = [id_to_point[idx][0], id_to_point[vertex][0]], [id_to_point[idx][1], id_to_point[vertex][1]]
            plt.plot(x, y, color='g', linestyle=':', linewidth=1)

    # plot the path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color='b', linewidth=2)

    plt.show()

def sumPolygonsRobot(start, robot, polygons):
    """Form the new polygons which originates from the configuration space of the robot and polygons

    :robot: TODO
    :polygons: TODO
    :returns: TODO

    """

    robot = -robot+start

    joined_polys = []
    for poly in polygons:
        new_poly = []
        for point in poly:
            for rb_point in robot:
                new_poly.append(point+rb_point)
                new_poly.append(point)
        new_poly = q1.convexHull(np.asarray(new_poly))
        joined_polys.append(np.asarray(new_poly))
    return joined_polys


def main(filename):
    """TODO: Docstring for main.

    :filename: TODO
    :returns: TODO

    """
    start, start_config, end, end_config, polygons = readFile(filename)


    sum_polygons = sumPolygonsRobot(start, start_config, polygons)
    print(sum_polygons)
    graph, idx_to_point = q2.createGraph(start, end, sum_polygons)
    path, min_distance = q2.dijkstra(graph)
    path_points = [idx_to_point[i] for i in path]
    print(path_points)
    print(min_distance)
    plotPolygons(polygons, sum_polygons, start, end, start_config, end_config, graph, idx_to_point, path_points)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need input file: robot_txt")
        exit(1)
    main(sys.argv[1])
