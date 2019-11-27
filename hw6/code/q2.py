import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict


def polarAngle(anchor_pt, point):
    """Find the polar angle between anchor and point
    :returns: angle value


    """
    y_len = point[1] - anchor_pt[1]
    x_len = point[0] - anchor_pt[0]
    return np.arctan2(y_len, x_len)

def plotPolygons(polygons, start_point, end_point, graph, id_to_point, path):
    """TODO: Docstring for poltPolygon.

    :polygon: TODO
    :returns: TODO

    """
    # Plot start and end points
    plt.figure()
    plt.scatter(start_point[0], start_point[1], color='b', marker='o')
    plt.text(start_point[0], start_point[1], 'Start', fontsize=15)
    plt.scatter(end_point[0], end_point[1], color='b', marker='o')
    plt.text(end_point[0], end_point[1], 'End', fontsize=15)
    # Plot the polygons
    for polygon in polygons:
        plt.scatter(polygon[:, 0], polygon[:, 1], color='g', marker='.')
        polygon = np.append(polygon, polygon[0, :][None, :], axis=0)
        plt.plot(polygon[:, 0], polygon[:, 1], color='r', linewidth=2)

    # plot the visibility graph
    for idx, neighs in graph.items():
        for vertex, dist in neighs:
            x, y = [id_to_point[idx][0], id_to_point[vertex][0]], [id_to_point[idx][1], id_to_point[vertex][1]]
            plt.plot(x, y, color='g', linestyle=':', linewidth=1)

    # plot the path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color='b', linewidth=2)

    plt.show()


def onSegment(point1, point2, point3):
    max_x = max(point2[0], point1[0])
    min_x = min(point2[0], point1[0])
    max_y = max(point2[1], point1[1])
    min_y = min(point2[1], point1[1])

    # don't want to consider vertices touching as intersected
    dist1 = np.square(np.sum(point3-point1))
    dist2 = np.square(np.sum(point3-point2))
    if dist1 < 1e-5 or dist2 < 1e-5:
        return False
    if (point3[0] <= max_x) and (point3[0] >= min_x) and (point3[1] <= max_y) and (point3[1] >= min_y):
           return True

    return False


def crossProduct(point1, point2, point3):
    """find crossProduct of 2 vectors in 2D space, magnitude tells direction

    :point1: TODO
    :point2: TODO
    :point3: TODO
    :returns: boolean

    """
    val = (point2[0]-point1[0])*(point3[1]-point1[1])-(point2[1]-point1[1])*(point3[0]-point1[0])
    if val > 0: return 1
    elif val < 0: return 2
    else: return 0

# Reference: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def lineIntersect(point1, point2, point3, point4):
    # don't want to consider vertices touching as intersected
    tolerance = 1e-5
    dist1 = np.sum(np.square(point3-point1))
    dist2 = np.sum(np.square(point3-point2))
    dist3 = np.sum(np.square(point4-point1))
    dist4 = np.sum(np.square(point4-point2))

    if (dist1 < tolerance) or (dist2 < tolerance) or (dist3 < tolerance) or (dist4 < tolerance):
        return False

    o1 = crossProduct(point1, point2, point3)
    o2 = crossProduct(point1, point2, point4)
    o3 = crossProduct(point3, point4, point1)
    o4 = crossProduct(point3, point4, point2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and onSegment(point1, point2, point3): return True
    if o2 == 0 and onSegment(point1, point2, point4): return True
    if o3 == 0 and onSegment(point3, point4, point1): return True
    if o4 == 0 and onSegment(point3, point4, point2): return True

    return False

def isPointInPolygons(point, polygons):
    extreme_point = [np.inf, point[1]]

    for polygon in polygons:
        count = 0
        n = polygon.shape[0]
        for i in range(polygon.shape[0]):
            vertex = polygon[i, :]
            next_vertex = polygon[(i+1)%n, :]
            if lineIntersect(vertex, next_vertex, point, extreme_point):
                # If point ray and edge of polygon are collinear, then check whether it is on the edge
                if crossProduct(vertex, next_vertex, point) == 0:
                    return onSegment(vertex, next_vertex, point)
                count=count+1
        if (count%2 == 1):
            return True

    return False

def dijkstra(graph):
    """Find the shortest path from start point to end point

    :graph: TODO
    :returns: list of indices, min_distances

    """
    dist = [np.inf for node_idx in range(len(graph))]
    cur_node = [0]
    dist[0] = 0
    while cur_node:
        next_node = []
        for item in cur_node:
            for neigh, d in graph[item]:
                neigh_dist = dist[item] + d
                if neigh_dist < dist[neigh]:
                    dist[neigh] = neigh_dist
                    next_node.append(neigh)
        cur_node = next_node

    path_length = dist[-1]
    # Add last point as destination
    path = [len(dist)-1]
    dest = len(dist)-1
    # while we don't reach the start point
    while path[-1] != 0:
        for neigh, d in graph[dest]:
            # distance from start to neighbor < distance[dest] - d
            if abs(dist[dest] - d - dist[neigh]) < 1e-3:
                path.append(neigh)
                dest = neigh
                path_length -= d
                break

    return path, dist[-1]

def createGraph(start_point, end_point, polygons):
    """Create a graph (adjacency list or dict) from given vertices

    :start_point:
    :end_point:
    :polygons:
    :returns: TODO

    """
    points = []
    lines = []
    polygon_vertices = []
    graph = defaultdict(list)

    # start point vertex_id = 0
    vertex_id = 1
    # Add connections in graph corresponding to the polygon edges
    no_polygons = len(polygons)
    for i in range(no_polygons):
        poly = polygons.pop(i)
        # compute points of polygon not inside other polygons
        current_points = [vertex for vertex in poly if not isPointInPolygons(vertex, polygons)]
        points.extend(current_points)
        # compute lines
        lines.extend([[poly[i], poly[i+1]] for i in range(poly.shape[0] - 1)])
        lines.append([poly[-1], poly[0]])

        polygon_set = set()
        no_vertices = poly.shape[0]
        for i in range(no_vertices):
            curr_vertex, next_vertex = poly[i], poly[(i+1)%no_vertices]

            if not isPointInPolygons(curr_vertex, polygons):
                polygon_set.add(vertex_id)
                if not isPointInPolygons(next_vertex, polygons):
                    dist = np.sqrt(np.sum(np.square(curr_vertex - next_vertex)))
                    if i != 0 and (i+1) % no_vertices == 0:
                        graph[vertex_id].append((vertex_id-len(current_points)+1, dist))
                        graph[vertex_id-len(current_points)+1].append((vertex_id, dist))
                    else:
                        graph[vertex_id].append((vertex_id+1, dist))
                        graph[vertex_id+1].append((vertex_id, dist))
                # increment the vertex id for each vertex
                vertex_id += 1
        # reinsert the polygon that was popped out
        polygons.insert(0, poly)
        polygon_vertices.append(polygon_set)

    # store mapping of vertex id (graph keys) to points
    vertex_id_to_points = {i+1: points[i] for i in range(len(points))}
    vertex_id_to_points[0] = start_point
    vertex_id_to_points[len(points)+1] = end_point

    # does not contain start and end points
    vertex_id_list = np.arange(1, vertex_id)
    # Create visibility graph by adding vertices which do not have line intersections
    for i in vertex_id_list:
        for j in vertex_id_list[i:]:
            # check whether two vertices are of same polygon
            same_polygon = False
            for vertices in polygon_vertices:
                if i in vertices and j in vertices:
                    same_polygon = True
                    break
            if same_polygon:
                continue
            # naive check for midpoint of line to be inside polygon
            c = (vertex_id_to_points[i]+vertex_id_to_points[j]) / 2
            if isPointInPolygons(c, polygons):
                continue
            # check whether the points intersect any existing polygon edges
            intersect = False
            for l in lines:
                if lineIntersect(vertex_id_to_points[i], vertex_id_to_points[j], l[0], l[1]):
                    intersect = True
                    break
            if not intersect:
                dist = np.sqrt(np.sum(np.square(vertex_id_to_points[i] - vertex_id_to_points[j])))
                graph[i].append((j, dist))
                graph[j].append((i, dist))

    # for start and end points
    for idx, point in enumerate([start_point, end_point]):
        for j in vertex_id_list:
            intersect = False
            for l in lines:
                if lineIntersect(point, vertex_id_to_points[j], l[0], l[1]):
                    intersect = True
                    break
            if not intersect:
                dist = np.sqrt(np.sum(np.square(point - vertex_id_to_points[j])))
                ind = 0 if idx == 0 else len(points)+1
                graph[ind].append((j, dist))
                graph[j].append((ind, dist))

    return graph, vertex_id_to_points

def readFile(filename):
    """parse and read file

    :filename: TODO
    :returns: TODO

    """
    polygons = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('start'):
                key, value = line.strip('\n').replace(' ', '').split(':')
                x, y = value.replace('(', '').replace(')', '').split(',')
                start_point = np.array([float(x), float(y)])
            elif line.startswith('end'):
                key, value = line.strip('\n').replace(' ', '').split(':')
                x, y = value.replace('(', '').replace(')', '').split(',')
                end_point = np.array([float(x), float(y)])
            else:
                polygon=[]
                poly_line = line.strip('\n').replace(' ','').split('),(')
                for idx, item in enumerate(poly_line):
                    x, y = item.replace('(', '').replace(')', '').split(',')
                    polygon.append([float(x), float(y)])

                min_idx = None
                for idx, (x, y) in enumerate(polygon):
                    if min_idx == None or y < polygon[min_idx][1]:
                        min_idx = idx
                    if y == polygon[min_idx][1] and x < polygon[min_idx][0]:
                        min_idx = idx
                anchor_pt = polygon[min_idx]
                del polygon[min_idx]
                polygon.sort(key = lambda p: polarAngle(anchor_pt, p))
                polygon.insert(0, anchor_pt)
                polygons.append(np.asarray(polygon))
    return start_point, end_point, polygons


def main(filename):
    """Main function
    :returns: TODO

    """
    # Read file and create polygons
    start_point, end_point, polygons = readFile(filename)

    if isPointInPolygons(start_point, polygons) and isPointInPolygons(end_point, polygons):
        print("No path between given start and end points")
        return

    graph, idx_to_point = createGraph(start_point, end_point, polygons)
    path, min_distance = dijkstra(graph)
    path_points = [idx_to_point[i] for i in path]
    print("Path from end to start: ")
    print(path_points)
    print(min_distance)
    plotPolygons(polygons, start_point, end_point, graph, idx_to_point, path_points)



if __name__ == "__main__":
    filename = 'polygon.txt'
    main(filename)
