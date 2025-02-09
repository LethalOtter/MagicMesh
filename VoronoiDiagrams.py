import numpy as np
import random

# Data structures


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.edge = None


class HalfEdge:
    def __init__(self, vertex, face=None):
        self.origin = vertex
        self.face = face  # incident face
        self.next = None  # halfedge origintating from vetrex being pointed at
        self.prev = None  # halfedge pointing to origin vertex
        self.twin = None  # Shared Half-edge


class Face:
    def __init__(self):
        self.edge = None


class TriangleMesh:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.faces = []
        self.edge_map = {}

    def add_triangle(self, n1, n2, n3):

        v1 = self.get_or_make_vertex(n1)
        v2 = self.get_or_make_vertex(n2)
        v3 = self.get_or_make_vertex(n3)

        cross = 1 / 2 * (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x)  # Ensure CCW direction

        if cross > 0:
            temp = v1
            v1 = v2
            v2 = temp
            

        e1 = HalfEdge(v1)
        e2 = HalfEdge(v2)
        e3 = HalfEdge(v3)


        v1.edge = e1
        v2.edge = e2
        v3.edge = e3

        existing_edges = [
            self.edge_map.get((v1, v2)),
            self.edge_map.get((v2, v3)),
            self.edge_map.get((v3, v1)),
        ]

        if any(existing_edges):
            e1.next = e3
            e2.next = e1
            e3.next = e2

        else:
            e1.next = e2
            e2.next = e3
            e3.next = e1

        self.link_twins(e1)
        self.link_twins(e2)
        self.link_twins(e3)

        face = Face(e1)
        e1.face, e2.face, e3.face = face, face, face

        self.edges.extend([e1, e2, e3])
        self.faces.append(face)

    def remove_triangle(self, face):
        edges_to_remove = []

        half_edges = [face.edge, face.edge.next, face.edge.next.next]

        for edge in half_edges:
            if edge.twin:
                edge.twin.twin = None
            else:
                edges_to_remove.append()

            key = (edge.origin.x, edge.origin.y, edge.next.origin.x, edge.next.origin.y)
            if key in self.edge_map:
                del self.edge_map[key]

        for edge in edges_to_remove:
            self.edges.remove(edge)

        self.faces.remove(face)

    def get_or_make_vertex(self, loc):
        for vertex in self.vertices:
            if (vertex.x, vertex.y) == loc:
                return vertex

        new_vertex = Vertex(*loc)
        self.vertices.append(new_vertex)

    def get_or_make_half_edge(self, v1, v2):
        key = (v1, v2)
        twin_key = (v2, v1)
        if key in self.edge_map:
            return(self.edge_map(key))
        else:
            edge = HalfEdge(v1)
            twin_edge = HalfEdge(v2)
            self.edge_map[key] = edge
            self.edge_map[twin_key] = edge


    def link_twins(self, edge):
        v1 = edge.origin
        v2 = edge.next.origin
        key = (v1, v2)
        twin_key = (v2, v1)

        if twin_key in self.edge_map:
            twin_edge = self.edge_map.pop(twin_key)
            edge.twin = twin_edge
            twin_edge.twin = edge

        else:
            self.edge_map[key] = edge


# Functions


def super_triangle(self, nodeList, expansion_factor=1.2):
    nodeArray = np.array(nodeList)
    min_x, min_y = np.min(nodeArray, axis=0)
    max_x, max_y = np.max(nodeArray, axis=0)

    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)

    buffer = expansion_factor * max_dim
    center_x = (min_x + max_x) / 2

    point1 = (center_x - buffer, min_y - 0.5 * buffer)  # Bottom left
    point2 = (center_x + buffer, min_y - 0.5 * buffer)  # Bottom right
    point3 = (center_x, max_y + buffer)  # Top center

    return point1, point2, point3


def circumcenter(v1, v2, v3):
    return


def circumradius(v1, v2, v3):
    return


# Algorithms


def bowyer_watson(nodeList):
    mesh = TriangleMesh
    n1, n2, n3 = super_triangle(nodeList)
    mesh.add_triangle(n1, n2, n3)

    for node in nodeList:
        badTriangles = ()
        for triangle in mesh:
            pass

    return


# Instantiation
