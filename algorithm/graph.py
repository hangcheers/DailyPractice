# graph adjacency matrix

class Graph:
    def __init__(self, numvertex):
        """

        :param numvertex: the number of vertex
        """
        self.adjMatrix = [[-1] * numvertex for x in range(numvertex)]
        # assume numvertex=5
        # [-1]*5=[-1,-1,-1,-1,-1]
        self.numvertex = numvertex
        self.vertices = {}
        self.verticeslist = [0] * numvertex
        # [0]*5=[0,0,0,0,0]

    def set_vertex(self, vtx, id):
        if 0 <= vtx <= self.numvertex:
            self.vertices[id] = vtx
            self.verticeslist[vtx] = id

    def set_edge(self, frm, to, cost=0):
        """

        :param frm: the index of starting point
        :param to: the index of ending point
        :param cost: weight
        :return:
        """
        frm = self.vertices[frm]
        to = self.vertices[to]
        self.adjMatrix[frm][to] = cost
        # self.adjMatrix[to][frm] = cost

    def get_vertex(self):
        return self.verticeslist

    def get_edges(self):
        edges = []
        for i in range(self.numvertex):
            for j in range(self.numvertex):
                if (self.adjMatrix[i][j] != -1):
                    edges.append((self.verticeslist[i], self.verticeslist[j], self.adjMatrix[i][j]))
        return edges

    def get_matrix(self):
        return self.adjMatrix


G = Graph(5)
G.set_vertex(0, 'a')
G.set_vertex(1, 'b')
G.set_vertex(2, 'c')
G.set_vertex(3, 'd')
G.set_vertex(4, 'e')
G.set_edge('a', 'e', 10)
G.set_edge('a', 'c', 20)
G.set_edge('c', 'b', 30)
G.set_edge('b', 'e', 40)
G.set_edge('e', 'd', 50)
print("Vertices of Graph")
print(G.get_vertex())
print("Edges of Graph")
print(G.get_edges())
print("Adjacency Matrix of Graph")
print(G.get_matrix())

输出结果为：
Vertices of Graph
['a', 'b', 'c', 'd', 'e']
Edges of Graph
[('a', 'c', 20), ('a', 'e', 10), ('b', 'e', 40), ('c', 'b', 30), ('e', 'd', 50)]
Adjacency Matrix of Graph
[[-1, -1, 20, -1, 10], [-1, -1, -1, -1, 40], [-1, 30, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, 50, -1]]
