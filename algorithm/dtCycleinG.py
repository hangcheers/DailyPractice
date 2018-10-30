from collections import defaultdict
# collection 是数据类型容器模块， defaultdict 构建的是类似dictionary的对象，
#其中的keys的值是自行确定赋值，但是value的类型是function_factory的类实例
"""
Disjoint Set Data Structure 并查集数据结构
Algorithm can be used to check whether an undirected graph contains cycle or not

"""


class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)  # default dictionary to store graph,the value is list

    # add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # a utility function to find the subset of an element i
    # the subsets in 1-D array, calls parent[]
    # all slots of parent array are initialize to -1
    def find_parent(self, parent, i):
        if parent[i] == -1:
            return i
        if parent[i] != -1:
            #当没有找到时，根据其父节点的引用节点向根一直追溯下去
            return self.find_parent(parent, parent[i])

    # a utility function to do union of two subsets
    # 即将一棵树的根连接到另一棵树上去
    def union(self, parent, x, y):
        x_set = self.find_parent(parent, x)
        y_set = self.find_parent(parent, y)
        parent[x_set] = y_set

    # check whether a given graph contains cycle or not
    def isCycle(self):
        # allocate memory for creating V subsets
        # initialize all subsets as all single element sets
        parent = [-1] * (self.V)

        # iterate through all edges of graph
        # find subset of both vertices of every edge
        # if both subsets are the same, then there is a cycle in the graph
        for i in self.graph:
            for j in self.graph[i]:
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                if x == y:
                    return True
                self.union(parent, x, y)


g = Graph(3)
g.addEdge(0, 1)
g.addEdge(1, 2)
g.addEdge(2, 0)

if g.isCycle():
    print("graph contains cycle")
else:
    print("graph doesnot contain cycle")
