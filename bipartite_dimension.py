import subprocess
import numpy as np
from argparse import ArgumentParser


class Graph:
    def __init__(self, vertices, edges, edgeliterals):
        self.vertices = vertices
        self.edges = edges
        self.edgeliterals = edgeliterals
        #self.verticesliterals = len(vertices)
        self.numberofedgeliterals = len(vertices)*(len(vertices)-1)//2 # to avoid unnecessary literals we have complete graph number of edges
        #self.edgeliterals = len(vertices)*len(vertices)

    #def get_vertice(self,n): # we dont need vertice literals
    #    return n + 1
    def get_edge_id(self, u, v):
        if u > v:
            u, v = v, u
        return (u - 1) * (len(self.vertices) - u // 2) + (v - u) - 1
    def get_biclique_edge_id(self, k, u, v):
        edge_id = self.get_edge_id(u, v)
        return k * self.numberofedgeliterals + edge_id + 1

    
def load_graph(file_path):
    edgelist = []
    with open(file_path, "r") as file:
        maximum_iterations = int(file.readline().strip())
        vertices = range(int(file.readline().strip()))
        edges  = [[False for _ in range(len(vertices))] for _ in range(len(vertices))]
        for line in file: # create matrix of edges and non-edges
            u, v = map(int, line.strip().split())
            edgevertices = sorted((u, v))
            edgelist.append(edgevertices)
            edges[edgevertices[0]-1][edgevertices[1]-1] = True
            edges[edgevertices[1]-1][edgevertices[0]-1] = True # shouldn't be necessary for undirected graph but just in case

    return vertices, edges, maximum_iterations
def edges_to_literals(edges): # convert adjacency matrix to edge literals list
    edgelit = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            edgelit.append((edges[i][j],i+1,j+1)) # store as (is_edge, u, v)
    return edgelit

def encode(graph, maximum_iterations):
    if maximum_iterations <1 : 
        maximum_iterations = float('inf')
        
    for k in range(1, maximum_iterations + 1):
        cnf = [] # list of clauses

        nr_vars = 0
        #setup edge literals
        nr_vars += graph.numberofedgeliterals
        # setup intial egde literals
        for i in range(len(graph.edgeliterals)):
            is_edge, u, v = graph.edgeliterals[i]
            lit = i + 1
            if is_edge:
                cnf.append([lit,0])  # edge must be present
            else:
                cnf.append([-lit,0])  # edge must be absent
        # Add edges present in biclique
        biclique_edges = []
        for i in range(k):
            biclique_edges.append(graph.edgeliterals)
        # if edge exists in graph it must exist in at least one biclique
        for i in range(len(graph.edgeliterals)):
            is_edge, u, v = graph.edgeliterals[i]
            if is_edge:
                clause = []
                for b in range(k):
                    biclique_edge_lit = graph.get_biclique_edge_id(b, u, v)
                    clause.append(biclique_edge_lit)
                clause.append(0)
                cnf.append(clause)





    # Encoding logic goes here

    return cnf, nr_vars

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default="instances/instance1.in",
        type=str,
        help=(
            "The instance file."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="formula.cnf",
        type=str,
        help=(
            "Output file for the DIMACS format (i.e. the CNF formula)."
        ),
    )

    args = parser.parse_args()

    # get the graph instance
    vertices,edges,maximum_iterations = load_graph(args.input)
    edgeliterals = edges_to_literals(edges)
    graph = Graph(vertices, edges, edgeliterals)

    # encode the problem to create CNF formula
    cnf, nr_vars = encode(graph, maximum_iterations)

    # call the SAT solver and get the result
    #result = call_solver(cnf, nr_vars, args.output, args.solver, args.verb)

    # interpret the result and print it in a human-readable format
