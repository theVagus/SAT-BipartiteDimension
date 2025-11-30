import subprocess
import numpy as np
import time
from argparse import ArgumentParser

import re

class Graph:
    def __init__(self, vertices, edges, edgeliterals):
        self.vertices = list(vertices)   # expect range(...) or list
        self.n = len(self.vertices)
        self.edges = edges
        self.edgeliterals = edgeliterals
        # number edges in complete graph on n vertices
        self.numberofedgeliterals = self.n * (self.n - 1) // 2

    def get_edge_id(self, u, v): # triangular number mapping of edge (u,v) to id in [0, n(n-1)/2-1]
        if u > v:
            u, v = v, u
        n = self.n
        return (u - 1) * (2 * n - (u)) // 2 + (v - u - 1)

    def get_biclique_edge_id(self, b, u, v, lits_before=0):

        edge_id = self.get_edge_id(u, v)  # 0-lits_befored within one biclique
        var_id = lits_before + (b - 1) * self.numberofedgeliterals + edge_id + 1
        return var_id

    def get_A_vertice_id(self, b, v, lits_before=0):        # layout: after all biclique-edge vars we place all A-variables blocks
        var_id = lits_before + (b - 1) * self.n + (v - 1) + 1
        return var_id

    def get_B_vertice_id(self, b, v, lits_before=0):
        var_id = lits_before + (b - 1) * self.n + (v - 1) + 1
        return var_id

    



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


def encode(graph, k):
    cnf = []
    # setup language in this order:
    # 1) biclique-edge variables: k blocks
    # 2) A variables: k blocks, each of size graph.n
    # 3) B variables: k blocks, each of size graph.n

    nr_vars = 0
    lits_before_edges = nr_vars
    nr_vars += k * graph.numberofedgeliterals

    lits_before_A = nr_vars
    nr_vars += k * graph.n

    lits_before_B = nr_vars
    nr_vars += k * graph.n

    # If an edge exists in the graph it must appear in at least one biclique:
    for (is_edge, u, v) in graph.edgeliterals:
        if is_edge:
            clause = []
            for b in range(1, k+1):
                clause.append(graph.get_biclique_edge_id(b, u, v, lits_before=lits_before_edges))
            clause.append(0)
            cnf.append(clause)

    # If an edge does not exist then it must not be chosen in any biclique:
    for (is_edge, u, v) in graph.edgeliterals:
        if not is_edge:
            for b in range(1, k+1):
                evar = graph.get_biclique_edge_id(b, u, v, lits_before=lits_before_edges)
                cnf.append([-evar, 0])

    # If an edge is chosen in a biclique then its vertice must be exclusively in A or B part:
    for (is_edge, u, v) in graph.edgeliterals:
        for b in range(1, k+1):
            evar = graph.get_biclique_edge_id(b, u, v, lits_before=lits_before_edges)
            A_u = graph.get_A_vertice_id(b, u, lits_before=lits_before_A)
            A_v = graph.get_A_vertice_id(b, v, lits_before=lits_before_A)
            B_u = graph.get_B_vertice_id(b, u, lits_before=lits_before_B)
            B_v = graph.get_B_vertice_id(b, v, lits_before=lits_before_B)

            # converted to CNF 
            cnf.append([-evar, A_u, A_v, 0])
            cnf.append([-evar, A_u, B_u, 0])
            cnf.append([-evar, B_v, A_v, 0])
            cnf.append([-evar, B_v, B_u, 0])
            cnf.append([-evar, -A_u, -B_u, -A_v, -B_v, 0])

    # If a vertex is in A part of a biclique then it has to have edges to all vertices in biclique's B part and none to A part:
    for u in graph.vertices:
        for b in range(1, k+1):
            A_u = graph.get_A_vertice_id(b, u, lits_before=lits_before_A)
            for v in graph.vertices:
                if u != v:
                    A_v = graph.get_A_vertice_id(b, v, lits_before=lits_before_A)
                    B_v = graph.get_B_vertice_id(b, v, lits_before=lits_before_B)
                    b_edge = graph.get_biclique_edge_id(b, u, v, lits_before=lits_before_edges)
                    cnf.append([-A_u, -A_v, -b_edge, 0])
                    cnf.append([-A_u, -B_v, b_edge, 0])
    # If a vertex is in B part of a biclique then it has to have edges to all vertices in biclique's A part and none to B part:
    for u in graph.vertices:
        for b in range(1, k+1):
            B_u = graph.get_B_vertice_id(b, u, lits_before=lits_before_B)
            for v in graph.vertices:
                if u != v:
                    A_v = graph.get_A_vertice_id(b, v, lits_before=lits_before_A)
                    B_v = graph.get_B_vertice_id(b, v, lits_before=lits_before_B)
                    b_edge = graph.get_biclique_edge_id(b, u, v, lits_before=lits_before_edges)
                    cnf.append([-B_u, -B_v, -b_edge, 0])
                    cnf.append([-B_u, -A_v, b_edge, 0])

    return cnf, nr_vars


def call_solver(cnf, nr_vars, output_file, solver="glucose-syrup", verb=0):
    # write the CNF formula to a DIMACS file
    with open(output_file, "w") as file:
        file.write(f"p cnf {nr_vars} {len(cnf)}\n")
        for clause in cnf:
            file.write(" ".join(map(str, clause)) + "\n")

    return subprocess.run(['./' + solver, '-model', '-verb=' + str(verb) , output_file], stdout=subprocess.PIPE)

import re

def parse_glucose_model(output_bytes):


    text = output_bytes.decode()

    if "UNSATISFIABLE" in text:
        return None
    if "SATISFIABLE" not in text:
        return None

    # All integers between lines starting with 'v'
    literals = []
    for line in text.splitlines():
        if line.startswith("v"):
            literals.extend([int(x) for x in line[1:].split() if x != '0'])
    return literals
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default="instances/instanceK32.in",
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
    k = 1
    completetime = 0
    kth_time_list = []
    while k <= maximum_iterations or maximum_iterations == 0:
    
        cnf, nr_vars = encode(graph,k)
        print("Running solver for", k, "bicliques...")

        start_time = time.time()
        result = call_solver(cnf, nr_vars, args.output, "glucose-syrup", 1)
        end_time = time.time()
        kth_time = end_time - start_time
        completetime += kth_time
        kth_time_list.append(kth_time)
    
        if result.returncode == 10:
            print("Bipartite dimension is", k)
            print("Total time taken:", completetime, "s")
            print(f"Time for last iteration ({k}): {kth_time} s")

            model = parse_glucose_model(result.stdout)
            break
        else:
            print("No solution for", k, "bicliques.")
        k += 1
    
    if k > maximum_iterations and maximum_iterations != 0:
        print("Bipartite dimension is greater than", maximum_iterations)