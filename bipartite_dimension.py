import subprocess
import numpy as np
from argparse import ArgumentParser


class Graph:
    def __init__(self, vertices, edges, edgeliterals):
        self.vertices = vertices
        self.edges = edges
        self.edgeliterals = edgeliterals # mapping of edge id to list
        #self.verticesliterals = len(vertices)
        self.numberofedgeliterals = len(vertices)*(len(vertices)-1)//2 # to avoid unnecessary literals we have complete graph number of edges
        #self.edgeliterals = len(vertices)*len(vertices)


    def get_edge_id(self, u, v, lits_before=0):
        if u > v:
            u, v = v, u
        return (u - 1) * (len(self.vertices) - u // 2) + (v - u) - 1 + lits_before
    def get_biclique_edge_id(self, b, u, v, lits_before=0):
        edge_id = self.get_edge_id(u, v)
        return (b-1) * self.numberofedgeliterals + edge_id + 1 + lits_before
    def get_Abiclique_vertice_id(self, b, v, lits_before=0):
        return (b-1) * len(self.vertices) + v + lits_before +1
    def get_Bbiclique_vertice_id(self, b, v, lits_before=0): #same as A but is used for clarity
        return (b-1) * len(self.vertices) + v  + lits_before +1

    



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


def encode(graph,k):
    

    cnf = [] # list of clauses

    nr_vars = 0
    literalsbeforeedges= nr_vars



    # Add literals for edges present in k-biclique
    biclique_edges = []
    for i in range(k):
        biclique_edges.append(graph.edgeliterals)
    # if edge exists in graph it must exist in at least one biclique
    for i in range(len(graph.edgeliterals)):
        is_edge, u, v = graph.edgeliterals[i]
        if is_edge:
            clause = []
            for b in range(1,k+1):
                biclique_edge_lit = graph.get_biclique_edge_id(b, u, v,nr_vars)
                clause.append(biclique_edge_lit)
            clause.append(0)
            cnf.append(clause)



    # if edge does not exist in graph it must not exist in any biclique?
    for i in range(len(graph.edgeliterals)):
        is_edge, u, v = graph.edgeliterals[i]
        if not is_edge:
            for b in range(1,k+1):
                biclique_edge_lit = graph.get_biclique_edge_id(b, u, v)
                cnf.append([-biclique_edge_lit, 0])
    nr_vars += k * graph.numberofedgeliterals
    literalsbefore_A= nr_vars




    # setup a vertices in biclique A and B
    vertices_in_biclique_A = []
    vertices_in_biclique_B = []
    for i in range(k):
        A = []
        for v in graph.vertices:
            A.append((graph.get_Abiclique_vertice_id(i+1, v,nr_vars),v+1))
        vertices_in_biclique_A.append(A)
    nr_vars +=  k * len(graph.vertices)
    literalsbefore_B= nr_vars
    for i in range(k):
        B = []
        for v in graph.vertices:
            B.append((graph.get_Bbiclique_vertice_id(i+1, v,nr_vars),v+1)) 
        vertices_in_biclique_B.append(B)     
    nr_vars +=  k * len(graph.vertices)



    # if an edge is present in a biclique k, then u in A_k and v in B_k or v in A_k and u in B_k , NOT both
    literalsbefore_A -= 1
    literalsbefore_B -= 1 
    for i in range(len(graph.edgeliterals)):
        _, u, v = graph.edgeliterals[i]
        for b in range(1,k+1):
            biclique_edge_lit = graph.get_biclique_edge_id(b, u, v,literalsbeforeedges)
            A_u_lit = graph.get_Abiclique_vertice_id(b, u,literalsbefore_A)
            B_v_lit = graph.get_Bbiclique_vertice_id(b, v,literalsbefore_B)
            A_v_lit = graph.get_Abiclique_vertice_id(b, v,literalsbefore_A)
            B_u_lit = graph.get_Bbiclique_vertice_id(b, u,literalsbefore_B)

            cnf.append([-biclique_edge_lit, A_u_lit, B_u_lit, 0])
            cnf.append([-biclique_edge_lit, A_u_lit, A_v_lit, 0])
            cnf.append([-biclique_edge_lit, B_u_lit, B_v_lit, 0])
            cnf.append([-biclique_edge_lit, B_v_lit, A_v_lit, 0])
            cnf.append([-biclique_edge_lit, -A_u_lit, -B_u_lit,-A_v_lit, -B_v_lit, 0])



    # if a vertice is in A_k then all other 
    for u in graph.vertices:
        for b in range(1,k+1):
            A_u_lit = graph.get_Abiclique_vertice_id(b, u,literalsbefore_A)
            for v in graph.vertices:
                if u != v:
                    A_v_lit = graph.get_Abiclique_vertice_id(b, v,literalsbefore_A)
                    B_v_lit = graph.get_Bbiclique_vertice_id(b, v,literalsbefore_B)
                    b_edge = graph.get_biclique_edge_id(b, u, v,literalsbeforeedges)
                    cnf.append([-A_u_lit, -A_v_lit, -b_edge, 0])
                    cnf.append([-A_u_lit, -B_v_lit, b_edge, 0])

    


    return cnf, nr_vars

def call_solver(cnf, nr_vars, output_file, solver="glucose-syrup", verb=0):
    # write the CNF formula to a DIMACS file
    with open(output_file, "w") as file:
        file.write(f"p cnf {nr_vars} {len(cnf)}\n")
        for clause in cnf:
            file.write(" ".join(map(str, clause)) + "\n")

    return subprocess.run(['./' + solver, '-model', '-verb=' + str(verb) , output_file], stdout=subprocess.PIPE)
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default="instances/instance3.in",
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
    while k <= maximum_iterations or maximum_iterations == 0:
    
        cnf, nr_vars = encode(graph,k)
        print("Running solver for", k, "bicliques...")
        result = call_solver(cnf, nr_vars, args.output, "glucose-syrup", 1)
        if result.returncode == 10:
            print("Bipartite dimension is", k)
            break
        k += 1


    # call the SAT solver and get the result
    #result = call_solver(cnf, nr_vars, args.output, args.solver, args.verb)

    # interpret the result and print it in a human-readable format
