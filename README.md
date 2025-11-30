# Biclique Decomposition via Reduction to SAT

This project provides a SAT-based solver for computing the **biclique
cover number** (also known as bipartite dimension) of an undirected
graph.\
A *biclique* is a complete bipartite subgraph, and the biclique cover
number is the smallest number of bicliques whose union covers all edges
of the graph.

The solver encodes the problem into propositional logic (CNF) and
invokes the **Glucose SAT solver** to determine whether the graph can be
decomposed into *k* bicliques. The program iteratively increases *k*
until a valid decomposition is found.



## Problem Description

Given an undirected graph ( G = (V, E) ), the goal is to determine its
**biclique cover number BFS(G)**. That is, find the smallest integer ( k
) such that the edges of (G) can be represented as the union of edges
from ( k ) bicliques.

Because SAT is a **decision problem** solver, we transform the optimization
problem into:

> "What is the smallest number of bicliques needed?"

into repeated decision problems:

> "Does there exist a decomposition with exactly ( k ) bicliques?"

The solver starts from ( k = 1 ) and increases *k* until a satisfying
assignment is found.



## Input Format

An instance file must have the following structure:

    <max_iterations>
    <number_of_vertices>
    u v
    u v
    ...

-   `max_iterations`: maximum allowed value of k (0 = unlimited)
-   `number_of_vertices`: nodes are assumed to be labeled 1..n
-   Following lines list edges of the graph, one per line

Example:

    5
    4
    1 2
    2 3
    3 4
    1 4



## SAT Encoding

We introduce these families of literals:

### 1. Biclique Edge Variables

For each unordered pair of vertices, we create a variable representing whether biclique b includes that edge. True edges must be covered by at least one biclique; false edges must be excluded from all bicliques.

->

For each biclique  \( b \in \{1,\dots,k\} \)  and each unordered pair
of vertices  (u, v) , a variable:

    E_b(u,v)

represents whether the bth-biclique includes edge (u,v).

If (u,v) is an actual edge in the graph, then it must be included in *at
least one* biclique:

    E_1(u,v) ∨ E_2(u,v) ∨ ... ∨ E_k(u,v)

If (u,v) is *not* an edge in the graph, it must be excluded from all
bicliques:

    ¬E_b(u,v)



### 2. Partition Variables for Biclique b

For each biclique b, every vertex belongs to either side A or side B:

    A_b(v)
    B_b(v)

Following constraints ensure the sets A_b and B_b properly define a
biclique structure.



### 3. Edge--Partition Consistency

If an edge (u,v) is included in biclique b (i.e., E_b(u,v) = True),\
then u and v must lie on *opposite sides* of the bipartition but these two variants must be mutually exclusive:

    E_b(u,v) → (((A_b(u) ∧ B_b(v)) ∨ (A_b(v) ∧ B_b(u))) ∧ ¬ ((A_b(u) ∧ B_b(v)) ∧ (A_b(v) ∧ B_b(u))))



### 4. Biclique Structure Consistency

If a vertex u belongs to A_b, then for every vertex v that is not u:

-   If v ∈ A_b, then no edge (u,v) is allowed in biclique b.

        A_b(u) => (A_b(v) => ¬E_b(u,v))
-   If v ∈ B_b, then (u,v) *must* be included.
    
        A_b(u) => (B_b(v) => E_b(u,v))


We use similar encoding for vertices present in B_b

This ensures A_b × B_b forms a complete bipartite graph.

# Usage

### Command-line Interface

```bash
python3 biclique_solver.py [-h] [-i INPUT] [-o OUTPUT]
```

### Options

| Option      | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `-i INPUT`  | Path to the instance file (default: `instances/instance2.in`)      |
| `-o OUTPUT` | Output DIMACS CNF file (default: `formula.cnf`)                    |



## Example
Input:

    python3 biclique_solver.py -i instances/example.in -o formula.cnf

Output:

    Running solver for 1 bicliques...
    Running solver for 2 bicliques...
    Bipartite dimension is: 2
    Total time taken: 26.535948991775513 s
    Time for last iteration (2): 1.1432709693908691 s
(times here are very inaccurate)



## Experiments

Model was mainly tested on trivial graphs and n-complete graphs which are the most dense and we know their dimension from theory (upperbound log2(n))
time measurements greatly increase after each dimension especially after 4 th 
see in "figure_1.png" dimension 5 was not even measurable

As for each iteration in a single instance we can observe that when we arrive on iteration of the solution the iteration before has much greater duration see in "figure_2.png" where measurements were conducted on 32 - complete graph

## Files Included

-   `biclique_solver.py` --- main program
-   `glucose-syrup` --- SAT solver binary
-   `instances/*.in` --- example graph files
-   `README.md` --- documentation (this file)

------------------------------------------------------------------------
## Notes
- i would like to visualize the bicliques but I wouldn't make it in time.
- Further testing should be made on different types of graphs 

