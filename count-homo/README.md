# customgraphhomo

A Python library for counting weighted graph homomorphisms with arbitrary vertex and edge weights.

## Introduction

This library implements functions to count homomorphisms between two graphs, a source graph H and a target graph G, based on definitions that incorporate arbitrary vertex and edge weights. Unlike some standard graph libraries, `customgraphhomo` is designed to handle custom weighting schemes precisely using Python's `Decimal` type.

The core functionality is based on the formula for counting homomorphisms, which involves summing over all possible mappings from the vertices of H to the vertices of G and multiplying the weights of mapped vertices and edges according to the specified definitions.

## Features

*   **Arbitrary Weighted Graphs:** Supports graphs where both vertices and edges can have arbitrary weights (non-negative numerical values).
*   **Precision with Decimal:** Uses Python's `decimal.Decimal` type for calculations to maintain high precision, especially crucial for computations involving large numbers or intricate weights. You can control the precision level.
*   **Flexible Weighting:** Node weights default to 1 if not explicitly provided, allowing the library to be easily used with unweighted graphs by only specifying edge weights.
*   **SageMath Integration:** Directly accepts SageMath `Graph` or `DiGraph` objects as input for the source (H) and target (G) graphs. The library automatically converts these to its internal `WeightedGraph` format.
*   **Homomorphism Counting:** Provides a function to compute the total count of weighted homomorphisms.

## Installation

This package assumes you have a working Python environment.

If you plan to use SageMath graph objects as input, you **must** have SageMath installed and accessible from your Python environment. This often involves running your Python scripts using the SageMath interpreter:

```bash
sage -python your_script_name.py
```

or configuring your environment to import SageMath modules.

Currently, this library is intended for direct use within your project by including the `counthomo.py` file. Standard Python package installation methods (like `pip install`) are not applicable without packaging the library first.

## Usage

The main function is `count_homomorphisms(H, G, precision=None)`.

*   `H`: The source graph. Can be a `WeightedGraph` object from this library or a SageMath `Graph`/`DiGraph`.
*   `G`: The target graph. Can be a `WeightedGraph` object from this library or a SageMath `Graph`/`DiGraph`.
*   `precision`: Optional integer to set the Decimal precision for this specific function call. If not provided, it uses the global precision set by `getcontext().prec`.

Here's a simple example demonstrating its use with SageMath graphs:

```python
# Ensure SageMath is installed and accessible (run this script with sage -python)
try:
    from sage.all import Graph, DiGraph, graphs
except ImportError:
    print("SageMath not found. Please ensure SageMath is installed and accessible.")
    exit()

from customgraphhomo.counthomo import count_homomorphisms
from customgraphhomo.counthomo import WeightedGraph # For reference, not needed for SageMath input

from decimal import getcontext

# Set global Decimal precision (optional, can also set in count_homomorphisms call)
getcontext().prec = 50

# Define your graphs using SageMath
# Example: a path graph of length 2 (P3) and a complete graph on 3 vertices (K3)
H_sage = graphs.PathGraph(2)
G_sage = graphs.CompleteGraph(3)

print(f"Source graph H (SageMath vertices): {H_sage.vertices()}")
print(f"Target graph G (SageMath vertices): {G_sage.vertices()}")

# Count homomorphisms from H to G using your library
# The library handles the conversion from SageMath graphs
hom_count = count_homomorphisms(H_sage, G_sage)

print(f"\nWeighted Homomorphism Count (H to G): {hom_count}")

# You can also specify precision directly in the function call
hom_count_high_prec = count_homomorphisms(H_sage, G_sage, precision=100)
print(f"Weighted Homomorphism Count (H to G, precision 100): {hom_count_high_prec}")

# You can also create WeightedGraph objects manually for custom weights
# Example: A simple graph with custom weights
# Nodes can be any hashable type
wg_H = WeightedGraph(nodes=['a', 'b'],
                     edges=[('a', 'b')],
                     node_weights={'a': Decimal('0.5'), 'b': Decimal('2.0')},
                     edge_weights={('a', 'b'): Decimal('1.5')})

wg_G = WeightedGraph(nodes=[1, 2, 3],
                     edges=[(1, 2), (2, 3), (3, 1)],
                     node_weights={1: Decimal('1.0'), 2: Decimal('1.0'), 3: Decimal('1.0')},
                     edge_weights={(1, 2): Decimal('1.0'), (2, 3): Decimal('1.0'), (3, 1): Decimal('1.0')}) # Default weights

print(f"\nSource graph H (WeightedGraph nodes): {wg_H.nodes()}")
print(f"Target graph G (WeightedGraph nodes): {wg_G.nodes()}")

hom_count_weighted = count_homomorphisms(wg_H, wg_G)
print(f"Weighted Homomorphism Count (WeightedGraph H to G): {hom_count_weighted}")
```

Remember that the time complexity of the algorithm is exponential in the number of vertices of the source graph H, so counting homomorphisms for large graphs can be computationally intensive.
