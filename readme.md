# Purpose

To demonstrate the locality sensitive hashing algorithms, purely in the context of hash generation. See [comparison.ipynb](comparison.ipynb) for a side-by-side hash generation (as well as a comparison to Gensim top N similarity search).

In contrast to this hash generation context, almost all online usages centre around the specific use case of vector similarity search, which has a specific set of requirements (e.g. the popular FAISS library doesn't expose a hash and instead exposes a further subdivided and optimised index)

The algorithms chosen are the most common algorithms along with some logical extensions for hashing appropriateness and hash-time performance.

In summary, [HierarchicalLsh](algorithms/HierarchicalLsh.py) is the gives the most correlated hash buckets, but is by far the worst in performance (both in pre-calculation and hash-time), and [HierarchicalHyperplaneLsh](algorithms/HierarchicalHyperplaneLsh.py) looks to approximate its correlation accuracy, but with a scalable hash-time performance (still with huge pre-calculation time).

# Python Notebook Setup

1. install python 3.10
1. install poetry
1. set poetry python version: `poetry env use C:/Python310/python.exe`
1. install poetry dependencies: `poetry install --no-root`
