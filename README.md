# TorchModels.jl

TorchNLPModels.jl is a [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) interface for PyTorch models written in Python.

## Installation

Install PyTorch in Python through PyCall's Conda:

```julia
ENV["PYTHON"] = ""
using Pkg
using PyCall.Conda
Pkg.build("PyCall")
Conda.add("pytorch", channel="pytorch")
```
