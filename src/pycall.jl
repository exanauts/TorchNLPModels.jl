using PyCall

# Get the directory of the current Julia file
const script_dir = @__DIR__

# Add the directory containing the Python file to the Python path
pushfirst!(PyVector(pyimport("sys")."path"), script_dir)

# Import the Python module
py"""
import mwe
import torch
import numpy as np  # Import numpy

def as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return np.asarray(tensor)
"""

function torch_obj(input_tensor::Array{Float64})
    # Convert Julia array to PyTorch tensor
    input_tensor_torch = py"torch.tensor"(input_tensor, requires_grad=true).to(py"torch.float64")

    # Call the Python obj function
    result = py"mwe.f"(input_tensor_torch)

    # Convert the result back to a Julia scalar
    return convert(Float64, result)
end

function torch_grad(input_tensor::Array{Float64})
    # Convert Julia array to PyTorch tensor
    input_tensor_torch = py"torch.tensor"(input_tensor, requires_grad=true).to(py"torch.float64")

    # Call the Python grad function
    result = py"mwe.grad"(py"mwe.f", input_tensor_torch)

    # Convert the result back to a Julia array
    numpy_result = py"as_numpy"(result)
    return convert(Array{Float64}, numpy_result)
end

function torch_vhp(input_tensor::Array{Float64}, v_vector::Array{Float64})
    # Convert Julia arrays to PyTorch tensors
    input_tensor_torch = py"torch.tensor"(input_tensor, requires_grad=true).to(py"torch.float64")
    v_vector_torch = py"torch.tensor"(v_vector).to(py"torch.float64")

    # Call the Python vhp function
    result = py"mwe.vhp"(py"mwe.f", input_tensor_torch, v_vector_torch)

    # Convert the result back to a Julia array
    numpy_result = py"as_numpy"(result)
    @show numpy_result
    return convert(Array{Float64}, numpy_result)
end

# # Example usage:
# input_tensor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# v_vector = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

# result = julia_vhp(input_tensor, v_vector)
# println(result)