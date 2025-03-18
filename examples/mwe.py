import torch
import torch.autograd.functional

def vhp(func, x, v):
    # input dimension td_x ~ 300,000,000
    # v (bs, input_dim)
    # func : R^{input_dim} -> R+
    return torch.autograd.functional.vhp(func, x, v=v)[1]

def grad(func, x):
    y = func(x)
    grad = torch.autograd.grad(y, x)[0]
    return grad

def f(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

if __name__ == '__main__':
    # Example usage (optional, for testing within Python)
    input_tensor = torch.tensor([-1.2, 1.0], requires_grad=True)
    v_vector = torch.tensor([1.0, 1.0])
    print("f(x0): ", f(input_tensor))
    print("grad(f(x0)): ", grad(f, input_tensor))
    print("vhp(f(x0)): ", vhp(f, input_tensor, v_vector))