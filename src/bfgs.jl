# Test function: Rosenbrock function
function f(x)
    # Assumes x has length 2
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# Example usage with an initial guess using an optimization package:
# using Optim
# result = optimize(f, [ -1.2, 1.0 ], BFGS())
# println("Optimiz
using JSOSolvers, ADNLPModels, NLPModels
x = [-1.2, 1.0]
# Rosenbrock
nlp = ADNLPModel(f, x)
f(x)
stats = lbfgs(nlp, true_hessian=false, verbose=3) # or trunk, tron, R2
dx = copy(x)

println("obj: $(stats.objective), x: $(stats.solution), iter: $(stats.iter)")

grad!(nlp, x, dx)
v = [1.0,1.0]
Hv = zeros(2)
hprod!(nlp, x, v, Hv)
H = hess(nlp, x)
Hv = H * v
using MadNLP, ADNLPModels

# Create an NLP model for the optimization problem.
nlp = ADNLPModel(f, x)

# Create a MadNLP solver that uses the BFGS quasi-Newton method.
solver = MadNLPSolver(
    nlp;
    hessian_approximation = MadNLP.BFGS,
    # callback = MadNLP.DenseCallback,
    kkt_system = MadNLP.DenseKKTSystem,
    linear_solver = MadNLP.LapackCPUSolver,
    # print_level = MadNLP.ERROR
)

# Solve the optimization problem.
result = MadNLP.solve!(solver)

println("Optimized objective: ", result.objective)
println("Optimized solution: ", result.solution)