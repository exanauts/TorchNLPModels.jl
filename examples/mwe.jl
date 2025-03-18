# Test function: Rosenbrock function
function f(x)
    # Assumes x has length 2
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end



using JSOSolvers, ADNLPModels, NLPModels
using TorchNLPModels
using MadNLP

torch_file = "$(@__DIR__)/mwe"
# Rosenbrock
x0 = [-1.2, 1.0]
nlp = ADNLPModel(f, x0)
tnlp = TorchModel(x0, torch_file, "f", "grad", "vhp")
mnlp = TorchModel(x0, torch_file, "f", "grad", "vhp")
v = [1.0,1.0]
println("f(x0): ", f(x0))
println("grad: ", grad(nlp, x0))
println("hvprod: ", hprod(nlp, x0, v))
println("grad: ", grad(tnlp, x0))
println("hvprod: ", hprod(tnlp, x0, v))
stats = lbfgs(nlp) # or trunk, tron, R2



tstats = lbfgs(tnlp) # or trunk, tron, R2

println("Solution: ", tstats.solution)
println("Objective value: ", tstats.objective)
println("Number of iterations: ", tstats.iter)


# Create an NLP model for the optimization problem.
# nlp = ADNLPModel(f, [-1.2, 1.0])

@show x0
hess_coord(mnlp, x0)
H = zeros(3)
hess_coord!(mnlp, x0, H)
@show H
# Create a MadNLP solver that uses the BFGS quasi-Newton method.
solver = MadNLPSolver(
    mnlp;
    # hessian_approximation = MadNLP.BFGS,
    # callback = MadNLP.DenseCallback,
    # kkt_system = MadNLP.DenseKKTSystem,
    # linear_solver = MadNLP.LapackCPUSolver,
    # print_level = MadNLP.ERROR
)

# Solve the optimization problem.
# result = MadNLP.solve!(solver)
# println("Solution: ", result.solution)
# println("Objective value: ", result.objective)
# println("Number of iterations: ", result.iter)