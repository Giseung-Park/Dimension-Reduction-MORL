import cvxpy as cp
import numpy as np
import pdb

# Generate a random non-trivial quadratic program.
m = 15
n = 10
p = 5
np.random.seed(1)
P = np.random.randn(n, n)
P = P.T @ P
qu = np.random.randn(n)
G = np.random.randn(m, n)
h = G @ np.random.randn(n)
A = np.random.randn(p, n)
b = np.random.randn(p)

pdb.set_trace()

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + qu.T @ x),
                 [G @ x <= h,
                  A @ x == b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)

# # Define the problem data
# A = np.array([[1, 2, 3], [4, 5, 6]])
# y = np.array([7, 8])
#
# # Define the variables
# x = cp.Variable(3)
#
# # Define the objective function
# objective = cp.Minimize(cp.sum_squares(A @ x - y))
#
# # Define the constraints
# constraints = [
#     x >= 0,
#     cp.sum(x) == 1
# ]
#
# # Formulate the problem
# problem = cp.Problem(objective, constraints)
#
# # Solve the problem
# problem.solve()
#
# # Output the results
# print(f"Optimal value: {problem.value}")
# print(f"Optimal x: {x.value}")
