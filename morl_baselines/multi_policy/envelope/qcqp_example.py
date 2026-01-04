import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x, Q, c):
    return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c, x)

# Define the equality constraint: sum(x_i^2) - 1 = 0
def constraint_eq(x):
    return np.sum(x**2) - 1

# Define the non-negativity constraint: x_i >= 0 for all i
def constraint_ineq(x):
    return x

# Define the matrix Q and vector c
Q = np.array([[2, 0], [0, 3]])
c = np.array([-1, -1])

# Initial guess for x
x0 = np.random.rand(Q.shape[0])

# Define the constraints
constraints = [{'type': 'eq', 'fun': constraint_eq},
               {'type': 'ineq', 'fun': constraint_ineq}]

# Define the bounds (all elements of x should be >= 0)
bounds = [(0, None) for _ in range(Q.shape[0])]

# Solve the problem
result = minimize(objective, x0, args=(Q, c), constraints=constraints, bounds=bounds, method='SLSQP')

# Check if the optimization was successful
if result.success:
    x_solution = result.x
    print("Optimal solution x:", x_solution)
    print("Objective value at x:", objective(x_solution, Q, c))
else:
    print("Optimization failed:", result.message)
