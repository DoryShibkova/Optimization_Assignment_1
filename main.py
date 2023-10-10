import numpy as np

# Read the objective function
variables_count = int(input("Enter the number of variables: "))
C = [int(x) for x in input("Enter the coefficients of the objective function: ").split()]
is_min = input("Maximize or minimize? (max/min): ") == "min"
if is_min:
    C = [-c for c in C]

# Read the constraints
constraints_count = int(input("Enter the number of constraints: "))
slack_count = constraints_count
A = []
for i in range(constraints_count):
    A.append([int(x) for x in input(f"Enter the coefficients of the constraint {i + 1}: ").split()])
b = [int(x) for x in input("Enter the right hand sides of the constraints: ").split()]
eps = float(input("Enter the precision (e.g. 0.001): "))

# Set the precision for printing
precision = int(np.log10(1 / eps))
np.set_printoptions(precision=precision, suppress=True)

# Show the problem
if is_min:
    print("\nMinimize:")
else:
    print("\nMaximize:")
objective = [f"{c * (-1 if is_min else 1)}*x{i + 1}" for i, c in enumerate(C)]
objective = " + ".join(objective)
print(f"z = {objective}")
print("Subject to:")
for i in range(constraints_count):
    constraint = [f"{c}*x{i + 1}" for i, c in enumerate(A[i])]
    constraint = " + ".join(constraint)
    print(f"{constraint} <= {b[i]}")

# Construct the initial tableau
matrix = np.zeros((constraints_count + 1, variables_count + slack_count + 1))
matrix[0, :variables_count] = C
matrix[1:, :variables_count] = A
matrix[1:, variables_count:-1] = np.identity(slack_count)
matrix[1:, -1] = np.array(b)
print(f"\nInitial matrix:\n{matrix}")

# Check if simplex method is applicable
if np.any(matrix[1:, -1] < 0):
    print("The method is not applicable!")
    exit()

# Iterate until the solution is optimal
i = 0
while not np.all(matrix[0, :-1] <= eps):
    i += 1
    print(f"\nIteration {i}.")

    pivot_column = np.argmax(matrix[0, :-1])
    print(f"Pivot: col {pivot_column}; ", end="")

    ratios = []
    for row in matrix[1:]:
        if row[pivot_column] > eps:
            ratios.append(row[-1] / row[pivot_column])
        else:
            ratios.append(np.inf)
    pivot_row = np.argmin(ratios) + 1
    print(f"row {pivot_row}; ", end="")

    pivot = matrix[pivot_row, pivot_column]
    print(f"val {np.round(pivot, precision)}")

    matrix[pivot_row] /= pivot
    for row in range(matrix.shape[0]):
        if row != pivot_row:
            matrix[row] -= matrix[row, pivot_column] * matrix[pivot_row]
    print(f"New matrix:\n{matrix}")

    if i > variables_count + constraints_count:
        break

# Find decision variables from the matrix
decision_variables = [0 for x in range(variables_count)]
for column in range(variables_count):
    for row in range(1, constraints_count + 1):
        if matrix[row, column] == 1:
            decision_variables[column] = np.round(matrix[row, -1], precision)
        elif matrix[row, column] != 0:
            decision_variables[column] = 0
            break

# Find the value of the objective function
objective_value = np.round(-matrix[0, -1], precision)
if -objective_value == objective_value:
    objective_value = 0.0

# Print the results
print(f"\nx = {decision_variables}")
print(f"z = {objective_value}")
