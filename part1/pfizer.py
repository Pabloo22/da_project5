import pulp
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# D
DISTANCE_MATRIX = [  # The commented number is the representative with the lowest distance
    [16.160, 24.080, 24.320, 21.120],  # 1
    [19.000, 26.470, 27.240, 17.330],  # 4
    [25.290, 32.490, 33.420, 12.250],  # 4
    [0.000, 7.930, 8.310, 36.120],  # 1
    [3.070, 6.440, 7.560, 37.360],  # 1
    [1.220, 7.510, 8.190, 36.290],  # 1
    [2.800, 10.310, 10.950, 33.500],  # 1
    [2.870, 5.070, 5.670, 38.800],  # 1
    [3.800, 8.010, 7.410, 38.160],  # 1
    [12.350, 4.520, 4.350, 48.270],  # 3
    [11.110, 3.480, 2.970, 47.140],  # 3
    [21.990, 22.020, 24.070, 39.860],  # 1
    [8.820, 3.300, 5.360, 43.310],  # 2
    [7.930, 0.000, 2.070, 43.750],  # 2
    [9.340, 2.250, 1.110, 45.430],  # 3
    [8.310, 2.070, 0.000, 44.430],  # 3
    [7.310, 2.440, 1.110, 43.430],  # 3
    [7.550, 0.750, 1.530, 43.520],  # 2
    [11.130, 18.410, 19.260, 25.400],  # 1
    [17.490, 23.440, 24.760, 23.210],  # 1
    [11.030, 18.930, 19.280, 25.430],  # 1
    [36.120, 43.750, 44.430, 0.000]  # 4
]

# P
LABOR_INTENSITY = [0.1609, 0.1164, 0.1026, 0.1516, 0.0939,
                   0.1320, 0.0687, 0.0930, 0.2116, 0.2529,
                   0.0868, 0.0828, 0.0975, 0.8177, 0.4115,
                   0.3795, 0.0710, 0.0427, 0.1043, 0.0997,
                   0.1698, 0.2531]

# A
CURRENT_ASSIGNMENT = [
    [0, 0, 0, 1],  # 4
    [0, 0, 0, 1],  # 4
    [0, 0, 0, 1],  # 4
    [1, 0, 0, 0],  # 1
    [1, 0, 0, 0],  # 1
    [1, 0, 0, 0],  # 1
    [1, 0, 0, 0],  # 1
    [1, 0, 0, 0],  # 1
    [0, 0, 1, 0],  # 3
    [0, 1, 0, 0],  # 2
    [0, 1, 0, 0],  # 2
    [0, 1, 0, 0],  # 2
    [0, 1, 0, 0],  # 2
    [0, 1, 0, 0],  # 2
    [1, 0, 0, 0],  # 1
    [0, 0, 1, 0],  # 3
    [0, 0, 1, 0],  # 3
    [0, 0, 1, 0],  # 3
    [0, 0, 0, 1],  # 4
    [0, 0, 0, 1],  # 4
    [0, 0, 0, 1],  # 4
    [0, 0, 0, 1],  # 4
]

# L
CURRENT_REPRESENTATIVES_LOCATIONS = [4, 14, 16, 22]


def define_problem(distance_matrix, labor_intensity):
    # Number of regions and representatives
    num_regions = len(distance_matrix)
    num_reps = len(distance_matrix[0])

    # Define the problem
    problem = LpProblem("Pfizer-Optimization", LpMinimize)

    # Decision variables: assign[i][j] = 1 if representative j is assigned to region i
    assign = [[LpVariable(f"assign_{i}_{j}", cat="Binary") for j in range(num_reps)] for i in range(num_regions)]

    # Auxiliary variable: delta[i][j] represents the absolute difference in assignments
    delta = [[LpVariable(f"delta_{i}_{j}", cat="Binary") for j in range(num_reps)] for i in range(num_regions)]

    # Auxiliary variable: weighted_delta[i][j] represents the labor intensity weighted difference in assignments
    weighted_delta = [[LpVariable(f"weighted_delta_{i}_{j}", lowBound=0) for j in range(num_reps)]
                      for i in range(num_regions)]

    # Constraint: each region must have exactly one representative assigned
    for i in range(num_regions):
        problem += lpSum(assign[i][j] for j in range(num_reps)) == 1

    # Constraint: total labor intensity of the regions allocated to each representative should be in
    # the range [0.9; 1.1]
    for j in range(num_reps):
        problem += lpSum(labor_intensity[i] * assign[i][j] for i in range(num_regions)) >= 0.9
        problem += lpSum(labor_intensity[i] * assign[i][j] for i in range(num_regions)) <= 1.1

    return problem, assign, delta, weighted_delta


def optimize_assignments(problem,
                         assign,
                         delta,
                         weighted_delta,
                         distance_matrix,
                         labor_intensity,
                         current_assignment,
                         epsilon,
                         verbose=False):
    # Number of regions and representatives
    num_regions = len(distance_matrix)
    num_reps = len(distance_matrix[0])

    # Objective function f1: minimize the sum of distances
    problem += lpSum(distance_matrix[i][j] * assign[i][j] for i in range(num_regions)
                     for j in range(num_reps))

    # Constraints that define the auxiliary variable delta
    for i in range(num_regions):
        for j in range(num_reps):
            # Create two auxiliary variables: x1 and x2
            x1 = LpVariable(f"x1_{i}_{j}", cat="Binary")
            x2 = LpVariable(f"x2_{i}_{j}", cat="Binary")
            b = LpVariable(f"x3_{i}_{j}", cat="Binary")
            problem += x1 - x2 == assign[i][j] - current_assignment[i][j]
            problem += x1 + x2 == delta[i][j]

            # Ensure that at least one of x1 and x2 is 0
            problem += x1 + x2 <= b

    # Constraints that define the auxiliary variable weighted_delta
    for i in range(num_regions):
        for j in range(num_reps):
            problem += weighted_delta[i][j] == labor_intensity[i] * delta[i][j]

    # Constraint: the sum of labor intensity weighted changes in assignments should not exceed ε
    # We divide by 2 because we are counting each change twice
    problem += lpSum(weighted_delta[i][j] for i in range(num_regions) for j in range(num_reps)) / 2 <= epsilon

    if verbose:
        print(problem)
    # Solve the problem
    problem.solve(pulp.GLPK(msg=False))

    # Retrieve the optimized assignments and the objective value
    optimized_assignments = [[int(assign[i][j].varValue) for j in range(num_reps)] for i in range(num_regions)]
    objective_value = pulp.value(problem.objective)

    # Retrieve the sum of weighted deltas
    sum_weighted_delta = pulp.value(lpSum(weighted_delta[i][j] for i in range(num_regions) for j in range(num_reps)))

    # Print weighted deltas values
    if verbose:
        print("Delta values:")
        for i in range(num_regions):
            for j in range(num_reps):
                print(f"{delta[i][j].name}: {delta[i][j].varValue}")
        print("Weighted deltas:")
        for i in range(num_regions):
            for j in range(num_reps):
                print(f"{weighted_delta[i][j].name}: {weighted_delta[i][j].varValue}")
    f2_value = sum_weighted_delta / 2
    return optimized_assignments, objective_value, f2_value


def generate_pareto_solutions(distance_matrix, labor_intensity, current_assignment, epsilon_values):
    solutions = []
    for epsilon in tqdm.tqdm(epsilon_values):
        optimized_assignments, objective_value, f2_value = find_solution(distance_matrix,
                                                                         labor_intensity,
                                                                         current_assignment,
                                                                         epsilon)
        solutions.append((epsilon, objective_value, f2_value, optimized_assignments))
    return solutions


def find_solution(distance_matrix, labor_intensity, current_assignment, epsilon, verbose=False):
    problem, assign, deltas, weighted_delta = define_problem(distance_matrix, labor_intensity)
    optimized_assignments, objective_value, f2_value = optimize_assignments(problem,
                                                                            assign,
                                                                            deltas,
                                                                            weighted_delta,
                                                                            distance_matrix,
                                                                            labor_intensity,
                                                                            current_assignment,
                                                                            epsilon,
                                                                            verbose=verbose)
    return optimized_assignments, objective_value, f2_value


def plot_pareto_solutions(objective_values, f2_values):
    plt.style.use("fivethirtyeight")

    f2_values = np.array(f2_values) * 100
    plt.figure(figsize=(10, 6))
    plt.scatter(f2_values, objective_values)
    plt.xlabel("% of labor intensity changes ($f_2$)")
    plt.ylabel("Total distance ($f_1$)")
    plt.title("Pareto Optimal Solutions")

    # Set the x and y ticks to the unique values of your data
    # Round x-axis labels to two decimal places
    x_ticks = np.unique(f2_values)
    x_ticks = np.round(x_ticks)
    plt.xticks(x_ticks)

    # Round y-axis labels to the nearest integer
    y_ticks = np.unique(objective_values)
    y_ticks = np.round(y_ticks)
    plt.yticks(y_ticks)

    plt.tight_layout()

    # Save the figure as pdf
    plt.savefig("pareto_optimal_solutions.pdf")


def print_solution(optimized_assignments, objective_value, sum_weighted_delta):
    print(f"Objective value (f1): {objective_value}")
    print(f"Objective value (f2): {sum_weighted_delta}")
    print("Optimized assignments:")
    for i, row in enumerate(optimized_assignments, start=1):
        print(f"Region {i}: Representative {row.index(1) + 1}")


def example(epsilon, verbose=False):
    # Distance from each region to each representative's office
    distance_matrix = [
        [2, 7],  # Region 1
        [9, 4],  # Region 2
        [5, 8],  # Region 3
        [6, 3],  # Region 4
    ]

    # The labor intensity for each region
    labor_intensity = [0.5, 0.5, 0.5, 0.5]

    # The current assignment of representatives to regions
    current_assignment = [
        [0, 1],  # Region 1 is assigned to Rep 2
        [1, 0],  # Region 2 is assigned to Rep 1
        [0, 1],  # Region 3 is assigned to Rep 2
        [1, 0],  # Region 4 is assigned to Rep 1
    ]

    optimized_assignments, f1_value, f2_value = find_solution(distance_matrix, labor_intensity,
                                                              current_assignment,
                                                              epsilon=epsilon, verbose=verbose)
    f2_value_normalized = f2(optimized_assignments, labor_intensity, current_assignment)
    print(f"f2_value: {f2_value / 2}")
    print(f"f2_value_: {f2_value_normalized}")
    print_solution(optimized_assignments, f1_value, f2_value_normalized)


def f2(new_assignments: list[list[int, int, int, int]],
       labor_intensity: list[float],
       old_assignments: list[list[int, int, int, int]]) -> float:
    """Calculates the value of the second objective function normalized [0, 1].

    Args:
        new_assignments: Binary matrix of shape (num_regions, num_reps) with the optimized assignments.
            optimized_assignments[i][j] = 1 if region i is now assigned to representative j, 0 otherwise.
        labor_intensity: The given matrix P. List of length num_regions with the labor intensity for each region.
            labor_intensity[i] is the labor intensity for region i.
        old_assignments: The given matrix A. Binary matrix of shape (num_regions, num_reps) with the current
            assignments. current_assignment[i][j] = 1 if region i was assigned to representative j, 0 otherwise.
    """
    num_regions = len(old_assignments)
    num_reps = len(old_assignments[0])
    f2_value = 0
    for i in range(num_regions):
        if old_assignments[i] != new_assignments[i]:
            f2_value += labor_intensity[i]
    return f2_value / num_reps


def solve_problem(epsilon, verbose=False):
    optimized_assignments, f1_value, f2_value = find_solution(DISTANCE_MATRIX,
                                                              LABOR_INTENSITY,
                                                              CURRENT_ASSIGNMENT,
                                                              epsilon=epsilon,
                                                              verbose=verbose)
    print_solution(optimized_assignments, f1_value, f2_value / 2)


def check_solution(optimized_assignments):
    for optimized_assignment, previous_assignment, distances in zip(optimized_assignments,
                                                                    CURRENT_ASSIGNMENT,
                                                                    DISTANCE_MATRIX):
        if sum(optimized_assignment) != 1:
            return False

    return True


def print_current_labor_intensity():
    """This code is useful to justify why there is no feasible solution for low values of epsilon."""
    print("Current labor intensity for each representative:")
    representatives_labor_intensity = [0] * 4
    for assignment, labor_intensity in zip(CURRENT_ASSIGNMENT, LABOR_INTENSITY):
        for i, rep in enumerate(assignment):
            representatives_labor_intensity[i] += labor_intensity * rep
    for i, labor_intensity in enumerate(representatives_labor_intensity):
        print(f"Rep {i + 1}: {labor_intensity}")


def main(n_epsilons=10):
    epsilon_values = np.linspace(0, 4, n_epsilons)
    solutions = generate_pareto_solutions(DISTANCE_MATRIX, LABOR_INTENSITY, CURRENT_ASSIGNMENT, epsilon_values)
    valid_solutions = []
    for i, solution in enumerate(solutions):
        _, objective_value, f2_value, optimized_assignments = solution
        if not check_solution(optimized_assignments):
            continue

        f2_value_normalized = f2(optimized_assignments, LABOR_INTENSITY, CURRENT_ASSIGNMENT)
        assert abs(f2_value_normalized - f2_value / 4) <= 0.0001, f"{f2_value_normalized} != {f2_value / 4}"
        valid_solutions.append((objective_value, f2_value_normalized))

    plot_pareto_solutions([s[0] for s in valid_solutions], [s[1] for s in valid_solutions])


if __name__ == "__main__":
    # print_current_labor_intensity()
    # main(n_epsilons=500)
    solve_problem(epsilon=2, verbose=False)
