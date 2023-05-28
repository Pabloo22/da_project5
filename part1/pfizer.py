from pulp import *
import matplotlib.pyplot as plt

### distance matrix
D = [
    [16.160, 24.080, 24.320, 21.120],
    [19.000, 26.470, 27.240, 17.330],
    [25.290, 32.490, 33.420, 12.250],
    [0.000, 7.930, 8.310, 36.120],
    [3.070, 6.440, 7.560, 37.360],
    [1.220, 7.510, 8.190, 36.290],
    [2.800, 10.310, 10.950, 33.500],
    [2.870, 5.070, 5.670, 38.800],
    [3.800, 8.010, 7.410, 38.160],
    [12.350, 4.520, 4.350, 48.270],
    [11.110, 3.480, 2.970, 47.140],
    [21.990, 22.020, 24.070, 39.860],
    [8.820, 3.300, 5.360, 43.310],
    [7.930, 0.000, 2.070, 43.750],
    [9.340, 2.250, 1.110, 45.430],
    [8.310, 2.070, 0.000, 44.430],
    [7.310, 2.440, 1.110, 43.430],
    [7.550, 0.750, 1.530, 43.520],
    [11.130, 18.410, 19.260, 25.400],
    [17.490, 23.440, 24.760, 23.210],
    [11.030, 18.930, 19.280, 25.430],
    [36.120, 43.750, 44.430, 0.000]
]

### labor intensity
P = [0.1609, 0.1164, 0.1026, 0.1516, 0.0939, 0.1320, 0.0687, 0.0930, 0.2116, 0.2529, 0.0868, 0.0828, 0.0975, 0.8177,
     0.4115, 0.3795, 0.0710, 0.0427, 0.1043, 0.0997, 0.1698, 0.2531]

### current assignment
A = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

### current locations of representatives
L = [4, 14, 16, 22]


def create_problem(D, P, A):
    # number of regions and representatives
    n_regions = len(D)
    n_reps = len(D[0])

    # create a binary variable to state that a representative is going to a region
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat='Binary') for j in range(n_reps)] for i in range(n_regions)]

    # create the problem
    problem = pulp.LpProblem("Pfizer_representative_problem", pulp.LpMinimize)

    # objective function f1
    f1 = pulp.lpSum(D[i][j] * x[i][j] for i in range(n_regions) for j in range(n_reps))

    # add objective function f1 to the problem
    problem += f1

    # constraints
    for i in range(n_regions):
        problem += pulp.lpSum(x[i][j] for j in range(n_reps)) == 1  # each region must have exactly one representative assigned

    for j in range(n_reps):
        problem += pulp.lpSum(P[i] * x[i][j] for i in range(n_regions)) >= 0.9  # total labor intensity of the regions allocated to each representative it should be in the range [0.9; 1.1]
        problem += pulp.lpSum(P[i] * x[i][j] for i in range(n_regions)) <= 1.1

    return problem, x


def solve_problem(problem, x, A, P):
    # solve the problem
    problem.solve()

    # calculate f2
    f2 = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if x[i][j].varValue == 1 and A[i][j] == 0:
                f2 += P[i]
    f2 = f2 * 100

    return pulp.value(problem.objective), f2


def plot_results(results):
    f1_values, f2_values = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.scatter(f1_values, f2_values, color='b')
    plt.title('Pareto Optimal Solutions')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()


def main():
    # create the problem
    problem, x = create_problem(D, P, A)

    # solve the problem and get the results
    results = []
    for i in range(10):
        f1, f2 = solve_problem(problem, x, A, P)
        results.append((f1, f2))

    # plot the results
    plot_results(results)


if __name__ == "__main__":
    main()
