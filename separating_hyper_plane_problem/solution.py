from typing import Tuple, Any
import pandas as pd, cvxpy as cp, numpy as np

def prompt_input() -> str:
    print("Write the input filename, there must be a column named 'output' and values should either be 1 or -1:")
    return input()

def validate_and_provide_input_and_output(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filename)
    if df.isna().any().any() or df.isnull().any().any():
        raise TypeError("NAN or NULL can not be passed.")
    output = df["output"]
    if any([x not in {1, -1} for x in output]):
        raise ValueError("Output must either be 1 or -1.")
    del df["output"]
    input = df.to_numpy()
    for arr in input:
        for num in arr:
            float(num)
    return [input, output.to_numpy()]

def formulate_and_solve_the_problem(input:np.ndarray, output:np.ndarray):
    w = cp.Variable(shape=len(input[0]))
    b = cp.Variable()
    objective = cp.Minimize(0.5*(cp.norm(w)**2))
    constraints = []
    for arr, y in zip(input, output):
        constraints.append(y * (w@arr+b )>= 1)
    problem = cp.Problem(
        objective,
        constraints
    )
    optval_or_err = problem.solve()
    try:
        opt = float(optval_or_err)
    except Exception as e:
        raise e
    opt_value_for_original_prblm = 1/((2*opt)**0.5)
    return {
        "optimal_value": opt_value_for_original_prblm,
        "weight_vector_w": w.value,
        "bias_term_b": b.value
    }

if __name__ == "__main__":
    filename = prompt_input()
    input, output = validate_and_provide_input_and_output(filename)
    solution = formulate_and_solve_the_problem(input, output)
    print("Solution of the problem is:")
    for k, v in solution.items():
        print("name: ", k)
        print("value: ", v)
        print("..................................")