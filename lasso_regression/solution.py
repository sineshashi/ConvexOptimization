from typing import Tuple, Any
import pandas as pd, cvxpy as cp, numpy as np

def prompt_input() -> str:
    print("Write the input filename, there must be a column named 'output' with real values")
    return input()

def validate_and_provide_input_and_output(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filename)
    if df.isna().any().any() or df.isnull().any().any():
        raise TypeError("NAN or NULL can not be passed.")
    target = df["target"]
    if target.apply(lambda x: not isinstance(x, float) and not isinstance(x, int)).any():
        raise ValueError("Target must real number.")
    del df["target"]
    features = df.to_numpy()
    for arr in features:
        for num in arr:
            float(num)
    return [features, target.to_numpy()]

def formulate_and_solve_the_problem(features:np.ndarray, target:np.ndarray, lmbda: float):
    w = cp.Variable(shape=len(features[0]))
    b = cp.Variable()
    objective = cp.Minimize(cp.sum_squares(features @ w + b - target)/(2*len(features)) + lmbda * cp.norm1(w))
    problem = cp.Problem(
        objective
    )
    optval_or_err = problem.solve()
    try:
        opt = float(optval_or_err)
    except Exception as e:
        raise e
    return {
        "minimum_error_and_regularization": opt,
        "weight_vector_w": w.value,
        "bias_term_b": b.value,
        "regularization_param": lmbda
    }

if __name__ == "__main__":
    filename = prompt_input()
    X, Y = validate_and_provide_input_and_output(filename)
    lambda_val = input("Regularization paramter lambda: ")
    solution = formulate_and_solve_the_problem(X, Y, lambda_val)
    print("Solution of the problem is:")
    for k, v in solution.items():
        print("name: ", k)
        print("value: ", v)
        print("..................................")