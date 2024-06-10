from typing import Tuple, Any
import pandas as pd, cvxpy as cp, numpy as np

def prompt_input() -> str:
    print("Write the input filename, there must be a column named 'target' with real values")
    return input()

def validate_and_provide_input_and_output(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filename)
    if df.isna().any().any() or df.isnull().any().any():
        raise TypeError("NAN or NULL can not be passed.")
    target = df["target"]
    if any(target.apply(lambda x: x!=1 and x!=0)):
        raise ValueError("Target be either zero or one.")
    if target.apply(lambda x: not isinstance(x, float) and not isinstance(x, int)).any():
        raise ValueError("Target must real number.")
    del df["target"]
    features = df.to_numpy()
    for arr in features:
        for num in arr:
            float(num)
    return [features, target.to_numpy()]

def sigmoid(z):
    return 1/(1+cp.exp(-z))


def formulate_and_solve_the_problem(features:np.ndarray, target:np.ndarray  ):
    w = cp.Variable(shape=len(features[0]))
    b = cp.Variable()
    z = features@w + b
    objective_function = cp.sum(cp.logistic(-cp.multiply(2*target - 1, z)))
    objective_function = objective_function/len(features)
    objective = cp.Minimize(objective_function)
    problem = cp.Problem(
        objective
    )
    optval_or_err = problem.solve()
    try:
        opt = float(optval_or_err)
    except Exception as e:
        raise e
    return {
        "minimum_loss_possible": opt,
        "weight_vector_w": w.value,
        "bias_term_b": b.value
    }

if __name__ == "__main__":
    filename = prompt_input()
    X, Y = validate_and_provide_input_and_output(filename)
    solution = formulate_and_solve_the_problem(X, Y)
    print("Solution of the problem is:")
    for k, v in solution.items():
        print("name: ", k)
        print("value: ", v)
        print("..................................")