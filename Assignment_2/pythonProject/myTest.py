from linear_regression import compute_linear_regression
import numpy as np

VERBOSE = False


def noisy_case():
    """Original noisy case from lecturer's tester"""
    if VERBOSE:
        print("Noisy case")
    X = np.array([
        [-2.912, 8.045],
        [-1.078, -3.956],
        [5.067, 5.032],
        [7.921, 2.084],
        [-6.935, -7.023],
        [6.912, -9.978],
        [-0.934, 4.056],
        [-5.932, -9.912],
        [-8.045, 8.978],
        [-1.056, 3.089],
    ])
    y = np.array([[16], [-6], [18], [15], [-18], [-10], [10], [-23], [13], [8]])
    ref_weights = np.array([3, 1, 2])

    try:
        weights, cost = compute_linear_regression(X, y, alpha=0.01, max_iterations=10_000)
    except Exception as e:
        if VERBOSE:
            print(e)
        return 0

    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if VERBOSE:
        print(f"Weight difference: {weight_diff}")

    if weight_diff < 0.06:
        return 100
    elif weight_diff < 5e-2:
        return 90
    elif weight_diff < 5e-1:
        return 80
    else:
        return 50


def simple_case():
    """Original simple case from lecturer's tester"""
    if VERBOSE:
        print("Simple case")
    X = np.array([
        [-3, 8],
        [-1, -4],
        [5, 5],
        [8, 2],
        [-7, -7],
        [7, -10],
        [-1, 4],
        [-6, -10],
        [-8, 9],
        [-1, 3],
    ])
    y = np.array([[16], [-6], [18], [15], [-18], [-10], [10], [-23], [13], [8]])
    ref_weights = np.array([3, 1, 2])

    try:
        weights, cost = compute_linear_regression(X, y, alpha=0.01, max_iterations=10_000)
    except Exception as e:
        if VERBOSE:
            print(e)
        return 0

    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if VERBOSE:
        print(f"Weight difference: {weight_diff}")

    if weight_diff < 5e-4:
        return 100
    elif weight_diff < 1e-3:
        return 90
    elif weight_diff < 2e-2:
        return 80
    else:
        return 50


def scaled_case():
    """Test case with moderately scaled values"""
    if VERBOSE:
        print("Scaled case")
    X = np.array([
        [-6, 16],
        [-2, -8],
        [10, 10],
        [16, 4],
        [-14, -14],
        [14, -20],
        [-2, 8],
        [-12, -20],
        [-16, 18],
        [-2, 6],
    ])
    y = np.array([[32], [-12], [36], [30], [-36], [-20], [20], [-46], [26], [16]])
    ref_weights = np.array([6, 1, 2])  # corresponds to y = 6 + 1*x1 + 2*x2

    try:
        weights, cost = compute_linear_regression(X, y, alpha=0.005, max_iterations=10_000)
    except Exception as e:
        if VERBOSE:
            print(e)
        return 0

    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if VERBOSE:
        print(f"Weight difference: {weight_diff}")

    if weight_diff < 0.1:
        return 100
    elif weight_diff < 0.5:
        return 90
    elif weight_diff < 1.0:
        return 80
    else:
        return 50


def small_scale_case():
    """Test case with smaller scale values"""
    if VERBOSE:
        print("Small scale case")
    X = np.array([
        [-0.3, 0.8],
        [-0.1, -0.4],
        [0.5, 0.5],
        [0.8, 0.2],
        [-0.7, -0.7],
        [0.7, -1.0],
        [-0.1, 0.4],
        [-0.6, -1.0],
        [-0.8, 0.9],
        [-0.1, 0.3],
    ])
    y = np.array([[1.6], [-0.6], [1.8], [1.5], [-1.8], [-1.0], [1.0], [-2.3], [1.3], [0.8]])
    ref_weights = np.array([0.3, 1, 2])

    try:
        weights, cost = compute_linear_regression(X, y, alpha=0.1, max_iterations=10_000)
    except Exception as e:
        if VERBOSE:
            print(e)
        return 0

    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if VERBOSE:
        print(f"Weight difference: {weight_diff}")

    if weight_diff < 1e-3:
        return 100
    elif weight_diff < 1e-2:
        return 90
    elif weight_diff < 5e-2:
        return 80
    else:
        return 50


def negative_bias_case():
    """Test case with negative bias term"""
    if VERBOSE:
        print("Negative bias case")
    X = np.array([
        [-3, 8],
        [-1, -4],
        [5, 5],
        [8, 2],
        [-7, -7],
        [7, -10],
        [-1, 4],
        [-6, -10],
        [-8, 9],
        [-1, 3],
    ])
    y = np.array([[10], [-12], [12], [9], [-24], [-16], [4], [-29], [7], [2]])
    ref_weights = np.array([-3, 1, 2])

    try:
        weights, cost = compute_linear_regression(X, y, alpha=0.01, max_iterations=10_000)
    except Exception as e:
        if VERBOSE:
            print(e)
        return 0

    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if VERBOSE:
        print(f"Weight difference: {weight_diff}")

    if weight_diff < 5e-4:
        return 100
    elif weight_diff < 1e-3:
        return 90
    elif weight_diff < 2e-2:
        return 80
    else:
        return 50


def sparse_case():
    """Test case with moderately sparse data distribution"""
    X = np.array([
        [-5, 10],
        [0, 0],
        [7, 7],
        [10, 2],
        [-10, -10],
        [10, -12],
        [0, 5],
        [-7, -12],
        [-10, 11],
        [0, 4],
    ])
    y = np.array([[20], [0], [22], [17], [-22], [-12], [12], [-29], [16], [10]])

    try:
        weights, cost = compute_linear_regression(X, y, alpha=0.01, max_iterations=10_000)

        # Compute and show predictions
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        predictions = np.dot(X_b, weights).reshape(-1, 1)  # Reshape to match y

        # Calculate metrics correctly
        mae = np.mean(np.abs(predictions - y))
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - predictions) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Score based on prediction accuracy
        score = 50
        if mae < 2.0 and r2 > 0.95:
            score = 100
        elif mae < 3.0 and r2 > 0.90:
            score = 90
        elif mae < 4.0 and r2 > 0.85:
            score = 80

        return score

    except Exception as e:
        print(f"Exception occurred: {e}")
        return 0

if __name__ == "__main__":
    test_cases = [
        ("Noisy Case", noisy_case),
        ("Simple Case", simple_case),
        ("Scaled Case", scaled_case),
        ("Small Scale Case", small_scale_case),
        ("Negative Bias Case", negative_bias_case),
        ("Sparse Case", sparse_case)
    ]

    total_score = 0
    print("\nRunning all test cases:")
    print("-" * 40)

    for name, test_func in test_cases:
        score = test_func()
        print(f"{name}: {score}")
        total_score += score

    average_score = total_score / len(test_cases)
    print("-" * 40)
    print(f"Average Score: {average_score:.2f}")