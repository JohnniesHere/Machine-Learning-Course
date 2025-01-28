from linear_regression import compute_linear_regression
import numpy as np


def test_zero_initialization():
    """Test if algorithm works with zero input"""
    X = np.zeros((5, 2))
    y = np.zeros((5, 1))
    weights, cost = compute_linear_regression(X, y)
    print("\nZero Initialization Test:")
    print(f"Weights: {weights}")
    print(f"Cost: {cost}")


def test_single_feature():
    """Test with single feature"""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[2], [4], [6], [8], [10]])  # y = 2x
    weights, cost = compute_linear_regression(X, y)
    print("\nSingle Feature Test:")
    print(f"Weights: {weights}")
    print(f"Cost: {cost}")


def test_different_learning_rates():
    """Test with different learning rates"""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([[2], [4], [6]])
    results = []
    alphas = [0.1, 0.01, 0.001]

    print("\nLearning Rate Sensitivity Test:")
    for alpha in alphas:
        weights, cost = compute_linear_regression(X, y, alpha=alpha)
        results.append((alpha, weights, cost))
        print(f"Alpha={alpha}:")
        print(f"  Weights: {weights}")
        print(f"  Cost: {cost}")


def test_different_iterations():
    """Test with different iteration counts"""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([[2], [4], [6]])
    iterations = [1, 10, 100, 1000]

    print("\nIteration Count Test:")
    for iters in iterations:
        weights, cost = compute_linear_regression(X, y, max_iterations=iters)
        print(f"Iterations={iters}:")
        print(f"  Weights: {weights}")
        print(f"  Cost: {cost}")


def test_extreme_values():
    """Test with extreme values"""
    X = np.array([[1e6, 1e6], [2e6, 2e6], [3e6, 3e6]])
    y = np.array([[1e6], [2e6], [3e6]])
    weights, cost = compute_linear_regression(X, y)
    print("\nExtreme Values Test:")
    print(f"Weights: {weights}")
    print(f"Cost: {cost}")


def test_negative_values():
    """Test with negative values"""
    X = np.array([[-1, -2], [-3, -4], [-5, -6]])
    y = np.array([[-3], [-7], [-11]])
    weights, cost = compute_linear_regression(X, y)
    print("\nNegative Values Test:")
    print(f"Weights: {weights}")
    print(f"Cost: {cost}")


def test_random_data():
    """Test with random data"""
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = np.random.randn(20, 1)
    weights, cost = compute_linear_regression(X, y)
    print("\nRandom Data Test:")
    print(f"Weights: {weights}")
    print(f"Cost: {cost}")


def test_perfect_fit():
    """Test with data that should have perfect fit"""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([[3], [6], [9]])  # y = x1 + 2x2
    weights, cost = compute_linear_regression(X, y)
    print("\nPerfect Fit Test:")
    print(f"Weights: {weights}")
    print(f"Cost: {cost}")
    print(f"Expected weights: [0, 1, 2]")


# Original test cases
def noisy_case():
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
        print(f"Error in noisy case: {e}")
        return 0
    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if weight_diff < 1e-3:
        return 100
    elif weight_diff < 1e-2:
        return 90
    elif weight_diff < 1e-1:
        return 80
    else:
        return 50


def simple_case():
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
        print(f"Error in simple case: {e}")
        return 0
    weight_diff = np.sqrt(np.mean(np.subtract(weights, ref_weights) ** 2))
    if weight_diff < 5e-4:
        return 100
    elif weight_diff < 1e-3:
        return 90
    elif weight_diff < 2e-2:
        return 80
    else:
        return 50


def test_error_handling():
    """Test error handling"""
    print("\nError Handling Tests:")
    try:
        # Test empty arrays
        X_empty = np.array([])
        y_empty = np.array([])
        weights, cost = compute_linear_regression(X_empty, y_empty)
        print("Empty array test failed to raise error")
    except Exception as e:
        print(f"Empty array test: {type(e).__name__}: {str(e)}")

    try:
        # Test mismatched dimensions
        X_bad = np.array([[1, 2], [3, 4]])
        y_bad = np.array([[1], [2], [3]])
        weights, cost = compute_linear_regression(X_bad, y_bad)
        print("Dimension mismatch test failed to raise error")
    except Exception as e:
        print(f"Dimension mismatch test: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    # Run all tests
    print("Running comprehensive test suite...")

    # Run basic functionality tests
    test_zero_initialization()
    test_single_feature()
    test_different_learning_rates()
    test_different_iterations()
    test_extreme_values()
    test_negative_values()
    test_random_data()
    test_perfect_fit()
    test_error_handling()

    # Run original graded tests
    print("\nRunning graded tests...")
    for case_name, test_func in [("Noisy case", noisy_case), ("Simple case", simple_case)]:
        print(f"\nTesting {case_name}:")
        score = test_func()
        print(f"Score: {score}")

        # Get test data
        if case_name == "Noisy case":
            X = np.array([[-2.912, 8.045], [-1.078, -3.956], [5.067, 5.032],
                          [7.921, 2.084], [-6.935, -7.023], [6.912, -9.978],
                          [-0.934, 4.056], [-5.932, -9.912], [-8.045, 8.978],
                          [-1.056, 3.089]])
        else:
            X = np.array([[-3, 8], [-1, -4], [5, 5], [8, 2], [-7, -7],
                          [7, -10], [-1, 4], [-6, -10], [-8, 9], [-1, 3]])
        y = np.array([[16], [-6], [18], [15], [-18], [-10], [10], [-23], [13], [8]])

        weights, cost = compute_linear_regression(X, y)
        print(f"Weights shape: {weights.shape}")
        print(f"Weights dtype: {weights.dtype}")
        print(f"Weights values: {weights.flatten()}")
        print(f"Target weights: [3, 1, 2]")

        ref_weights = np.array([3, 1, 2])
        weight_diff = np.sqrt(np.mean((weights.flatten() - ref_weights) ** 2))
        print(f"Weight RMSD: {weight_diff}")

        # Test weight comparison exactly as in test cases
        target = np.array([3, 1, 2])
        test_diff = np.sqrt(np.mean(np.subtract(weights.flatten(), target) ** 2))
        print(f"Test case RMSD calculation: {test_diff}")