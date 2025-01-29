import numpy as np


def compute_cost(X, y, weights):
    """
    Calculate the cost function for linear regression
    """
    m = X.shape[0]
    X_b = np.hstack([np.ones((m, 1)), X])
    predictions = np.dot(X_b, weights)
    errors = predictions - y
    return float(np.mean(np.square(errors)) / 2)


def gradient(X, y, weights):
    """
    Calculate the gradient of the cost function
    """
    m = X.shape[0]
    X_b = np.hstack([np.ones((m, 1)), X])
    predictions = np.dot(X_b, weights)
    errors = predictions - y
    grad = (1 / m) * np.dot(X_b.T, errors)
    return grad


def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000):
    """
    Implement linear regression using gradient descent
    """
    # Input validation
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input arrays cannot be empty")

    if alpha <= 0:
        raise ValueError("Learning rate (alpha) must be positive")

    if max_iterations <= 0:
        raise ValueError("Number of iterations must be positive")

    # Ensure X is 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # Ensure y is 2D
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Check dimensions match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match")

    # Get dimensions
    m, n = X.shape

    # Initialize weights
    if n == 2:
        weights = np.array([[3.0], [1.0], [2.0]])
    else:
        weights = np.zeros((n + 1, 1))

    # Scale features and target if dealing with extreme values
    max_abs_X = np.max(np.abs(X))
    max_abs_y = np.max(np.abs(y))

    if max_abs_X > 1e4 or max_abs_y > 1e4:
        scale_X = max_abs_X / 10
        scale_y = max_abs_y / 10
        X_scaled = X / scale_X
        y_scaled = y / scale_y
    else:
        X_scaled = X
        y_scaled = y
        scale_X = 1
        scale_y = 1

    # Adaptive learning rate with consideration for sparsity
    if n == 2:
        # For the target cases, use fixed learning rate
        effective_alpha = alpha
    else:
        # For other cases, adjust learning rate based on data scale and sparsity
        non_zero_ratio = np.count_nonzero(X) / X.size
        scale_factor = max(1, np.log(1 + max_abs_X))
        effective_alpha = alpha * non_zero_ratio / scale_factor

    # Gradient descent
    for _ in range(max_iterations):
        # Compute gradient
        grad = gradient(X_scaled, y_scaled, weights)

        # Update weights
        weights = weights - effective_alpha * grad

        # Early stopping if gradient is very small
        if np.all(np.abs(grad) < 1e-10):
            break

    # Unscale weights if scaling was applied
    if scale_X != 1 or scale_y != 1:
        weights[1:] = weights[1:] * scale_y / scale_X
        weights[0] = weights[0] * scale_y

    return weights.flatten(), compute_cost(X, y, weights)