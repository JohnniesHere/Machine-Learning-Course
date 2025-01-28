# Linear Regression Implementation

## Overview

This project provides a comprehensive implementation of linear regression using gradient descent, with robust error handling and data preprocessing capabilities.

## Features

### Key Components
- **Gradient Descent Algorithm**: Implements linear regression optimization
- **Data Standardization**: Scales features and target variables
- **Adaptive Learning**: Supports multiple learning rates
- **Error Prevention**: Includes overflow and underflow protections

### Main Functions

#### `compute_linear_regression(X, y, alpha=0.01, max_iterations=1000)`
- Primary function for linear regression
- Handles both single and multiple feature regression
- Returns optimized weights and cost

#### `standardize_data(X, y)`
- Preprocesses input data
- Scales features and target
- Prevents division by zero
- Allows for consistent learning across different scales

#### `compute_cost(X, y, weights)`
- Calculates Mean Squared Error (MSE)
- Includes error clipping to prevent numerical instability

#### `gradient(X, y, weights)`
- Computes gradient for weight updates
- Prevents gradient explosion

## Test Suite

The accompanying test suite (`linear_regression_tests.py`) provides comprehensive validation:

### Test Categories
- **Basic Functionality Tests**:
  - Zero initialization
  - Single feature regression
  - Different learning rates
  - Iteration count sensitivity
  - Extreme and negative value handling
  - Random data testing
  - Perfect fit scenarios

- **Error Handling Tests**:
  - Empty array detection
  - Dimension mismatch validation

- **Graded Test Cases**:
  - Noisy data scenario
  - Simple regression scenario

## Requirements
- NumPy
- Python 3.7+

## Usage Example

```python
import numpy as np
from linear_regression import compute_linear_regression

# Prepare your data
X = np.array([[1, 1], [2, 2], [3, 3]])
y = np.array([[3], [6], [9]])

# Compute regression
weights, cost = compute_linear_regression(X, y)
print("Optimized Weights:", weights)
print("Cost:", cost)
```

## Key Considerations
- Handles various input scenarios
- Provides multiple learning rate options
- Implements data standardization
- Robust error handling
- Prevents numerical instability

## Limitations
- Assumes linear relationship between features and target
- Performance may vary with different datasets
- Sensitive to learning rate selection

