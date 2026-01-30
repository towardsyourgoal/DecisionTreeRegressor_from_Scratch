**A simple implementation of a Decision Tree Regressor from scratch.**
# Decision Tree Regressor (Python)
Decision Tree Regressor Implementation from Scratch

## Description
This repository contains code that describes how Decision Tree Regression takes place step by step by using not any publicly availble packages like scikit-learn. 
We only use numpy for the implementation.

## Requirements

*   **Python 3.x +**
*   **NumPy:** `pip install numpy`

## Installation

1.  Clone the repository or download the code.
2.  Save the code as a `.py` file (e.g., `decision_tree_regressor.py`).
3.  Ensure you have NumPy installed: `pip install numpy`

## Usage

1.  **Import the class:**
    ```python
    from decision_tree_regressor import DecisionTreeRegressor
    ```

2.  **Create an instance:**
    ```python
    tree_reg = DecisionTreeRegressor(max_depth=float("inf"), min_samples_split=2) # You can adjust max_depth
    ```

3.  **Fit the model:**
    ```python
    X = np.array([[1], [2], [3], [4], [5]])  # Input features
    y = np.array([2, 4, 5, 4, 5])        # Target values
    tree_reg.fit(X, y)
    ```

4.  **Make predictions:**
    ```python
    predictions = tree_reg.predict(X)
    print(predictions)
    ```

5.  **Various evaluation metrics**
    - mean_squared_error
    - rmse
    - mean_absolute_error
    - r_squared (includes adjusted r_squared)
    - mbe
    - mape
    - smape
    - explained_var_score

   

## Key Features

*   **Recursive Tree Building:**  The `_build_tree` method recursively creates the decision tree structure.
*   **Best Split Finding:**  The `_find_best_split` method selects the best feature and value for splitting the data.
*   **Prediction:**  The `predict` method traverses the tree to make predictions for new input data.
*   **Maximum Depth:** Limits the tree's complexity to prevent overfitting.

## Limitations

*   **No Pruning:** The implementation does not include any pruning techniques.
*   **Basic Splitting:** Uses variance reduction as the sole splitting criterion.
*   **Simple Implementation:**  This is a simplified implementation and may not be suitable for complex datasets.

## Contributing

Contributions are welcome! Please submit pull requests with well-documented code changes.


