import numpy as np

class LinearRegression:
    """Linear Regression from scratch using Gradient Descent."""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        m, n = X.shape
        
        # 1. Initialize self.weights to a zero vector of length n
        self.weights = np.zeros(n)
        
        # 2. Initialize self.bias to 0.0
        self.bias = 0.0

        for i in range(self.n_iterations):
            # 3. y_pred = X dot weights + bias (Use your self.predict method)
            y_pred = self.predict(X)
            
            # 4. error = y_pred - y
            error = y_pred - y
            
            # 5. dw = (1/m) * X transposed dot error
            dw = X.T.dot(error) / m
            
            # 6. db = (1/m) * sum of error
            db = np.sum(error) / m
            
            # 7. update self.weights = self.weights - learning_rate * dw
            self.weights = self.weights - self.learning_rate * dw
            
            # 8. update self.bias = self.bias - learning_rate * db
            self.bias = self.bias - self.learning_rate * db
            
            # Tracks the cost history
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
        return self

    def predict(self, X):
        # 9. return X dot weights + bias
        return X.dot(self.weights) + self.bias

    def compute_cost(self, X, y):
        m = X.shape[0]
        y_pred = self.predict(X)
        # 10. return Andrew Ng's average squared error cost: (1 / (2*m)) * sum of squared errors
        return np.sum((y_pred - y) ** 2) / (2 * m)

    def mse(self, X, y):
        m = X.shape[0]
        y_pred = self.predict(X)
        return np.sum((y_pred - y) ** 2) / m

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# === QUICK TEST — run: python linear_regression.py ===
if __name__ == "__main__":
    np.random.seed(42)
    # Generate simple linear dummy data
    X_test = 2 * np.random.rand(100, 1)
    y_test = 4 + 3 * X_test.squeeze() + np.random.randn(100) * 0.5
    
    # Train the model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_test, y_test)
    
    # Output metrics
    print(f"Weights: {model.weights}")        # expect close to [3.0]
    print(f"Bias: {model.bias:.2f}")           # expect close to 4.0
    print(f"R²: {model.r2_score(X_test, y_test):.4f}")  # expect > 0.95
    print("Cost went DOWN ✅" if model.cost_history[-1] < model.cost_history[0] else "Cost went UP ❌")