# Calories_Burned_Calculator
Project Structure:

.
├── data/
│   ├── exercise.csv          # Exercise metrics (Duration, Heart Rate, Body Temp, etc.)
│   └── calories.csv          # Target calorie consumption values
├── linear_regression.py      # Custom Linear Regression class implemented from scratch
├── main.py                   # Data preprocessing, training, evaluation, & plotting pipeline
├── requirements.txt          # Environment dependencies
└── results.png               # Generated diagnostic dashboard

How It Works (Algorithm Details)
## ⚙️ How It Works (Algorithm Details)

The custom `LinearRegression` engine implements multi-variable Linear Regression using vectorization and Gradient Descent from scratch with NumPy.

### 1. Mathematical Formulation

* **Hypothesis (Prediction Vector)**:
  $$y_{pred} = X \cdot w + b$$
  * $X$: Feature matrix of shape $(m, n)$ where $m$ is samples and $n$ is features
  * $w$: Weights vector of shape $(n,)$
  * $b$: Scalar bias term

* **Cost Function (Mean Squared Error)**:
  Measures the average squared difference between predictions and actual values:
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y_{pred}^{(i)} - y^{(i)})^2$$

* **Gradients Computation**:
  Calculates partial derivatives of the cost function with respect to weights and bias:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T \cdot (y_{pred} - y)$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (y_{pred}^{(i)} - y^{(i)})$$

* **Parameter Update Rule**:
  Updates parameters along the negative gradient direction controlled by learning rate ($\alpha$):
  $$w = w - \alpha \cdot \frac{\partial J}{\partial w}$$
  $$b = b - \alpha \cdot \frac{\partial J}{\partial b}$$

---

### 2. Execution & Data Pipeline Step-by-Step

1. **Data Preprocessing & Encoding**: 
   Merges exercise logs with calorie consumption data on `User_ID`, encodes binary features (e.g., `Gender`), and drops non-predictive identifiers.

2. **Feature Scaling (Z-Score Normalization)**: 
   Standardizes feature matrices to zero mean and unit variance ($X_{norm} = \frac{X - \mu}{\sigma}$) to ensure stable gradient descent convergence across different scale features (such as `Duration` vs `Body_Temp`).

3. **Train-Test Splitting**: 
   Shuffles dataset indices deterministically and partitions data into an **80% training set** and a **20% testing set**.

4. **Model Optimization**: 
   Runs gradient descent iterations, updating $w$ and $b$ while recording cost values at each step to build a convergence history.

5. **Model Evaluation & Benchmarking**: 
   Calculates Mean Squared Error (MSE) and $R^2$ Score on unseen test data, directly validating performance outputs against `scikit-learn`'s `LinearRegression`.


   Getting Started:

# Clone repository
git clone [https://github.com/your-username/calorie-burn-predictor.git](https://github.com/your-username/calorie-burn-predictor.git)
cd calorie-burn-predictor

# Install dependencies
pip install -r requirements.txt

ren the file:
python linear_regression.py

Run the main pipeline:
python main.py


