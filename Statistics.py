import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
# Generate Synthetic Data
def generate_synthetic_data(sample_size):
    X = np.random.rand(sample_size, 1)
    y = 2 * X.squeeze() + np.random.normal(scale=0.5, size=sample_size)
    return X, y

# ------------------------------------------------------------------------------------------
# Train and Evaluate Models
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    models = {'Linear Regression': model_lr, 'Random Forest': model_rf}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
    
    return results

# ------------------------------------------------------------------------------------------
# Evaluate Performance Metrics with Cross-Validation
def evaluate_performance_with_cv(X, y, model):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    return -scores.mean()

# ------------------------------------------------------------------------------------------
# Visualize Results
sample_sizes = [50, 100, 200, 500, 1000]
results_lr = []
results_rf = []

for sample_size in sample_sizes:
    X, y = generate_synthetic_data(sample_size)
    
    mse_results = train_and_evaluate_models(X, y)
    results_lr.append(mse_results['Linear Regression'])
    results_rf.append(mse_results['Random Forest'])

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, results_lr, marker='o', label='Linear Regression')
plt.plot(sample_sizes, results_rf, marker='o', label='Random Forest')
plt.xlabel('Sample Size')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Impact of Sample Size on Model Performance')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------
# Additional Analysis
best_sample_size = 200
X_best, y_best = generate_synthetic_data(best_sample_size)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
best_cv_score = evaluate_performance_with_cv(X_best, y_best, model_rf)
print(f'Cross-validated MSE for Random Forest with sample size {best_sample_size}: {best_cv_score}')