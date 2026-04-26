import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv('house_price_regression_dataset.csv')

# 2. Separate Features and Target
X = df.drop('House_Price', axis=1)
y = df['House_Price']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate (Optional console output)
predictions = model.predict(X_test_scaled)
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, predictions):,.2f}")

# 7. Export Model and Scaler
pickle.dump(model, open('house_model.pkl', 'wb'))
pickle.dump(scaler, open('house_scaler.pkl', 'wb'))

print("Files 'house_model.pkl' and 'house_scaler.pkl' exported successfully!")