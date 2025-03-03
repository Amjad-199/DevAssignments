import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("M1_final.csv")

print(df.head())
numeric_df = df.select_dtypes(include=["number"])

correlation_matrix = numeric_df.corr()[["TAXI_OUT"]].sort_values(by="TAXI_OUT", ascending=False)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Matrix for TAXI_OUT")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['TAXI_OUT'], kde=True, color='skyblue')
plt.axvline(df['TAXI_OUT'].mean(), color='red', linestyle='--', label=f'Mean: {df["TAXI_OUT"].mean():.2f}')
plt.title('Distribution of Taxi Out Time with Mean')
plt.xlabel('Taxi Out Time')
plt.ylabel('Frequency')
plt.legend()
plt.show()
X = df[['MONTH','DEP_DELAY','DAY_OF_WEEK','DAY_OF_MONTH','CRS_ELAPSED_TIME','DISTANCE','DEP_TIME_M','CRS_ARR_M','Temperature','Dew Point','Humidity','Wind Speed','Wind Gust','Pressure','sch_dep','sch_arr']]
y = df['TAXI_OUT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))
kf = KFold(n_splits=30, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")

model_lasso = Lasso(alpha=1, random_state= 32)
model_lasso.fit(X_train, y_train)
pred_lasso = model_lasso.predict(X_test)
pd.DataFrame({'Actual': y_test, 'Predicted': pred_lasso})

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=pred_lasso, alpha=0.6, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel("Actual Taxi Out Time")
plt.ylabel("Predicted Taxi Out Time")
plt.title("Lasso Regression: Actual vs Predicted")

plt.show()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))

# Scatter plot for actual vs predicted values
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="blue", label="Predictions")

# Plotting the perfect prediction line (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Fit")

plt.xlabel("Actual Taxi Out Time")
plt.ylabel("Predicted Taxi Out Time")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()