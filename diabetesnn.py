import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, log_loss,classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("diabetes.csv")


df.info(), df.head(), df.describe()

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
# Load dataset
df = pd.read_csv("diabetes.csv")

# Define features where zero might indicate missing data
zero_features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Count zeros in each row
df["zero_count"] = (df[zero_features] == 0).sum(axis=1)

# Separate classes
df_0 = df[df["Outcome"] == 0]  # Majority class
df_1 = df[df["Outcome"] == 1]  # Minority class

# Sort by zero count and drop the 230 rows with the most zeros
df_0 = df_0.sort_values(by="zero_count", ascending=False).iloc[232:]

# Drop the temporary zero_count column
df_0 = df_0.drop(columns=["zero_count"])
df_1 = df_1.drop(columns=["zero_count"])

# Merge back the balanced dataset
df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)  # Shuffle the data

# Save or use the balanced dataset
df_balanced.to_csv("balanced_diabetes.csv", index=False)
print("Dataset balanced! New class distribution:")
print(df_balanced["Outcome"].value_counts())

# Load the dataset
df = pd.read_csv("balanced_diabetes.csv")  

# Count occurrences of each outcome
outcome_counts = df["Outcome"].value_counts()

# Plot the bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=outcome_counts.index, y=outcome_counts.values, palette="viridis")

# Labels and title
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.title("Number of Each Outcome in the Dataset")

# Show the values on top of bars
for i, count in enumerate(outcome_counts.values):
    plt.text(i, count + 2, str(count), ha='center', fontsize=12)

plt.show()
# Identify the number of zeros in each of the relevant columns
columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

zero_counts = (df[columns_with_zeros] == 0).sum()
print(zero_counts)
# Replace zeros with the median of each column
for col in columns_with_zeros:
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)
    

# Initialize scaler
scaler = MinMaxScaler()

# Apply MinMax scaling to all features except the target column
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Define features and target
X = df.drop(columns=["Outcome"])  # Features
y = df["Outcome"]  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the neural network architecture
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    layers.Dense(8, activation='relu'),  # Hidden layer
    layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
# Step 1: Predict the class labels and probabilities for the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to class labels
y_pred_prob = model.predict(X_test)  # Predicted probabilities (for Log Loss, ROC-AUC)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")


f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1}")


cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0
print(f"Specificity: {specificity}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
