import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

train_data = pd.read_csv("train.csv")

train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})

df_cleaned = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

plt.figure(figsize=(5,5))
ax = sns.countplot(x='Survived', data=train_data, palette='Set2')


for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',   
                (p.get_x() + p.get_width() / 2., p.get_height()),  
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')


plt.title('Survival Count', fontsize=14, fontweight='bold')
plt.xlabel('Survival Status', fontsize=12)
plt.ylabel('Count', fontsize=12)

plt.xticks(ticks=[0, 1], labels=["No", "Yes"])

plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(x='Survived', hue='Sex', data=train_data, palette='Set1')
plt.title('Survival Count by Gender')


ax.set_xticklabels(['No', 'Yes'])


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Female', 'Male'], title='Gender')

for p in ax.patches:
    if p.get_height() > 0: 
        ax.annotate(f'{int(p.get_height())}',  
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.show()

plt.figure(figsize=(8,6))
ax = sns.countplot(x='Pclass', hue='Survived', data=train_data, palette='Set2')


for p in ax.patches:
    if p.get_height() > 0:  
        ax.annotate(f'{int(p.get_height())}',  
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])  
plt.title('Passenger Class vs Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

X = train_data.iloc[:, [0,2,4 ,5, 6, 7, 9]]  
y = train_data.iloc[:, 1]  



train_data = pd.read_csv("train.csv")

train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
X = train_data[features]
y = train_data['Survived']


X = X.fillna(X.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)


undersample = RandomUnderSampler(random_state=32)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)


print("Class distribution after undersampling:\n", y_train_under.value_counts())

sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)


data = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': y_pred_log})

accuracy = accuracy_score(y_test, y_pred_log)
conf_matrix = confusion_matrix(y_test, y_pred_log)
class_report = classification_report(y_test, y_pred_log)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)