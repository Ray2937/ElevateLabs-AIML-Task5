import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Load and preprocess the dataset
print('--- 1. LOADING & PREPROCESSING DATA ---')
df = pd.read_csv('heart.csv')

# Features (X) and Target (y) - 'target' indicates presence of heart disease (1 or 0)
X = df.drop(columns=['target'])
y = df['target']

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train a Decision Tree Classifier and visualize the tree structure
print('\n--- 2. TRAINING DECISION TREE ---')
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
print(f'Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}')

# Plot and save Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['No Disease', 'Heart Disease'],
    filled=True,
    rounded=True,
)
plt.title('Decision Tree Visualization (Max Depth = 4)')
plt.savefig('decision_tree.png', bbox_inches='tight')
plt.close()

# 3. Train a Random Forest Classifier and compare accuracy
print('\n--- 3. TRAINING RANDOM FOREST ---')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}')

# 4. Interpret Feature Importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print('\nTop 5 Most Important Features (Random Forest):')
print(feature_importances.nlargest(5))

# Plot and save feature importances
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh', color='crimson')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.savefig('feature_importances.png', bbox_inches='tight')
plt.close()

# 5. Evaluate using Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(
    f'\nRandom Forest 5-Fold CV Accuracy: {cv_scores.mean():.4f}'
    f' (±{cv_scores.std():.4f})'
)

print(
    '\nTask 5 code executed successfully! Visualizations saved as'
    ' decision_tree.png and feature_importances.png'
)