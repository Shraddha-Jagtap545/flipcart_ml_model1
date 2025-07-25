import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Train model
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save it
joblib.dump(clf, 'model.pkl')

