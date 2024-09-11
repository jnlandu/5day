from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X, y)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# TODO: Save logistic regression model to disk.

# TODO: Save random forest model to disk.

import joblib
from pathlib import Path
save_dir = Path('./models')

# TODO: Save logistic regression model to disk.
# Save models to disk
logreg_filename = save_dir/ 'logreg.pkl'
rf_filename = save_dir/ 'rf.pkl'
with open(logreg_filename,  "wb") as f:
    joblib.dump(logreg_model, logreg_filename )

with open(rf_filename, "wb") as f:
   joblib.dump(rf_model, rf_filename)



print('Successfull!')