import mlflow
from xgboost import XGBClassifier

# Objective function for Optuna optimization
def objective(trial, splits, X, y):
    with mlflow.start_run() as run:
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
        cv_scores = []
        for train_index, test_index in splits:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
        
        avg_score = sum(cv_scores) / len(cv_scores)
        
        mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
        mlflow.log_metric('cv_score', avg_score)
        
        return avg_score