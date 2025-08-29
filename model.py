import polars as pl
from xgboost import XGBRegressor as XGB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pl.read_parquet('cves_final.parquet').drop(['vendor','product'])
    y = df['has_future_vuln']
    X = df.drop('has_future_vuln')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGB(n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                early_stopping_rounds=20,
                objective='binary:logistic',
                eval_metric='logloss',
                )
    model.fit(X_train,
              y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False,
              )
    
    results = model.evals_result()

    plt.figure(figsize=(10,7))
    plt.plot(results["validation_0"]["logloss"], label="Training loss")
    plt.plot(results["validation_1"]["logloss"], label="Validation loss")
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    
    # # save model
    # import joblib
    # joblib.dump(model, 'risk_model.pkl')
    
    