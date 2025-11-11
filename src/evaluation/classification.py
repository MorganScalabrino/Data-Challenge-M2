import pandas as pd

def generate_predictions(model, test_path, preprocessor, feature_engineering, output_path='submission.csv'):
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=['Unnamed: 0', 'row_id'])
    X_test = feature_engineering(X_test)
    X_test = preprocessor.transform(X_test)

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'row_id': test['row_id'],
        'popularity': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    
def generate_predictions_ova(model_ova, model_bin, test_path, preprocessor, feature_engineering, output_path='submission.csv'):
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=['Unnamed: 0', 'row_id'])
    X_test = feature_engineering(X_test)
    X_test = preprocessor.transform(X_test)
    
    y_proba = model_ova.predict_proba(X_test)[:,1]  # positive class probability
    threshold = 0.1  # high confidence only
    y_pred_ova = (y_proba >= threshold).astype(int)
    y_pred_bin = model_bin.predict(X_test)

    predictions = pd.Series(np.zeros(len(X_test.iloc[:,0])))    
    for i in range(len(y_pred_ova)) :
        if y_pred_ova[i]==1 :
            predictions[i]=2
        else :
            predictions[i]=y_pred_bin

    submission = pd.DataFrame({
        'row_id': test['row_id'],
        'popularity': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    

