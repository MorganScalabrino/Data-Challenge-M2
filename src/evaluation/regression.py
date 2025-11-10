import pandas as pd

def generate_predictions(model, test_path, preprocessor, output_path='submission.csv'):
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=['Unnamed: 0', 'row_id'])
    X_test = preprocessor.transform(X_test)

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'row_id': test['row_id'],
        'popularity': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
