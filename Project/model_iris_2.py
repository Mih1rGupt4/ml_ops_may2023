
import mlflow
logged_model = 'runs:/bcb3eb4ab4b84003a277c6acbe4e5e35/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(X_test))