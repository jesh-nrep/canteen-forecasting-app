import os
import pandas as pd
from pycaret.time_series import load_model
from azure.storage.blob import BlobClient

def load_model_and_data():
    model = load_model("ts_model", "azure", {"container": "models"})
    blob = BlobClient.from_connection_string(conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"], 
                                             container_name="data", 
                                             blob_name="forecast_data")
    blob_data = blob.download_blob()
    data = pd.read_csv(blob_data, parse_dates=['date'], index_col="date")
    return model, data

if __name__ == "__main__":
    load_model_and_data()