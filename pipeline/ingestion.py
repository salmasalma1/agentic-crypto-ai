"""Step 1: automated data ingestion from Streamlit upload or default files."""

import os
import pandas as pd
from utils.config import DEFAULT_CSV_PATH, DEFAULT_EXCEL_PATH


def read_csv_safely(file_path_or_buffer):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(file_path_or_buffer, encoding=encoding)
        except Exception as e:
            last_error = e
    raise ValueError(f"Failed to read CSV file. Last error: {last_error}")


def read_dataset(file_path_or_buffer, file_name: str):
    name = str(file_name).lower()
    if name.endswith(".csv"):
        return read_csv_safely(file_path_or_buffer)
    if name.endswith(".xlsx"):
        return pd.read_excel(file_path_or_buffer, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(file_path_or_buffer)
    raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")


def load_uploaded_data(uploaded_file):
    if uploaded_file is None:
        return None
    return read_dataset(uploaded_file, uploaded_file.name)


def load_default_dataset():
    if os.path.exists(DEFAULT_CSV_PATH):
        return read_dataset(DEFAULT_CSV_PATH, DEFAULT_CSV_PATH)
    if os.path.exists(DEFAULT_EXCEL_PATH):
        return read_dataset(DEFAULT_EXCEL_PATH, DEFAULT_EXCEL_PATH)
    raise FileNotFoundError("No dataset found. Place dataset.csv or dataset.xlsx inside data/raw/.")


def load_data(uploaded_file=None):
    return load_uploaded_data(uploaded_file) if uploaded_file is not None else load_default_dataset()
