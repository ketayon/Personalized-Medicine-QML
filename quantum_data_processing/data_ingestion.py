import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_data(file_paths):
    """Load multiple CSV files into a dictionary of DataFrames."""
    log.info("Loading data from provided file paths...")
    dataframes = {path: pd.read_csv(path) for path in file_paths}
    
    log.info("Data loading completed successfully.")
    return dataframes


def explore_data(dataframes):
    """Explore each DataFrame by displaying information, summary statistics, and missing values."""
    for path, df in dataframes.items():
        log.info("\nExploring dataset: %s", path)
        log.info("\nDataset Info:\n%s", df.info())
        log.info("\nDataset Summary:\n%s", df.describe())
        log.info("\nMissing values:\n%s", df.isnull().sum())


def encode_labels(dataframes):
    """Encode categorical features using Label Encoding."""
    label_encoder = LabelEncoder()
    
    for path, df in dataframes.items():
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = label_encoder.fit_transform(df[col])
        dataframes[path] = df
    
    log.info("Label encoding completed for categorical columns.")
    return dataframes


def normalize_data(dataframes, method="standard"):
    """Normalize numerical features using StandardScaler or MinMaxScaler."""
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    
    for path, df in dataframes.items():
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        
        if len(numeric_cols) > 0:  # Prevent errors on empty datasets
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            dataframes[path] = df

    log.info("Data normalization completed using %s scaling.", method)
    return dataframes


def filter_data(dataframes, variance_threshold=0.01):
    """Filter features based on variance threshold."""
    for path, df in dataframes.items():
        variance = df.var()
        selected_features = variance[variance > variance_threshold].index
        dataframes[path] = df[selected_features]

    log.info("Feature selection completed based on variance threshold: %f", variance_threshold)
    return dataframes


def merge_data(dataframes, join_on=None):
    """Merge multiple DataFrames either on a specific column or by concatenation."""
    if join_on and join_on in dataframes.values():
        merged_df = pd.merge(*dataframes.values(), on=join_on, how='inner')
        log.info("Data merged on column: %s", join_on)
    else:
        merged_df = pd.concat(dataframes.values(), axis=1, join="inner", ignore_index=True)
        log.info("Data merged using inner join on index.")

    return merged_df


def add_patient_id_column(csv_file, output_file="updated_file.csv"):
    """
    Adds a 'PatientId' column to a CSV file, starting from '0000001' and incrementing for each row.

    Parameters:
    - csv_file (str): The path to the input CSV file.
    - output_file (str): The path to save the updated CSV file (default: 'updated_file.csv').

    Returns:
    - None (saves the updated CSV file)
    """
    log.info("Adding 'PatientId' column to the dataset: %s", csv_file)

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Add a new column 'PatientId' with incremental values, formatted as 7-digit numbers
    df["PatientId"] = [f"{i:07d}" for i in range(1, len(df) + 1)]

    # Save back to CSV
    df.to_csv(output_file, index=False)

    log.info("Column 'PatientId' added successfully! Updated file saved as '%s'", output_file)
