import os
import pandas as pd


def read_uploaded_file(file_path):
    # Get the file extension and convert to lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Dictionary mapping extensions to their respective pandas read functions
    readers = {
        ".csv": lambda f: pd.read_csv(f),
        ".txt": lambda f: pd.read_csv(f, sep=None, engine="python"),
        ".xls": lambda f: pd.read_excel(f),
        ".xlsx": lambda f: pd.read_excel(f),
        ".xlsm": lambda f: pd.read_excel(f),
        ".json": lambda f: pd.read_json(f),
        ".parquet": lambda f: pd.read_parquet(f),
        ".feather": lambda f: pd.read_feather(f),
        ".pickle": lambda f: pd.read_pickle(f),
        ".pkl": lambda f: pd.read_pickle(f),
        ".h5": lambda f: pd.read_hdf(f),
        ".hdf5": lambda f: pd.read_hdf(f),
    }

    if file_extension in readers:
        print(f"Reading file with extension: {file_extension}")
        return readers[file_extension](file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{file_extension}' provided."
        )


# --- Example Usage ---
# df = read_uploaded_file("path/to/your/file.csv")
# print(df.head())
