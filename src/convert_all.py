import pandas as pd

# =========================
# File paths
# =========================
input_file = "../data/adult/adult.test"
output_file = "../data/adult/adult_test.names"

# =========================
# Column names (Adult dataset)
# =========================
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

# =========================
# Load raw CSV (comma separated, no header)
# =========================
df = pd.read_csv(
    input_file,
    header=None,
    names=columns,
    skipinitialspace=True   # removes spaces after commas
)

# =========================
# Save clean CSV with headers
# =========================
df.to_csv(output_file, index=False)

print("File with headers created:", output_file)
print("Shape:", df.shape)
print(df.head())