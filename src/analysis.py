import pandas as pd

# Chargement
df = pd.read_csv("../data/adult/adult_train.csv")

# Aperçu
print(df.head())
print("\nShape:", df.shape)
print("\nTypes de colonnes:\n", df.dtypes)

# Variables numériques
print(df.describe())

# Variables catégorielles
categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
for col in categorical_cols:
    print(f"\nValeurs uniques pour {col}:")
    print(df[col].value_counts())
