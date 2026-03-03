# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]


# %%
from skrub.datasets import fetch_midwest_survey

dataset = fetch_midwest_survey()
X = dataset.X
y = dataset.y

# %% [markdown]
# ## Question 1: How many examples are there in the dataset?

# %%
print(" Q1 ")
print("Shape of X (rows, columns):", X.shape)
print("Number of examples:", X.shape[0])
print("Number of features:", X.shape[1])
print()

# %% [markdown]
# ## Question 2

# %%
print(" Q2 ")
target_counts = y.value_counts()
print("Target distribution (counts):\n", target_counts)
print()

target_props = y.value_counts(normalize=True)
print("Target distribution (proportions):\n", target_props)
print()

is_imbalanced = (target_props.max() / target_props.min()) > 2
print("Target looks:", "IMBALANCED" if is_imbalanced else "roughly balanced")
print()

# %% [markdown]
# ## Question 3

# %%
print(" Q3 ")
print("Feature columns:\n", list(X.columns))
print("\nDtypes:\n", X.dtypes)
print()

num_cols = X.select_dtypes(include=["number", "bool"]).columns
cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns
print("Numerical features:", len(num_cols), list(num_cols))
print("Categorical/text features:", len(cat_cols))
print()

# %% [markdown]
# ## Question 4

# %%
print(" Q4 ")
missing = X.isna().sum().sort_values(ascending=False)
print("Missing values per column (top 20):\n", missing.head(20))
print("\nTotal missing values:", int(missing.sum()))
print()

# Optional: look for "implicit missing" values in a couple of columns
for col in ["Household_Income", "Education"]:
    if col in X.columns:
        print(f"Unique values for {col} (first 30):")
        uniques = list(X[col].dropna().unique())
        print(uniques[:30])
        print()

# %% [markdown]
# ## Question 5

# %%
print(" Q5 ")
col_midwest = "How_much_do_you_personally_identify_as_a_Midwesterner"
midwest_counts = X[col_midwest].value_counts()
print(midwest_counts)
print("\nMost common answer:", midwest_counts.idxmax())
print()
