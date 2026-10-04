# ============================================================
# MINI PROJECT: CUSTOMER CHURN ANALYSIS
# Beginner-friendly Python project
# Tools: NumPy, Pandas, Matplotlib, Seaborn
# ============================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: make charts look cleaner
sns.set_theme(style="whitegrid")

# STEP 2: LOAD DATASET
# Put Customer_Churn_Dataset.csv in the same folder as this Python file.
df = pd.read_csv("Customer_Churn_Dataset.csv")

# STEP 3: FIRST FIVE RECORDS
print("\nFIRST 5 RECORDS")
print(df.head())

# STEP 4: LAST FIVE RECORDS
print("\nLAST 5 RECORDS")
print(df.tail())

# STEP 5: DATASET SHAPE
print("\nDATASET SHAPE")
print(df.shape)

# STEP 6: COLUMN NAMES
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# ============================================================
# PART 2 - DATA EXPLORATION & CLEANING
# ============================================================

# 7. DATA TYPES
print("\nDATA TYPES")
print(df.dtypes)

# 8. STATISTICAL INFORMATION
print("\nSTATISTICAL INFORMATION")
print(df.describe(include="all").T)

# 9. MISSING VALUES
print("\nMISSING VALUES")
print(df.isnull().sum())

# 10. DUPLICATE RECORDS
print("\nNUMBER OF DUPLICATES")
print(df.duplicated().sum())

# 11. REMOVE DUPLICATES
df = df.drop_duplicates()

# 12. UNIQUE VALUES
print("\nUNIQUE VALUES")
for col in df.columns:
    print(col, ":", df[col].nunique())

# 13. HANDLE MISSING VALUES
# Total_Charges is numeric, so use median.
df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors="coerce")
df["Total_Charges"] = df["Total_Charges"].fillna(df["Total_Charges"].median())

# 14. CONVERT REQUIRED COLUMNS
numeric_columns = [
    "Senior_Citizen",
    "Tenure_Months",
    "Monthly_Charges",
    "Total_Charges"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove any rows that still have missing values in essential fields
df = df.dropna(subset=["Customer_ID", "Tenure_Months",
                       "Monthly_Charges", "Total_Charges", "Churn"])

# Convert Senior_Citizen to readable labels for analysis
df["Senior_Citizen_Label"] = df["Senior_Citizen"].map({
    0: "No",
    1: "Yes"
})

# 15. CREATE TENURE GROUPS
bins = [0, 12, 24, 48, 72]
labels = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]

df["Tenure_Group"] = pd.cut(
    df["Tenure_Months"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# 16. VERIFY CLEANED DATASET
print("\nCLEANED DATASET INFO")
print(df.info())

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

print("\nCLEANED DATA SAMPLE")
print(df.head())

# Save cleaned CSV for Power BI
df.to_csv("Customer_Churn_Cleaned.csv", index=False)

# ============================================================
# PART 3 - DATA ANALYSIS
# ============================================================

# 17. TOTAL CUSTOMERS
total_customers = df["Customer_ID"].nunique()

# 18. TOTAL CHURNED CUSTOMERS
churned_customers = (df["Churn"] == "Yes").sum()

# 19. TOTAL RETAINED CUSTOMERS
retained_customers = (df["Churn"] == "No").sum()

# 20. OVERALL CHURN RATE
churn_rate = (churned_customers / total_customers) * 100

# 21. AVERAGE MONTHLY CHARGES
avg_monthly_charges = df["Monthly_Charges"].mean()

# 22. AVERAGE TOTAL CHARGES
avg_total_charges = df["Total_Charges"].mean()

# 23. AVERAGE CUSTOMER TENURE
avg_tenure = df["Tenure_Months"].mean()

print("\n========== KEY KPIs ==========")
print("Total Customers:", total_customers)
print("Churned Customers:", churned_customers)
print("Retained Customers:", retained_customers)
print("Churn Rate: {:.2f}%".format(churn_rate))
print("Average Monthly Charges: {:.2f}".format(avg_monthly_charges))
print("Average Total Charges: {:.2f}".format(avg_total_charges))
print("Average Tenure: {:.2f} months".format(avg_tenure))

# 24. CUSTOMERS BY GENDER
print("\nCUSTOMERS BY GENDER")
print(df["Gender"].value_counts())

# 25. CUSTOMERS BY CONTRACT
print("\nCUSTOMERS BY CONTRACT")
print(df["Contract"].value_counts())

# 26. CUSTOMERS BY INTERNET SERVICE
print("\nCUSTOMERS BY INTERNET SERVICE")
print(df["Internet_Service"].value_counts())

# 27. CUSTOMERS BY PAYMENT METHOD
print("\nCUSTOMERS BY PAYMENT METHOD")
print(df["Payment_Method"].value_counts())

# Helper function for churn rate tables
def churn_rate_by(column):
    result = df.groupby(column)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).sort_values(ascending=False)
    return result.round(2)

# 28. CHURN RATE BY GENDER
print("\nCHURN RATE BY GENDER")
print(churn_rate_by("Gender"))

# 29. CHURN RATE BY CONTRACT
print("\nCHURN RATE BY CONTRACT")
print(churn_rate_by("Contract"))

# 30. CHURN RATE BY INTERNET SERVICE
print("\nCHURN RATE BY INTERNET SERVICE")
print(churn_rate_by("Internet_Service"))

# 31. CHURN RATE BY PAYMENT METHOD
print("\nCHURN RATE BY PAYMENT METHOD")
print(churn_rate_by("Payment_Method"))

# 32. CHURN RATE BY SENIOR CITIZEN STATUS
print("\nCHURN RATE BY SENIOR CITIZEN STATUS")
print(churn_rate_by("Senior_Citizen_Label"))

# 33. CHURN RATE BY TENURE GROUP
print("\nCHURN RATE BY TENURE GROUP")
print(churn_rate_by("Tenure_Group"))

# 34. MONTHLY CHARGES: CHURNED VS RETAINED
print("\nMONTHLY CHARGES BY CHURN")
print(df.groupby("Churn")["Monthly_Charges"].mean().round(2))

# 35. TENURE: CHURNED VS RETAINED
print("\nTENURE BY CHURN")
print(df.groupby("Churn")["Tenure_Months"].mean().round(2))

# 36. TOP 10 CUSTOMERS BY TOTAL CHARGES
print("\nTOP 10 CUSTOMERS BY TOTAL CHARGES")
top_10 = df.nlargest(10, "Total_Charges")[
    ["Customer_ID", "Contract", "Tenure_Months",
     "Monthly_Charges", "Total_Charges", "Churn"]
]
print(top_10)

# 37. CUSTOMER SEGMENTS WITH HIGHER CHURN
print("\nCUSTOMER SEGMENTS WITH HIGHER CHURN")

segment_tables = {
    "Contract": churn_rate_by("Contract"),
    "Internet Service": churn_rate_by("Internet_Service"),
    "Payment Method": churn_rate_by("Payment_Method"),
    "Tenure Group": churn_rate_by("Tenure_Group"),
    "Gender": churn_rate_by("Gender")
}

for name, table in segment_tables.items():
    print("\n", name)
    print(table)

# 38. MAJOR FACTORS ASSOCIATED WITH CHURN
print("\nMAJOR FACTORS ASSOCIATED WITH CHURN")

# Convert churn to numeric for correlation
df["Churn_Flag"] = df["Churn"].map({"No": 0, "Yes": 1})

numeric_for_corr = [
    "Senior_Citizen",
    "Tenure_Months",
    "Monthly_Charges",
    "Total_Charges",
    "Churn_Flag"
]

correlation = df[numeric_for_corr].corr()["Churn_Flag"].sort_values(ascending=False)
print(correlation)

# ============================================================
# PART 4 - PYTHON VISUALIZATION
# ============================================================

# 1. CHURN DISTRIBUTION - PIE CHART
plt.figure(figsize=(7, 7))
df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Churn Distribution")
plt.ylabel("")
plt.show()

# 2. CUSTOMER DISTRIBUTION BY CONTRACT - BAR CHART
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract")
plt.title("Customer Distribution by Contract")
plt.xlabel("Contract")
plt.ylabel("Customer Count")
plt.xticks(rotation=15)
plt.show()

# 3. CHURN BY CONTRACT - BAR CHART
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract")
plt.xlabel("Contract")
plt.ylabel("Customer Count")
plt.xticks(rotation=15)
plt.show()

# 4. CHURN BY GENDER - BAR CHART
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender", hue="Churn")
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Customer Count")
plt.show()

# 5. CHURN BY INTERNET SERVICE - BAR CHART
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Internet_Service", hue="Churn")
plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Customer Count")
plt.show()

# 6. CHURN BY PAYMENT METHOD - BAR CHART
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Payment_Method", hue="Churn")
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Customer Count")
plt.xticks(rotation=20)
plt.show()

# 7. TENURE DISTRIBUTION - HISTOGRAM
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Tenure_Months", bins=20, kde=True)
plt.title("Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Customer Count")
plt.show()

# 8. MONTHLY CHARGES DISTRIBUTION - HISTOGRAM
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Monthly_Charges", bins=25, kde=True)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Customer Count")
plt.show()

# 9. TENURE VS MONTHLY CHARGES - SCATTER PLOT
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Tenure_Months",
    y="Monthly_Charges",
    hue="Churn",
    alpha=0.6
)
plt.title("Tenure vs Monthly Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.show()

# 10. MONTHLY CHARGES BY CHURN - BOX PLOT
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="Monthly_Charges")
plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()

# 11. TOTAL CHARGES BY CHURN - BOX PLOT
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="Total_Charges")
plt.title("Total Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Total Charges")
plt.show()

# 12. CHURN RATE BY TENURE GROUP - BAR CHART
tenure_churn = churn_rate_by("Tenure_Group").reindex(labels)

plt.figure(figsize=(9, 5))
sns.barplot(x=tenure_churn.index, y=tenure_churn.values)
plt.title("Churn Rate by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)
plt.show()

# 13. SERVICE USAGE - COUNT PLOT
service_columns = [
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Streaming_Movies"
]

service_long = df[service_columns].melt(
    var_name="Service",
    value_name="Usage"
)

plt.figure(figsize=(12, 6))
sns.countplot(data=service_long, x="Service", hue="Usage")
plt.title("Service Usage")
plt.xlabel("Service")
plt.ylabel("Customer Count")
plt.xticks(rotation=25)
plt.show()

# 14. CORRELATION HEATMAP
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_for_corr].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# EXTRA: AUTOMATIC BUSINESS INSIGHT HELPERS
# ============================================================

contract_churn = churn_rate_by("Contract")
payment_churn = churn_rate_by("Payment_Method")
internet_churn = churn_rate_by("Internet_Service")
tenure_churn = churn_rate_by("Tenure_Group")

print("\n========== BUSINESS INSIGHT SUMMARY ==========")
print("Highest contract churn rate:",
      contract_churn.index[0], "-", contract_churn.iloc[0], "%")

print("Highest payment-method churn rate:",
      payment_churn.index[0], "-", payment_churn.iloc[0], "%")

print("Highest internet-service churn rate:",
      internet_churn.index[0], "-", internet_churn.iloc[0], "%")

print("Highest tenure-group churn rate:",
      tenure_churn.index[0], "-", tenure_churn.iloc[0], "%")

print("\nAverage monthly charges:")
print(df.groupby("Churn")["Monthly_Charges"].mean().round(2))

print("\nAverage tenure:")
print(df.groupby("Churn")["Tenure_Months"].mean().round(2))

print("\nPROJECT COMPLETED")
print("Cleaned dataset saved as: Customer_Churn_Cleaned.csv")
