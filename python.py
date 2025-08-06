import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os


print(os.path.exists("Titanic-Dataset.csv"))


df = pd.read_csv(
    "C:\\Users\\khann\\Desktop\\New folder\\DAY 1\\Titanic-Dataset.csv")

print("Summary Statistics:\n", df.describe())

numeric_cols = ['Age', 'Fare']
for col in numeric_cols:

    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    plt.title(f'{col} Boxplot')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
corr_matrix = df[['Survived', 'Pclass',
                  'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()

fig = px.histogram(df, x="Age", color="Survived",
                   nbins=30, title="Age vs Survival")
fig.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()
