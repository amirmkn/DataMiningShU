from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import scipy.stats as statis

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
num_df = pd.DataFrame(iris.data, columns=iris.feature_names)


label_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
df['label'] = [label_map[label] for label in iris.target]

results=[]

header = ['label', 'missing','min','q1','med','q3','p95','max','mean','range','iqr','std','std_pop','mad']

for label, group in df.groupby('label'):
    stats={}
    stats = {'label': label}


    column_data= group['sepal width (cm)'].dropna()

    stats['missing'] = group['sepal width (cm)'].isna().sum()
    stats['min'] = column_data.min()
    stats['q1'] = column_data.quantile(0.25)
    stats['med'] = column_data.median()
    stats['q3'] = column_data.quantile(0.75)
    stats['p95'] = column_data.quantile(0.95)
    stats['max'] = column_data.max()
    stats['mean'] = column_data.mean()
    stats['range'] = stats['max'] - stats['min']
    stats['iqr'] = stats['q3'] - stats['q1']
    stats['std'] = column_data.std(ddof=1)  # Sample standard deviation
    stats['std_pop'] = column_data.std(ddof=0)  # Population standard deviation
    stats['mad'] = np.abs(column_data - stats['mean']).mean()

    results.append(stats)

results_df = pd.DataFrame(results)

# Get the parent directory (one level up from the current directory)
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))


# Define the 'dist' folder path in the parent directory
dist_dir = os.path.join(parent_dir, 'dist')

print("Parent directory:", parent_dir)
print("Dist directory:", dist_dir)

# Create 'dist' folder if it doesn't exist
if not os.path.exists(dist_dir):
    os.makedirs(dist_dir)

results_df.to_csv(os.path.join(dist_dir, 'statistics.csv'), index=False, columns=header)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<CORRELATION PART>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Compute the Pearson correlation matrix
correlation_matrix = num_df.corr(method='pearson')

# Save the correlation matrix to a CSV file without headers or indices
correlation_matrix.to_csv(os.path.join(dist_dir, 'correlations.csv'), header=False, index=False)

# Find the minimum and maximum absolute correlations, excluding self-correlations
# Convert correlation matrix to a stacked format and filter out self-correlations
correlation_values = correlation_matrix.where(~correlation_matrix.eq(1)).stack()

# Identify the pairs with minimum and maximum absolute correlation
min_corr_pair = correlation_values.idxmin()
max_corr_pair = correlation_values.idxmax()
min_corr_value = correlation_values[min_corr_pair]
max_corr_value = correlation_values[max_corr_pair]

# Display the pairs and their correlations
print(f"Minimum absolute correlation: {min_corr_pair} with correlation {min_corr_value}")
print(f"Maximum absolute correlation: {max_corr_pair} with correlation {max_corr_value}")

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<PLOTS PART>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# df['species'] = iris.target_names[iris.target]

# Bar plot to show distribution of species labels
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('Distribution of Species Labels')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Separate histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['petal length (cm)'], kde=True, bins=15)
plt.title('Histogram of Petal Length')

plt.subplot(1, 2, 2)
sns.histplot(df['sepal width (cm)'], kde=True, bins=15)
plt.title('Histogram of Sepal Width')
plt.show()

# 3D histogram
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(df['petal length (cm)'], df['sepal width (cm)'], bins=15)

xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 0.2 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Count')
plt.title("3D Histogram of Petal Length and Sepal Width")
plt.show()
# Box plot for each feature by species
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='label', y='petal length (cm)', data=df)
plt.title('Box Plot of Petal Length by Species')

plt.subplot(1, 2, 2)
sns.boxplot(x='label', y='sepal width (cm)', data=df)
plt.title('Box Plot of Sepal Width by Species')
plt.show()

# 2D Box plot for petal length vs. sepal width
plt.figure(figsize=(8, 6))
sns.boxplot(x='petal length (cm)', y='sepal width (cm)', data=df)
plt.title('2D Box Plot of Petal Length vs. Sepal Width')
plt.show()

# QQ plot for petal length against normal distribution
plt.figure(figsize=(8, 6))
statis.probplot(df['petal length (cm)'], dist="norm", plot=plt)
plt.title('QQ Plot of Petal Length')
plt.show()

# QQ plot for sepal width against normal distribution
plt.figure(figsize=(8, 6))
statis.probplot(df['sepal width (cm)'], dist="norm", plot=plt)
plt.title('QQ Plot of Sepal Width')
plt.show()

# 2D scatter plots for each pair of features
sns.pairplot(df, hue='label', diag_kind='kde')
plt.show()

# 3D scatter plot of sepal length, sepal width, and petal length
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for label in df['label'].unique():
    species_data = df[df['label'] == label]
    ax.scatter(
        species_data['sepal length (cm)'], 
        species_data['sepal width (cm)'], 
        species_data['petal length (cm)'], 
        label=label
    )

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('3D Scatter Plot by Species')
plt.legend()
plt.show()
# PDF for petal length by species
plt.figure(figsize=(8, 6))
for label in df['label'].unique():
    sns.kdeplot(df[df['label'] == label]['petal length (cm)'], label=label)

plt.title('Probability Density Function of Petal Length by Species')
plt.xlabel('Petal Length')
plt.ylabel('Density')
plt.legend()
plt.show()