import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
from SDG_GAN import Generator, Discriminator, CGAN
from tensorflow.keras.losses import MeanSquaredError
from RF_Classifier import RandomForestModel
import warnings

warnings.filterwarnings("ignore")

os.chdir("C:\\Users\\cedri\\PycharmProjects\\ADA_Project")

def plot_fraud_over_time(df, time_col, target_col):
    # Filter to only fraudulent transactions
    fraud_df = df[df[target_col] == 1]

    # Convert time to a more interpretable unit - hours, and cycle every 24 hours
    fraud_df[time_col] = (fraud_df[time_col] / 3600) % 24

    plt.figure(figsize=(10,6))
    plt.plot(fraud_df[time_col], fraud_df[target_col], 'ro')
    plt.xlabel('Time (hours)')
    plt.ylabel('Fraudulent Transaction')
    plt.title('Fraudulent transactions over time')

    plt.show()

def plot_feature_distributions(df):
    # Get the list of columns in the dataframe
    cols = df.columns

    # Set up the figure with a grid of plots
    sns.set(style="ticks")
    fig, axes = plt.subplots(nrows=len(cols), figsize=(10, 40))

    # Loop over the columns and plot each one
    for i, col in enumerate(cols):
        ax = axes[i]
        sns.kdeplot(df[col], ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')

    # Tighten up the layout and show the plot
    fig.tight_layout()
    plt.savefig('Features distribution')
    plt.show()


def split_data(X, y, train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42):
    # Check if proportions add up to 1
    if train_size + valid_size + test_size != 1.0:
        raise ValueError("Proportions do not add up to 1.")

    # Calculate split sizes
    valid_test_size = valid_size + test_size
    valid_ratio_from_valid_test = valid_size / valid_test_size

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=valid_test_size, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=(1 - valid_ratio_from_valid_test),
                                                        random_state=random_state)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def calculate_num_samples(num_majority, num_minority, desired_ratio):
    num_samples = int(num_majority / desired_ratio) - num_minority
    # If calculated number is negative, set it to zero
    num_samples = max(num_samples, 0)
    return num_samples


def visualize_real_and_synthetic_data(real_data, synthetic_data):
    feature_names = real_data.columns

    if not isinstance(real_data, np.ndarray):
        real_data = real_data.values
    if not isinstance(synthetic_data, np.ndarray):
        synthetic_data = synthetic_data.values

    real_mean = np.mean(real_data, axis=0)
    synthetic_mean = np.mean(synthetic_data, axis=0)
    real_std = np.std(real_data, axis=0)
    synthetic_std = np.std(synthetic_data, axis=0)

    for feature_name, real_m, synthetic_m, real_s, synthetic_s in zip(feature_names, real_mean, synthetic_mean, real_std, synthetic_std):
        print(f"Feature: {feature_name}")
        print(f"Real Fraud Mean: {real_m:.4f}  Synthetic Fraud Mean: {synthetic_m:.4f}")
        print(f"Real Fraud Std: {real_s:.4f}  Synthetic Fraud Std: {synthetic_s:.4f}")
        print()

    num_features = real_data.shape[1]
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 6 * num_features))

    for i, ax in enumerate(axes):
        ax.hist(real_data[:, i], bins=50, alpha=0.5, label='Real Fraud')
        ax.hist(synthetic_data[:, i], bins=50, alpha=0.5, label='Synthetic Fraud')
        ax.set_title(feature_names[i])
        ax.legend()

    plt.tight_layout()
    plt.show()

    real_corr = np.corrcoef(real_data, rowvar=False)
    synthetic_corr = np.corrcoef(synthetic_data, rowvar=False)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Real Fraud Correlation')
    axes[1].imshow(synthetic_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Synthetic Fraud Correlation')

    plt.tight_layout()
    plt.show()

def compare_distributions(X_train, X_valid, X_test, synthetic_data, feature_names):
    # Combine training data and synthetic data
    X_train_augmented = pd.concat([X_train, synthetic_data])

    num_features = X_train_augmented.shape[1]
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 6 * num_features))

    for i, ax in enumerate(axes):
        ax.hist(X_train_augmented.iloc[:, i], bins=50, alpha=0.5, label='Training (original + synthetic)')
        ax.hist(X_valid.iloc[:, i], bins=50, alpha=0.5, label='Validation')
        ax.hist(X_test.iloc[:, i], bins=50, alpha=0.5, label='Test')
        ax.set_title(feature_names[i])
        ax.legend()

    plt.tight_layout()
    plt.show()

data = pd.read_csv('creditcard.csv')

"""
# Explore the time dependencies of our data
plot_fraud_over_time(data, 'Time', 'Class')

# Explore the features available in dataframe and check for variances, max values and missing values for each feature
data_description = data.describe()
print(data.head())
print(data.max())
print(data.var())
print(data.isna().sum())

# Count the occurrences of fraud and no fraud and print them
fraud_occ = data['Class'].value_counts()
print(fraud_occ)

#Print the ratio of fraud cases
print(fraud_occ / len(data.index))"""

# Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# Scale the features Time and Amount
scaler = RobustScaler()
scaled_amount = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
scaled_time = scaler.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)
data.insert(0,'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Plot the distribution for all features
"""plot_feature_distributions(data)

# Plot boxplots for all features
fig, ax = plt.subplots(figsize=(15,20))
data.boxplot(ax=ax)
plt.xticks(rotation=90)
plt.savefig('Features boxplot')
plt.show()"""

# Split data into training, validation and test sets
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

# Set up the cGAN architecture
input_dim = X_train.shape[1]
label_dim = 1
latent_dim = 50

# Instantiate and compile your models
cgan = CGAN(input_dim, label_dim, latent_dim)
optimizer = Adam(0.0001, 0.5)
cgan.compile_models(optimizer)

# Train the model
cgan.train(X_train, y_train, epochs=100, batch_size=64)

# Determine the number of fraudulent transactions dependant of the desired ratio of non-fraudulent to fraudulent cases
num_majority = sum(y_train == 0)
num_minority = sum(y_train == 1)
desired_ratio = 4  # ratio of non-fraudulent to fraudulent
num_samples = calculate_num_samples(num_majority, num_minority, desired_ratio)

# Generate synthetic fraudulent transactions
synthetic_data = cgan.generate_samples(num_samples)

# Convert synthetic_data to DataFrame
synthetic_data = pd.DataFrame(synthetic_data, columns=X_train.columns)

# Create labels for synthetic_data
synthetic_labels = pd.DataFrame(np.ones((synthetic_data.shape[0], )))

# Concatenate the synthetic data with the original data
X_train_combined = pd.concat([X_train, synthetic_data])
y_train_combined = pd.concat([y_train, synthetic_labels])

# Initialize the Random Forest model
rf_model = RandomForestModel()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Tune hyperparameters using validation set
#rf_model.tune_hyperparameters(X_valid, y_valid, param_grid)

# Train the Random Forest classifier with the augmented training data
rf_model.train(X_train_combined, y_train_combined)

# Evaluate the classifier on the test set
rf_model.evaluate(X_test, y_test)

# Extract real fraud data from train set and generate synthetic fraud data
real_fraud_train = X_train[y_train == 1]
num_synthetic_fraud = len(real_fraud_train)
synthetic_fraud_train = cgan.generate_samples(num_synthetic_fraud)
synthetic_fraud_train = pd.DataFrame(synthetic_fraud_train, columns=X_train.columns)

# Call the visualization functions
print("Training Data Comparison")
visualize_real_and_synthetic_data(real_fraud_train, synthetic_fraud_train)
compare_distributions(X_train, X_valid, X_test, synthetic_data, X_train.columns)