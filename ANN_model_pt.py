import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pickle
import joblib
from collections import Counter
from utils.FFNN_pt import TorchClassifierWrapper, saved_model_usage

# Set specific seed number
np.random.seed(42)

# Part1: Load data
def load_data(path, show_detail=False):
    
    df = pd.read_csv(path, sep=',')
    na = df.isnull().sum()      # missing value at ['age']

    if show_detail == True:
        print(f'example dataset:\n {df.head(5)}')
        print(f'columns name: {df.columns.values}')
        print(f'target: {df['y'].unique()}')
        print(f'original shape: {df.shape}')
        print(f'Null data:\n{na}')

    return df
# -----------------------------------------------------------------------------------------------------
def plot_category(df, column):
    # Plot the distribution of the target variable
    plt.figure(figsize=(8, 6))
    plt.bar(df[column].value_counts().index, df[column].value_counts().values, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
# -----------------------------------------------------------------------------------------------------
def resample_data(df, target_column='y'):
    # Split into majority and minority
    df_majority = df[df[target_column] == 'no']
    df_minority = df[df[target_column] == 'yes']

    # Downsample majority
    df_majority_downsampled = resample(df_majority,
                                    replace=False,              # without replacement
                                    n_samples=len(df_minority),  # match minority size
                                    random_state=42)

    # Combine
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

# Part2: Pre-processing
def pre_processing(df, plot=False, show_detail=False):
    # df = resample_data(df, target_column='y')  # Resample data to balance classes

    # Fill null with mean values
    df['age'] = df['age'].fillna(df['age'].mean())

    # Focus on customer that age < 80
    df = df[df['age'] < 80]

    # Drop unnecessary attributes and assign to new df
    df_n = df.drop(['id','contact','day','month','default','duration'], axis=1).reset_index(drop=True)

    # Categorical features and numerical features
    categorical_features = df_n.select_dtypes(include=[object]).columns.values
    categorical_features = categorical_features [~np.isin(categorical_features, ['y', 'education'])]
    categorical_features = np.append(categorical_features, 'campaign')

    ordinal_features = ['education']
    
    numerical_features = df_n.select_dtypes(include=[np.number]).columns.values
    numerical_features = numerical_features[numerical_features != 'campaign']

    # plot
    if plot == True:
        print("Plotting data distribution...\n")
        plot_category(df_n, 'y')  # Plot distribution of target variable

        for col in categorical_features:
            plot_category(df_n, col)  # Plot histogram for categorical features

        for col in numerical_features:
            plot_histogram(df_n, col)

        for col in ordinal_features:
            plot_category(df_n, col)  # Plot histogram for ordinal features


    # Defines parameters
    X = df_n.drop(columns='y')
    y = df_n['y']             # Labels:  ['no' 'yes']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Converts to 0, 1, 2, ...

    nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', nominal_transformer, categorical_features),
        ('ordinal', ordinal_transformer, ordinal_features),
        ('numerical', MinMaxScaler(), numerical_features)
    ], remainder='passthrough'  # numeric columns untouched
    )
    
    if show_detail == True:
        print("columns usage: ",df_n.columns)
        print("category feature: ", categorical_features)
        print("ordinal features: ", ordinal_features)
        print("numerical feature: ", numerical_features)
        print(f"Final X shape: {X.shape}, y shape: {y.shape}")               # Check shapes

    return X, y, preprocessor, label_encoder

# -----------------------------------------------------------------------------------------------------
# Part3: Create the Pipeline model
def pos_weight(y):
    # Compute class weights to handle class imbalance
    class_weight = compute_class_weight(class_weight='balanced',
                                        classes=np.unique(y),
                                        y=y)
    pos_weight = class_weight[1] / class_weight[0]  # Ratio of positive to negative class weights
    return pos_weight
# -----------------------------------------------------------------------------------------------------
# Train the model
def train(X, y, preprocessor):

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', TorchClassifierWrapper(hidden_dim=7, output_dim=1,  # number feature input
                                         epochs=10, lr=0.001, criteria='binary-logit',
                                         batch_size=32, val_size=0.2))       # number feature input
    ])

    pipeline.fit(X, y)
    print("\nModel training complete..........")

    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()
    print("Feature names after preprocessing:", feature_names)

    # Get predictions
    print("\nModel get prediction..........")
    x = X[500:505]
    predictions = pipeline.predict(x)
    print(predictions)

    # plot performances
    print("\nModel plot..........")
    pipeline.named_steps['model'].plot_performance()

    return pipeline
# -----------------------------------------------------------------------------------------------------
def evaluate_model(pipeline, X_test, y_test, X_train, y_train, cv=False):
    print("\nModel evaluation..........")

    # Cross-validation scores while training
    if cv == True:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print("Cross-validation scores:", scores)
        print("Average CV accuracy:", scores.mean())

    # Evaluate the Model with test set
    report, acc, precision, recall, f1 = pipeline.score(X_test, y_test)
    print(report)


# -----------------------------------------------------------------------------------------------------
# Part4: Saved relevant files
def save_model(pipeline, label_encoder):
    model_name = "ANN_pt"
    # Save the model pipeline> model> model(keras)
    pipeline.named_steps['model'].save_model(model_name)

    # Save the preprocessor
    preprocess = pipeline.named_steps['preprocess']
    joblib.dump(preprocess, f'model/{model_name}_preprocessor.pkl')

    # Save the LabelEncoder
    joblib.dump(label_encoder, f'model/{model_name}_label_encoder.pkl')

# -----------------------------------------------------------------------------------------------------
# Part5: Make Predictions and usage model:
def use_model(X):
    path_model = "model/ANN_pt_complete.pth"
    path_his = "model/ANN_pt_history.pth"
    path_pre = "model/ANN_pt_preprocessor.pkl"
    path_label = "model/ANN_pt_label_encoder.pkl"
    saved_model = saved_model_usage(path_model=path_model, path_his=path_his, path_pre=path_pre)
    model, history = saved_model.load_model()
    # saved_model.plot_saved_history()    # plot model

    # Get predictions
    print("\nModel get prediction..........")
    x = X[500:505]

    predictions = saved_model.get_prediction(x)
    print(predictions)


if __name__ == "__main__" :
    path = "https://drive.google.com/uc?id=1QctGSSR5wSQk6cbdjBrKjYcUPs6PHNHN"
    df = load_data(path=path, show_detail=False)                                                 # Load data
    X, y, preprocessor, label_encoder = pre_processing(df, plot=False, show_detail=False)        # pre-processing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    # Split data
    pipeline = train(X_train, y_train, preprocessor)                                             # training
    evaluate_model(pipeline, X_test, y_test, X_train, y_train, cv=False)                          # evaluate model
    # save_model(pipeline, label_encoder)                                                       # save model
    # use_model(X)                                                                              # usage saved model
