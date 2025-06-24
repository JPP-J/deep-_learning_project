import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import joblib
from utils.FFNN_pt import TorchClassifierWrapper, saved_model_usage

# Set specific seed number
np.random.seed(42)

# Part1: Load data
def load_data():
    # path = "https://drive.google.com/uc?id=17XQMzAh3_zSq63eCVgPKV2CJjSM7IbJQ"
    path = "https://drive.google.com/uc?id=1QctGSSR5wSQk6cbdjBrKjYcUPs6PHNHN"
    df = pd.read_csv(path, sep=',')
    na = df.isnull().sum()      # missing value at ['age']

    print(f'example dataset:\n {df.head(5)}')
    print(f'columns name: {df.columns.values}')
    print(f'target: {df['y'].unique()}')
    print(f'shape: {df.shape}')
    print(f'Null data:\n{na}')

    return df
# -----------------------------------------------------------------------------------------------------
# Part2: Pre-processing
def pre_processing(df):
    # Fill null with mean values
    df['age'] = df['age'].fillna(df['age'].mean())

    # Focus on customer that age < 80
    df = df[df['age'] < 80]

    # Drop unnecessary attributes and assign to new df
    df_n = df.drop(['id','contact','day','month','default','duration'], axis=1).reset_index(drop=True)

    # Categorical features and numerical features
    categorical_features = df_n.select_dtypes(include=[object]).columns.values
    categorical_features = categorical_features[categorical_features != 'y']
    categorical_features = np.append(categorical_features, 'campaign')
    numerical_features = df_n.select_dtypes(include=[np.number]).columns.values
    numerical_features = numerical_features[numerical_features != 'campaign']
    print("columns: ",df_n.columns)
    print("cat: ", categorical_features)
    print("num: ", numerical_features)

    # Defines parameters
    X = df_n.drop(columns='y')
    y = df_n['y']             # Labels:  ['no' 'yes']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Converts to 0, 1, 2, ...

    preprocessor = ColumnTransformer([
        # 'cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),  # Label Encoding
        ('num', StandardScaler(), numerical_features)     # Standardization
    ], remainder='passthrough')

    return X, y, preprocessor, label_encoder

# -----------------------------------------------------------------------------------------------------
# Part3: Create the Pipeline model
def train(X, y, preprocessor):

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', TorchClassifierWrapper(hidden_dim=32, output_dim=1,  # number feature input
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
def evaluate_model(pipeline, X_test, y_test, X_train, y_train):
    print("\nModel evaluation..........")

    # Cross-validation scores while training
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validation scores:", scores)
    print("Average CV accuracy:", scores.mean())

    # Evaluate the Model with test set
    accuracy = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}")


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
    df = load_data()                                              # Load data
    X, y, preprocessor, label_encoder = pre_processing(df)        # pre-processing
    print(f"X shape: {X.shape}, y shape: {y.shape}")               # Check shapes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data
    pipeline = train(X_train, y_train, preprocessor)              # training
    evaluate_model(pipeline, X_test, y_test, X_train, y_train)                      # evaluate model


    # save_model(pipeline, label_encoder)                         # save model
    # use_model(X)                                                # usage saved model
