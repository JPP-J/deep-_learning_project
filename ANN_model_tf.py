import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import joblib
from utils.FFNN_tf import KerasClassifierWrapper

# Set specific seed number
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------------------------------------------------------------------------------
# Part1: Load data
def load_data(path, sep=';'):
    df = pd.read_csv(path, sep=';')
    print(f'example dataset:\n {df.head()}')
    print(f'columns name: {df.columns.values}')
    print(f'dataset shape: {df.shape}')
    print(f'dataset info: {df.info()}')
    print(f'dataset description: {df.describe()}')  

    return df

# -----------------------------------------------------------------------------------------------------
# Part2: Pre-processing
def data_preprocessing(df):
    # Defines parameters
    X = df.drop(columns='label')
    y = df['label']             # Labels: species ['>50K' '<=50K']

    # Encode the labels (y)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f'labels size: {np.unique(y, return_counts=True)}')  # Balanced

    return X, y_encoded, label_encoder

# -----------------------------------------------------------------------------------------------------
# Part3: Create the Pipeline model
def create_pipeline(input_dim, X_train, X_test, y_train, y_test, epochs=100, cv=False):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KerasClassifierWrapper(input_dim=input_dim, epochs=epochs))       # number feature input
    ])


    # Train the Neural Network
    pipeline.fit(X_train, y_train)
    print("\nModel training complete..........")

    # Cross-validation scores while training
    if cv == True:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        print("Cross-validation scores:", scores)

    # Evaluate the Model with test set
    accuracy = pipeline.score(X_test, y_test)  # Evaluate the model on the test set
    print(f"Test accuracy: {accuracy:.2f}")

    # Get model summary
    pipeline.named_steps['model'].get_summary()

    # Plot model
    # pipeline.named_steps['model'].plot_model()        # Only have Graphviz Software

    # Plot the training history
    pipeline.named_steps['model'].plot_history()

    return pipeline

# -----------------------------------------------------------------------------------------------------
# Part4: Saved relevant files
def save_model(pipeline, label_encoder):
# Save the model pipeline> model> model(keras)
    pipeline.named_steps['model'].model.save('data/ann_model.keras')

    # Save the training history
    history = pipeline.named_steps['model'].get_history()
    joblib.dump(history, 'data/ann_training_history.pkl')

    # Save the scaler
    scaler = pipeline.named_steps['scaler']
    joblib.dump(scaler, 'data/ann_scaler.pkl')

    # Save the LabelEncoder
    joblib.dump(label_encoder, 'data/ann_label_encoder.pkl')

# -----------------------------------------------------------------------------------------------------
# Part5: Make Predictions: 5 samples (Example not real data)
def infer_and_predict(pipeline, label_encoder):
    sample_data = np.array([
        [-0.24320004666320394, -0.6149243270096076, -0.5736920616133286, -0.20008579059127254, -0.2676239234289699, 0.18656568084577085],
        [-0.5568042332050533, 0.11790742504660037, 0.9125568026699057, -0.20008579059127254, -0.2676239234289699, 0.6933081367419336],
        [0.7760135595978066, -0.3008455796228537, -1.3168164937549458, -0.20008579059127254, -0.2676239234289699, 2.3824496563958095],
        [-0.7920073731114404, -1.0317648341083885, -0.5736920616133286, -0.20008579059127254, -0.2676239234289699, 1.9601642764823404],
        [1.8736282124942796, 1.2151026167136045, 0.9125568026699057, 1.0830963206219297, -0.2676239234289699, -0.15126262308500432]
    ])

    # Get prediction
    scaler = pipeline.named_steps['scaler']

    sample_data_scaled = scaler.transform(sample_data)
    predictions = pipeline.predict(sample_data_scaled)
    predicted_labels = label_encoder.inverse_transform(predictions)

    print(f"Predicted labels for sample data: {predicted_labels}")

if __name__ == "__main__":
    path = "https://drive.google.com/uc?id=17XQMzAh3_zSq63eCVgPKV2CJjSM7IbJQ"
    df  = load_data(path, sep=';')
    X, y, label_encoder = data_preprocessing(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
    pipeline = create_pipeline(X_train.shape[1], X_train, X_test, y_train, y_test, epochs=50, cv=True)
    # save_model(pipeline, label_encoder)
    # infer_and_predict(pipeline, label_encoder)

