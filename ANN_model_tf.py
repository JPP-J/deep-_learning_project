import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Part1: Load data
path = "https://drive.google.com/uc?id=17XQMzAh3_zSq63eCVgPKV2CJjSM7IbJQ"
df = pd.read_csv(path, sep=';')
print(f'example dataset:\n {df.head()}')
print(f'columns name: {df.columns.values}')

# -----------------------------------------------------------------------------------------------------
# Part2: Pre-processing
# Defines parameters
X = df.drop(columns='label')
y = df['label']             # Labels: species ['>50K' '<=50K']

# Encode the labels (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f'labels size: {np.unique(y, return_counts=True)}')  # Balanced

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42,
                                                    stratify=y_encoded)
print(X_train.shape[1])

# -----------------------------------------------------------------------------------------------------
# Part3: Create the Pipeline model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KerasClassifierWrapper(input_dim=X_train.shape[1], epochs=100))       # number feature input
])

# Train the Neural Network
pipeline.fit(X_train, y_train)
print("\nModel training complete..........")

# Cross-validation scores while training
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)

# Evaluate the Model with test set
accuracy = pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Get model summary
pipeline.named_steps['model'].get_summary()

# Plot model
# pipeline.named_steps['model'].plot_model()        # Only have Graphviz Software

# Plot the training history
pipeline.named_steps['model'].plot_history()

# -----------------------------------------------------------------------------------------------------
# Part4: Saved relevant files
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
sample_data = np.array([
    [-0.24320004666320394, -0.6149243270096076, -0.5736920616133286, -0.20008579059127254, -0.2676239234289699, 0.18656568084577085],
    [-0.5568042332050533, 0.11790742504660037, 0.9125568026699057, -0.20008579059127254, -0.2676239234289699, 0.6933081367419336],
    [0.7760135595978066, -0.3008455796228537, -1.3168164937549458, -0.20008579059127254, -0.2676239234289699, 2.3824496563958095],
    [-0.7920073731114404, -1.0317648341083885, -0.5736920616133286, -0.20008579059127254, -0.2676239234289699, 1.9601642764823404],
    [1.8736282124942796, 1.2151026167136045, 0.9125568026699057, 1.0830963206219297, -0.2676239234289699, -0.15126262308500432]
])

# Get prediction
sample_data_scaled = scaler.transform(sample_data)
predictions = pipeline.predict(sample_data_scaled)
predicted_labels = label_encoder.inverse_transform(predictions)
print(f"Predicted labels for sample data: {predicted_labels}")

