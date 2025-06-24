import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import joblib
import pickle
from utils.FFNN_tf import load_model, load_history, plot_saved_history

def processes_data_test_acc_loss(path, path_scaler, path_label_encoder, sep=','):
    #
    path = path
    df = pd.read_csv(path, sep=sep)

    # Defines parameters
    X = df.drop(columns='label')
    y = df['label']                 # Labels: species ['>50K' '<=50K']

    # Encode the labels
    # Load the saved Label Encoder and saved scaler
    scaler = joblib.load(path_scaler)
    label_encoder = joblib.load(path_label_encoder)

    y_encoded = label_encoder.transform(y)

    # Step3: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test, scaler, label_encoder


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # ... (from previous training code) ...
    path_model = "data/ann_model.keras"
    path_his = "data/ann_training_history.pkl"
    path_label_encoder = "data/ann_label_encoder.pkl"
    path_scaler = "data/ann_scaler.pkl"

    # Load model
    model = load_model(path_model)

    # Load history
    history = load_history(path_his)
    print("\nHistory Keys:", history.keys())

    # Plot the training history
    plot_saved_history(path_his)

    # Get test accuracy and loss
    path = "https://drive.google.com/uc?id=17XQMzAh3_zSq63eCVgPKV2CJjSM7IbJQ"

    # Get parameters from same processes in training
    X_test_scaled, y_test, scaler, label_encoder = processes_data_test_acc_loss(path, path_scaler,
                                                                                path_label_encoder, sep=';')

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)

    # test accuracy and loss results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # -------------------------------------------------------------------------------------------------
    # Prediction from saved model
    sample_data = np.array([
        [-0.24320004666320394, -0.6149243270096076, -0.5736920616133286, -0.20008579059127254, -0.2676239234289699, 0.18656568084577085],
        [-0.5568042332050533, 0.11790742504660037, 0.9125568026699057, -0.20008579059127254, -0.2676239234289699, 0.6933081367419336],
        [0.7760135595978066, -0.3008455796228537, -1.3168164937549458, -0.20008579059127254, -0.2676239234289699, 2.3824496563958095],
        [-0.7920073731114404, -1.0317648341083885, -0.5736920616133286, -0.20008579059127254, -0.2676239234289699, 1.9601642764823404],
        [1.8736282124942796, 1.2151026167136045, 0.9125568026699057, 1.0830963206219297, -0.2676239234289699, -0.15126262308500432]
    ])

    # Scale new data
    sample_data_scaled = scaler.transform(sample_data)
    predictions = model.predict(sample_data_scaled)

    # Convert probabilities to binary labels
    binary_predictions = (predictions > 0.5).astype(int)

    predicted_labels = label_encoder.inverse_transform(binary_predictions)
    print(f"Predicted labels for sample data: {predicted_labels}")