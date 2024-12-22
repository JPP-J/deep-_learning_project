import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import pickle
import joblib

class cnn_train_set_pic:
    def __init__(self, data_path, test_path):
        self.data_path = data_path
        self.test_path = test_path
        self.model = None       # Optional initialization
        self.history = None     # Optional initialization

    def load_and_preprocess_data(self, path= None):
        """Load and preprocess images from the directory"""
        if path is None:
            path = self.data_path  # Default to training path if not specified

        images = []
        labels = []

        # Loop through each folder (0-9)
        for digit_folder in range(10):
            folder_path = os.path.join(path, str(digit_folder))
            # Get-search all jpg files in the folder
            files = glob.glob(os.path.join(folder_path, "*.jpg"))

            for file_path in files:
                # Load and preprocess image
                img = Image.open(file_path).convert('L')    # Convert to grayscale
                img = img.resize((28, 28))                  # Resize to MNIST standard size
                img_array = np.array(img) / 255             # Normalize pixel values
                images.append(img_array)
                labels.append(digit_folder)

        return np.array(images), np.array(labels)

    def create_cnn_model(self):
        """Create and return the CNN model"""
        model = models.Sequential([
            # First Convolutional Layer
            # layers.Conv2D(
            # filter number, : 32, 64
            # (kernel size), : filter size [(3,3), (5,5)]
            # activation='relu',
            # input_shape=(28, 28, 1)) : 1 for grey (1 channels), 3 for rgb
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # stride=(1,1) default
            layers.MaxPooling2D((2, 2)),                                            # pool_size=(2, 2) default or (3,3)

            # Second Convolutional Layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # Third Convolutional Layer
            layers.Conv2D(64, (3, 3), activation='relu'),

            # Flatten layer to connect to Dense layers
            layers.Flatten(),

            # Dense layers/Fully connected
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),     # Add dropout to prevent overfitting 0.5 is common (0.2-0.5 is OK)
            layers.Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
        ])

        return model

    def train_set_pic_cnn(self):
        # 1. Load and preprocess data
        print("Loading and preprocessing data...")
        X, y = self.load_and_preprocess_data()

        # Reshape images to include channel dimension
        X = X.reshape(X.shape[0], 28, 28, 1)

        # 2. Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Create and compile model
        print("Creating and compiling model...")
        self.model = self.create_cnn_model()
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # 4. Train the model
        print("Training model...")
        self.history = self.model.fit(X_train, y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(X_val, y_val))

        return self.model, self.history

    def get_summary(self):
        self.model.summary()

    def plot_model(self):       # Only have Graphviz Software
        # Visualize the model
        plot_model(self.model, to_file='data/cnn_model_structure.png', show_shapes=True, show_layer_names=True)

    def plot_accuracy_loss(self):
        # Plot training history
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_cm_test(self):
        # Create confusion matrix for test set
        print("\nEvaluating model on test set...")
        X_test, y_test = self.load_and_preprocess_data(path= self.test_path)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)  # Fixed: Use X_test instead of X

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate and print test set accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

        # Create and plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 'd' integer format
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()



class cnn_saved_model_usage:
    def __init__(self, path_model, path_his, test_path):
        self.path_model = path_model
        self.path_his = path_his
        self.test_path = test_path
        self.model = None           # Optional initialization
        self.history_dict = None     # Optional initialization

    def load_model(self):  # to thrive data after trained
        # Load the model
        self.model = keras.models.load_model(self.path_model)

        # Recompile the model with metrics
        self.model.compile(
            optimizer='adam',  # or whatever optimizer you're using
            loss='sparse_categorical_crossentropy',  # or your loss function
            metrics=['accuracy']  # or your desired metrics
        )

        # model summary
        print(self.model.summary())

        return self.model

    def load_history(self):
        self.history_dict = joblib.load(self.path_his)
        return self.history_dict

    def plot_saved_history(self):
        # Load the saved history
        # saved_history = self.load_history(self.path_his)

        saved_history = self.history_dict

        # Create plots
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(saved_history['accuracy'], label='Training Accuracy')
        plt.plot(saved_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(saved_history['loss'], label='Training Loss')
        plt.plot(saved_history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print final metrics
        print("\nFinal Training Metrics from Saved History:")
        print(f"Training Accuracy: {saved_history['accuracy'][-1]:.4f}")
        print(f"Validation Accuracy: {saved_history['val_accuracy'][-1]:.4f}")
        print(f"Training Loss: {saved_history['loss'][-1]:.4f}")
        print(f"Validation Loss: {saved_history['val_loss'][-1]:.4f}")

    def load_and_preprocess_data(self, path= None):
        """Load and preprocess images from the directory"""
        if path is None:
            path = self.test_path  # Default to training path if not specified

        images = []
        labels = []

        # Loop through each folder (0-9)
        for digit_folder in range(10):
            folder_path = os.path.join(path, str(digit_folder))
            # Get-search all jpg files in the folder
            files = glob.glob(os.path.join(folder_path, "*.jpg"))

            for file_path in files:
                # Load and preprocess image
                img = Image.open(file_path).convert('L')    # Convert to grayscale
                img = img.resize((28, 28))                  # Resize to MNIST standard size
                img_array = np.array(img) / 255             # Normalize pixel values
                images.append(img_array)
                labels.append(digit_folder)

        return np.array(images), np.array(labels)

    def plot_cm_test(self):
        # Create confusion matrix for test set
        print("\nEvaluating model on test set...")
        X_test, y_test = self.load_and_preprocess_data(path= self.test_path)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)  # Fixed: Use X_test instead of X

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate and print test set accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

        # Create and plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 'd' integer format
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


# ===================================================================================

# # Save the history with pickle example
#     with open('data/training_history.pkl', 'wb') as file:
#         pickle.dump(history.history, file)