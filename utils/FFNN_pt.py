import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import os
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

# PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Define a simple model using Embeddings for Collaborative Filtering
class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(TorchModel, self).__init__()

        # Input Layer
        self.fc1 = nn.Linear(input_dim, out_features=hidden_dim)  # Input layer (equivalent to Dense(6, input_dim=input_dim))

        # Hidden Layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # First hidden layer
        self.dropout = nn.Dropout(0.3)  # Dropout after hidden layer

        # Output Layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer (equivalent to Dense(1))

    def forward(self, X):
        x = X.to(self.fc1.weight.device)
        x = torch.relu(self.fc1(x))     # Input layer + ReLU activation
        x = torch.relu(self.fc2(x))     # First hidden layer + ReLU
        x = self.dropout(x)             # Apply Dropout
        x = self.fc3(x)                 # final output
        # x = torch.sigmoid(x)            # Sigmoid activation for binary output
        # x = torch.sigmoid(self.fc3(x))  # Output layer + Sigmoid activation
        return x

class TorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim, output_dim, epochs=50, lr=0.001, criteria='cross-ent', 
                 batch_size=16, val_size=0.2, patience=3):
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_size = val_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.criteria = criteria
        self.pos_weight = None  # Optional: for handling class imbalance
        self.model = None  # initialize as None
        self.optimizer = None
        self.criterion = None
        self.history = None

    @ staticmethod
    def _get_pos_weight(y):
        counter = Counter(y)
        print("Counter:", counter)
        print(counter[0], counter[1])
        num_neg = counter[0]
        num_pos = counter[1]
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
        print("Positive weight:", pos_weight)
        return torch.tensor(pos_weight, dtype=torch.float32) 
    
    def _get_criterion(self):
        # Criteria choices
        if self.criteria == 'cross-ent':
            self.criterion = nn.CrossEntropyLoss()
        elif self.criteria == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.criteria == 'binary-logit':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif self.criteria == 'binary':
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Invalid criteria: {self.criteria}")
        
        return self.criterion
        # weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float).to(self.device)  
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    def fit(self, X, y):
        print("Unique labels in full y:", np.unique(y))
        if hasattr(X, "toarray"):
            X = X.toarray()

        input_dim = X.shape[1]  # Dynamically get input dimension for this fold
        self.input_dim = input_dim
        self.model = TorchModel(input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim).to(self.device)
        

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.pos_weight = self._get_pos_weight(y).to(self.device)  # Calculate positive weight for handling class imbalance
        # self.pos_weight = torch.tensor([3.0], dtype=torch.float32).to(self.device)  # Example fixed weight, can be calculated dynamically
        # self.criterion = nn.BCEWithLogitsLoss()  # Temporarily ignore pos_weight

        self.criterion = self._get_criterion()  # Get the criterion based on the specified criteria
        self.scaler = torch.amp.GradScaler()  # For mixed precision training

        torch.cuda.empty_cache()  # Clear GPU cache

        # Load and prepare data
        print(f'Initial load data....')

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)                      # Features
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(self.device)         # Target
        # y_tensor = y_tensor.squeeze()  # Fix: Remove extra dimension (Shape: [16])

        # Create dataset and split into train/validation sets
        # Use DataLoader for batch training
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int((1-self.val_size) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # can Reduced batch size
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)     # can Reduced batch size

        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        print(f'Start training data....')
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        epochs = self.epochs

        print(f"Training for {epochs} epochs with batch size {self.batch_size}...")
        val_labels = []
        for _, batch_y in val_dataloader:
            val_labels.extend(batch_y.view(-1).tolist())
        print("Val label distribution:", Counter(val_labels))
        
        # batch_x, batch_y = next(iter(train_dataloader))

        # for i in range(100):
        #     self.optimizer.zero_grad()
        #     outputs = self.model(batch_x)
        #     loss = self.criterion(outputs.view(-1), batch_y.view(-1).float())
        #     loss.backward()
        #     self.optimizer.step()

        #     probs = torch.sigmoid(outputs.view(-1))
        #     preds = (probs > 0.5).float()
        #     acc = (preds == batch_y.view(-1)).float().mean().item()
        #     print(f"Epoch {i+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")


        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in train_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device.type):  # Mixed precision
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs.view(-1), batch_y.view(-1).float())  # Fix: Ensure correct shape for BCEWithLogitsLoss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                # preds = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
                preds = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
                batch_y = batch_y.view(-1).float()
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_y.size(0)

                # print("Logits:", outputs.view(-1)[:5].detach().cpu().numpy())
                # print("Probs :", torch.sigmoid(outputs.view(-1))[:5].detach().cpu().numpy())
                # print("Preds :", preds[:5].detach().cpu().numpy())
                # print("Label :", batch_y[:5].cpu().numpy())

                # print("BATCH LABEL:", batch_y.view(-1)[:5])
                # print("BATCH LABEL dtype:", batch_y.dtype)
                # print("MODEL OUTPUT shape:", outputs.shape)
                # print("Model raw output logits:", outputs[:5])
                # print("Model raw output min/max:", outputs.min(), outputs.max())


            avg_train_loss = train_loss / len(train_dataloader)
            train_acc =  correct_train / total_train
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_x, batch_y in val_dataloader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs.view(-1), batch_y.view(-1).float())

                    val_loss += loss.item()
                    # preds = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
                    preds = (torch.sigmoid(outputs.view(-1)) > 0.5).float()
                    # preds = (outputs.view(-1) > 0).float()
                    batch_y = batch_y.view(-1).float()
                    correct_val += (preds == batch_y).sum().item()
                    total_val += batch_y.size(0)

                    # Inside validation loop
                    if total_val == 0:  # only once
                        print("VAL PRED:", preds[:10].cpu().numpy())
                        print("VAL TRUE:", batch_y[:10].cpu().numpy())

                    # print("Logits:", outputs.view(-1)[:5].detach().cpu().numpy())
                    # print("Probs :", torch.sigmoid(outputs.view(-1))[:5].detach().cpu().numpy())
                    # print("Preds :", preds[:5].detach().cpu().numpy())
                    # print("Label :", batch_y[:5].cpu().numpy())

            avg_val_loss = val_loss / len(val_dataloader)
            val_acc =  correct_val / total_val
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_acc"].append(val_acc)

            # Check for early stopping condition
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()  # Save the best model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Restore best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f'Finished training data....')
    
    def score(self, X, y):
        
        y_pred = self.predict(X)
        print(np.unique(y_pred, return_counts=True))

        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred , zero_division=1)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        report = classification_report(y, y_pred)

        return report, acc, precision, recall, f1

    def save_model(self, model_name:str=None):
        # Check if the folder already exists
        folder_name = "model"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        torch.save(self.model.state_dict(), f"model/{model_name}.pth")
        torch.save(self.model, f"model/{model_name}_complete.pth")
        torch.save(self.history, f"model/{model_name}_history.pth")

        print("Model and training history saved!")

    def predict(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits.view(-1))  # Apply sigmoid to get probabilities
            preds = (probs > 0.5).float()  # Convert probabilities to binary predictions
            # (outputs.cpu().numpy() > 0.5).astype(int)  # Convert to 0 or 1
            return preds.cpu().numpy()
        
    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        """Returns probability scores for classification"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()  # Probability scores

    def plot_performance(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(range(len(self.history['train_loss'])))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(self.history['train_loss'])))
        plt.legend()

        plt.tight_layout()
        plt.show()


class saved_model_usage:
    def __init__(self, path_model, path_his, path_pre):
        self.path_model = path_model
        self.path_his = path_his
        self.path_pre = path_pre
        self.model = None           # Optional initialization
        self.history = None     # Optional initialization
        self.preprocessor = None

    def load_model(self):
        # Load the saved model and training history
        self.model = torch.load(self.path_model)  # Load the model
        self.history = torch.load(self.path_his)  # Load the training history
        self.preprocessor = joblib.load(self.path_pre)


        # Optionally load the model's state_dict if needed
        # model.load_state_dict(torch.load("collaborative_filtering_param_model.pth"))

        print(f'\nModel Architecture:\n{self.model}')

        return self.model, self.history

    def preprocess_data_for_prediction(self, X):
        # Apply the same preprocessing steps that were applied during training
        X_processed = self.preprocessor.transform(X)  # This applies all preprocessing steps

        return X_processed
    def get_prediction(self, X):
        self.model.eval()
        X = self.preprocess_data_for_prediction(X)

        # Set the model to evaluation mode
        # Making a prediction with the loaded model
        with torch.no_grad():                                            # No need to track gradients during inference
            input_data = torch.tensor(X, dtype=torch.float32)            # test data
            predictions = self.model(input_data)
            predictions = (predictions.cpu().numpy() > 0.5).astype(int)  # Convert to 0 or 1
        return  predictions.flatten()

    def plot_saved_history(self):
        history = self.history

        # Plot the loss curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot the accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Print the final metrics after plotting
        print("\nFinal Metrics:")
        print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
