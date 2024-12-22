import pickle
import joblib
from utils.CNN import (cnn_train_set_pic)

# Training model
if __name__ == "__main__":
    # Set parameters
    train_path = "data/MNIST - JPG - training"
    test_path = "data/MNIST - JPG - testing"

    cnn_model_trainer = cnn_train_set_pic(train_path, test_path)
    model, history = cnn_model_trainer.train_set_pic_cnn()

    # Model summary
    cnn_model_trainer.get_summary()

    # Plot model
    # cnn_model_trainer.plot_model()        # Only have Graphviz Software

    # accuracy values
    print("\nFinal Training Accuracy:", f"{history.history['accuracy'][-1]:.4f}")
    print("Final Validation Accuracy:", f"{history.history['val_accuracy'][-1]:.4f}")

    # Plot accuracy and loss while training
    cnn_model_trainer.plot_accuracy_loss()

    # Plot confusion matrix with test set and show test accuracy and loss
    cnn_model_trainer.plot_cm_test()

    # Save the model
    model.save('data/mnist_cnn_model.keras')

    # Save the history
    joblib.dump(history.history, 'data/mnist_cnn_training_history.pkl')






