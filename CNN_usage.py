from utils.CNN import (cnn_saved_model_usage)


# Usage saved model
if __name__ == "__main__":
    # ... (previous training code) ...
    path_model = "data/mnist_cnn_model.keras"
    path_his = "data/mnist_cnn_training_history.pkl"

    test_path = "data/MNIST - JPG - testing"

    saved_model = cnn_saved_model_usage(path_model=path_model, path_his=path_his, test_path=test_path)
    print("\nLoading summary model...")
    saved_model.load_model()

    history = saved_model.load_history()
    print("\nLoading and plotting saved history...")
    saved_model.plot_saved_history()

    # saved_model.plot_cm_test()
