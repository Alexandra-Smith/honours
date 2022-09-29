from all_modalities import *

def main():

    # prepare patches
    X, y = get_data(X4, patch_size=35, number_of_ims=10, num_strides=6) # strides = 6 -> 1225 patches used per training image
    # print(X[0].shape)
    # split data
    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y)

    # fit CNN model
    model, history = train(X_train, y_train, X_val, y_val)

    # evaluate model performance
    loss, accuracy = evaluate_model(model, history, X_test, y_test)
    print("\nModel's Evaluation Metrics: ")
    print("---------------------------")
    print("Accuracy: {} \nLoss: {}".format(accuracy, loss))

    # Predict accuracy manually
    perc = acc(model, X_test, y_test, 0.5)
    print("Accuracy: {}".format(perc))

if __name__ == "__main__":
    main()