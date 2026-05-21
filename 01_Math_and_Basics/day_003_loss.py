import numpy as np

def mse_loss(y_true, y_pred):
    """
    Calculates Mean Squared Error.
    Takes the difference, squares it, and finds the average (mean).
    """
    return np.mean(np.square(y_true - y_pred))

def main():
    print("Day 3: Calculating Loss (Mistakes)")

    # Correct Answers (e.g., 1 = True, 0 = False)
    y_true = np.array([1.0, 0.0, 1.0, 1.0])

    # AI's Predictions (Sigmoid function)
    y_pred = np.array([0.9, 0.1, 0.4, 0.8])

    print("True Answers: ", y_true)
    print("AI Predictions: ", y_pred)

    # Calculate the loss
    loss = mse_loss(y_true, y_pred)
    print(f"\nMean Squared Error (Loss): {loss:.4f}")

if __name__ == "__main__":
    main()