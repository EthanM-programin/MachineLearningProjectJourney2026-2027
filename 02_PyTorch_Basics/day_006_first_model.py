import torch
import torch.nn as nn

def main():
    print("Day 6: Our First PyTorch Neural Network")

    # 1. Data (Inputs and True Answers)
    inputs = torch.tensor([[0.5], [1.0], [1.5]])
    y_true = torch.tensor([[1.0], [0.0], [1.0]])

    # 2. Model (1 Linear dot-product layer + Sigmoid activation)
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.Sigmoid()
    )

    # 3. Loss Function (MSE) and Optimizer (Gradient Descent)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 4. Training Loop (Learn in 10 epochs)
    for epoch in range(1, 11):
        # Forward Pass (Guess)
        predictions = model(inputs)
        loss = loss_fn(predictions, y_true)

        # Backward Pass (Learn)
        optimizer.zero_grad() # Clear old math
        loss.backward() # Calculate new gradients
        optimizer.step() # Update weights

        print(f"Epoch {epoch:2} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()