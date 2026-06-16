import torch
import torch.nn as nn

def main():
    print("Day 7: Building a Deep Neural Network")

    # 1. Data (XOR Problem - A famous problem that a single layer can't solve)
    # Inputs: 2 numbers. Target: 1 number.
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_true = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # 2. DEEP Model
    # We are adding a "Hidden Layer" with 4 internal neurons.
    model = nn.Sequential(
        nn.Linear(2, 4), # Input layer (2 inputs) -> Hidden Layer (4 nodes)
        nn.ReLU(),       # The bend, non-linear activation
        nn.Linear(4, 1), # Hidden Layer (4 nodes) -> Output Layer (1 guess)
        nn.Sigmoid()     # Final percentage guess
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # 3. Training Loop (Run it for 1000 epochs since it's a harder problem)
    for epoch in range(1, 1001):
        predictions = model(inputs)
        loss = loss_fn(predictions, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every 200 epochs to keep the terminal clean
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4} | Loss: {loss.item():.4f}")

    print("\nFinal Predictions:")
    print(model(inputs).detach().numpy())

if __name__ == "__main__":
    main()