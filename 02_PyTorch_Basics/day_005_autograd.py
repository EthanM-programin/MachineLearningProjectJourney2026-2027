import torch

def main():
    print("Day 5: Enter PyTorch & Autograd")

    # Create a tensor. 'requires_grad=True' tells PyTorch to track operations on this tensor
    weight = torch.tensor([2.5], requires_grad=True)
    print(f"Starting Weight: {weight.item()}")

    # Loss function: L = w^2 (Simple parabolic Loss)
    loss = weight ** 2

    # Calculate gradient (slope/derivative) automatically, no calculus needed.
    loss.backward()

    print(f"Calculated Gradient (Slope): {weight.grad.item():.4f}")

    # Take step down slope (Gradient Descent)
    learning_rate = 0.1
    with torch.no_grad(): # Pause tracking while we update weight
        weight -= learning_rate * weight.grad

    print(f"New Updated Weight: {weight.item():.4f}")

if __name__ == "__main__":
    main()