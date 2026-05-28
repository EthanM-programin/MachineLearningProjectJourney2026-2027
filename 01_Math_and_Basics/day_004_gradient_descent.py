import numpy as np

def gradient_descent_step(current_weight, gradient, learning_rate):
    """
    Updates the weight by taking a step in the opposite direction of the gradient.
    """
    return current_weight - (learning_rate * gradient)

def main():
    print("Day 4: Gradient Descent (AI Learning Rate)")

    # Starting weight (usually a random guess in a real network)
    weight = 2.5

    # The 'learning rate' controls how big of a step the AI takes
    learning_rate = 0.1

    print(f"Initial Starting Weight: {weight}\n")

    # Simulate 5 rounds of training (called 'epochs')
    for epoch in range(1, 6):
        # Simulate the slope calculation.
        # If the Loss function is a simple curve like y = x^2, then the 
        # calculus derivative (gradient) is exactly 2 * x
        gradient = 2 * weight

        # The AI updates the weight to aim toward zero loss gradient_descent_step
        weight = gradient_descent_step(weight, gradient, learning_rate)
        print(f"Epoch {epoch} | Gradient (Slope): {gradient:.4f} | New Updated Weight: {weight:.4f}")

if __name__ == "__main__":
    main()