import numpy as np

def relu(x):
    """Rectified Linear Unit: returns x if x > 0, else 0"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: squishes values to the range (0, 1)"""
    return 1 / (1 + np.exp(-x))

def main():
    print("Day 2: Activation Functions")

    # Simulating raw output from Day 1's dot product Calculation
    # We will use some negative and positive numbers to see the effect of activations
    raw_layer_outputs = np.array([[-2.5],
                                  [0.0],
                                  [1.2],
                                  [3.8]])
    
    print("Raw Layer Outputs:\n", raw_layer_outputs)
    print("\nApplied ReLU (Notice: Negatives become 0):")
    print(relu(raw_layer_outputs))
    print("\nApplied Sigmoid (Notice: Every value is squished between 0 and 1):")
    print(sigmoid(raw_layer_outputs))

if __name__ == "__main__":
    main()