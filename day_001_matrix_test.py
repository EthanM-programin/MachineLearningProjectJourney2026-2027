import numpy as np

def main():
    print("Day 1: Verifying ML Environment")

    # Initialize a 2x3 matrix (Weights) and a 3x1 vector (Inputs)
    weights = np.array([[1, 2, 3],
                        [4, 5, 6]])
    inputs = np.array([[0.5],
                       [1.0],
                       [1.5]])
    
    #Compute the dot product (the core of a single layer calculation)
    output = np.dot(weights, inputs)

    print("Weights (Matrix 2x3):\n", weights)
    print("\nInputs (Vector 3x1):\n", inputs)
    print("\nOutput (Layer Result):\n", output)

if __name__ == "__main__":
    main()