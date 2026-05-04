import numpy as np

# simple rule-based classifier (placeholder)
def classify(image_array):
    mean_val = np.mean(image_array)
    if mean_val > 0.5:
        return "Bright Image"
    return "Dark Image"

if __name__ == "__main__":
    sample = np.random.rand(10,10)
    print(classify(sample))
