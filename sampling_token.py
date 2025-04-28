import numpy as np

# Step 1: Simulate some logits (raw scores from a model)
logits = np.array([3.2, 1.5, -0.8])  # Example logits for 3 tokens: "cat", "dog", "fish"

# Step 2: Apply softmax to turn logits into probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

probabilities = softmax(logits)

print("Probabilities:", probabilities)

# Step 3: Sample one token based on these probabilities
# (simulate picking the next token)

# Tokens corresponding to logits
tokens = ["cat", "dog", "fish"]

# Randomly pick a token according to the probabilities
next_token = np.random.choice(tokens, p=probabilities)

print("Sampled next token:", next_token)
