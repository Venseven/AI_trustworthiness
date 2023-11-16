import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input
        self.linear = nn.Linear(input_dim, 1)  # Linear layer for logistic regression

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        return self.linear(x)
