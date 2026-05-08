#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
from torch import nn


class NonNegativeLogisticRegression(nn.Module):
    def __init__(self, n_features, features=None, n_classes=1, lam=0.0):
        super().__init__()

        if n_classes == 1:  # Binary case (backward compatibility)
            self.weight = nn.Parameter(torch.abs(torch.randn(n_features) * 0.01))
            self.bias = nn.Parameter(torch.zeros(1))
        else:  # Multi-class case
            self.weight = nn.Parameter(torch.abs(torch.randn(n_classes, n_features) * 0.01))
            self.bias = nn.Parameter(torch.zeros(n_classes))

        self.n_classes = n_classes
        self.regularization = lam
        self.features = features

    def forward(self, x):
        if self.n_classes == 1:  # Binary case
            return torch.sigmoid(torch.matmul(x, self.weight) + self.bias)
        else:  # Multi-class case
            logits = torch.matmul(x, self.weight.T) + self.bias
            return torch.softmax(logits, dim=-1)

    def project_weights(self):
        """Project weights to non-negative orthant (in-place)."""
        with torch.no_grad():
            self.weight.clamp_(min=0.0)

    def compute_loss(self, features_full, labels):
        predictions = self(features_full)

        if self.n_classes == 1:  # Binary case
            predictions = predictions.squeeze()
            bce_loss = nn.functional.binary_cross_entropy(predictions, labels, reduction='mean')
        else:  # Multi-class case
            # Convert labels to long type for cross-entropy
            if labels.dtype != torch.long:
                labels = labels.long()
            ce_loss = nn.functional.cross_entropy(
                torch.matmul(features_full, self.weight.T) + self.bias,  ## logits, not softmaxed
                labels,
                reduction='mean'
            )
            bce_loss = ce_loss

        # Add L1 regularization if specified
        if self.regularization > 0:
            l1_reg = self.regularization * torch.sum(torch.abs(self.weight))
            return bce_loss + l1_reg

        return bce_loss

    def fit(self, features_full, labels, max_iter=1000):
        features_full = torch.tensor(features_full).to(torch.float32)
        labels = torch.tensor(labels).to(torch.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

        for _ in range(max_iter):
            optimizer.zero_grad()
            loss = self.compute_loss(features_full, labels)
            loss.backward()
            optimizer.step()
            self.project_weights()

        # Store coefficients for sklearn compatibility
        if self.n_classes == 1:
            self.coef_ = self.weight.detach().numpy().reshape(1, -1)
        else:
            self.coef_ = self.weight.detach().numpy()

    def score(self, X_test, y_test):
        y_preds = self.predict(X_test)
        if self.n_classes == 1:
            acc = (y_preds.squeeze() == torch.tensor(y_test).float()).float().mean()
        else:
            acc = (y_preds == torch.tensor(y_test).long()).float().mean()
        return acc.detach().numpy()

    def predict(self, X_test):
        X_test_tensor = torch.tensor(X_test).to(torch.float32)

        if self.n_classes == 1:
            return (self.forward(X_test_tensor) > 0.5).float()
        else:
            probs = self.forward(X_test_tensor)
            return torch.argmax(probs, dim=-1).detach().numpy()

    def predict_proba(self, X_test):
        X_test_tensor = torch.tensor(X_test).to(torch.float32)
        probs = self.forward(X_test_tensor)

        if self.n_classes == 1:
            # For binary case, return probabilities for both classes
            probs = probs.squeeze()
            return torch.stack([1 - probs, probs], dim=-1).detach().numpy()
        else:
            return probs.detach().numpy()

    def get_string_representation(self):
        """Return a human-readable string of feature weights."""
        if self.features is None:
            raise ValueError("Cannot generate string representation without feature names.")
        model_weights = self.weight.detach().numpy()
        sorted_weight_indices = np.argsort(model_weights)[::-1]
        modelstr = "feature: weight\n\n"
        for idx in sorted_weight_indices:
            if model_weights[idx] != 0:
                modelstr += f"{self.features[idx]}: {model_weights[idx]:.3f}\n"
        return modelstr

    def get_features_used(self):
        """Return indices of features with non-zero weights."""
        return np.argwhere(self.weight.detach().numpy() > 0).flatten().tolist()
