# Bank-Performance-Prediction-Using-Machine-Learning

LR

z = intercept + (weight_1 * feature_1) + (weight_2 * feature_2) + ... + (weight_n * feature_n)
probability = 1 / (1 + e^(-z))

kNN

distance = square root of [ (new_feature_1 - train_feature_1)^2 + (new_feature_2 - train_feature_2)^2 + ... + (new_feature_n - train_feature_n)^2 ]

SVM

w^T * x + b = 0
w^T * x + b = +1
w^T * x + b = -1
w^T * x_i + b >= +1  if y_i = +1
w^T * x_i + b <= -1  if y_i = -1
Minimize: ||w||^2 / 2
Subject to: y_i * (w^T * x_i + b) >= 1 for all training examples i.
decision_score = w^T * x_new + b
Polynomial kernel: (x_i^T * x_j + c)^d
Radial Basis Function (RBF) kernel: exp(-gamma * ||x_i - x_j||^2)

ANN

z_j = bias_j + (weight_j1 * output_from_prev_neuron_1) + (weight_j2 * output_from_prev_neuron_2) + ...
a_j = activation_function(z_j)
new_weight = old_weight - learning_rate * gradient_of_loss_with_respect_to_weight

CART

Gini(S) = 1 - Σ (p_i)^2
Entropy(S) = - Σ (p_i * log2(p_i))
MSE(S) = (1 / |S|) * Σ (y_i - μ)^2

