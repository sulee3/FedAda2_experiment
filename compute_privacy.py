from privacy_analysis import compute_privacy_sgm as cps

n = 400 # total number of clients
batch_size = 40 # number of selected clients
noise_multiplier = 1
glob_epochs = 1801 # global epoch rounds
epochs = glob_epochs * batch_size / n # total number of communication rounds times number of selected clients each round divided by total number of clients
delta = 1/n

eps, opt_order = cps.compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta)
print(f"Outputs are epsilon: {eps}, opt_order: {opt_order} for n {n}, batch size {batch_size}, noise multiplier {noise_multiplier}, epochs {epochs}, delta {delta}")

# n: Number of examples in the training data.
# batch_size: Batch size used in training.
# noise_multiplier: Noise multiplier used in training.
# (gaussian noise: var=sigma^2 C^2, sigma is the noise multiplier)
# epochs: Number of epochs in training.
# delta: Value of delta for which to compute epsilon.
