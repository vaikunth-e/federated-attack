# Federated Gradient Inversion Attack

This project demonstrates a gradient inversion attack in a simple federated learning setup. A Flower server coordinates multiple PyTorch clients trained on Fashion-MNIST. During local training, a client logs the model weights and gradients from a private training batch. The attack script then reconstructs an input image by optimizing a dummy image until its gradients match the captured client gradients.

## What this demonstrates

Federated learning avoids sending raw client data to the server, but gradients can still leak information about private examples. This project shows that, under a simplified setup, an attacker with access to a client's gradients and model weights can recover a visual approximation of the original input.

## Architecture

- `server.py`: starts a Flower FedAvg server
- `client.py`: trains a Fashion-MNIST client and logs one-batch gradients
- `attack.py`: performs gradient matching to reconstruct the private input
- `grads/`: generated gradient snapshots and reconstruction outputs
