import time
from typing import Generator, Iterable, List, Tuple

import jax
import jax.numpy as np
import tensorflow_datasets as tfds
from jax import grad, jit, random, vmap

from data import MNIST
from nn_utils import init_network_params

Array = np.ndarray


def relu(x):
    return np.maximum(0, x)


@jit
def predict(params: List[Tuple[Array]], image: Array) -> Array:
    activations = image
    for w, b in params[:-1]:
        out = np.dot(w, activations) + b
        activations = relu(out)
    w, b = params[-1]
    logits = np.dot(w, activations) + b
    return logits - jax.scipy.special.logsumexp(logits)


batch_predict = vmap(predict, in_axes=(None, 0))


def loss(params: List[Tuple[Array]], images: Array, targets: Array) -> float:
    preds = batch_predict(params, images)
    return -np.sum(preds * targets)


@jit
def update(params: List[Tuple[Array]], x: Array, y: Array) -> List[Array]:
    grads = grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def accuracy(params: List[Tuple[Array]], images: Array, targets: Array) -> Array:
    target_class = np.argmax(targets, axis=1)
    preds = batch_predict(params, images)
    predicted_class = np.argmax(preds, axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    data = MNIST()

    layers = [data.num_pixels, 512, 512, data.num_labels]
    param_scale = 0.1
    step_size = 1e-4
    num_epochs = 10
    params = init_network_params(layers, random.PRNGKey(0))

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in data.get_batches():
            params = update(params, x, y)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}: {epoch_time:0.2f} s")

        train_acc = accuracy(params, data.train_images, data.train_labels)
        print(f"Training set accuracy: {train_acc:0.5f}")

        test_acc = accuracy(params, data.test_images, data.test_labels)
        print(f"Test set accuracy: {test_acc:0.5f}")
