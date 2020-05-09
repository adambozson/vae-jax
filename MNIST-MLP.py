import time
from typing import Generator, Iterable, List, Tuple

import jax
import jax.numpy as np
import tensorflow_datasets as tfds
from jax import grad, jit, random, vmap

Array = np.ndarray

# Data loading

def one_hot(x: Array, k: int) -> Array:
    return np.array(x[:, None] == np.arange(k))


mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)

num_labels = info.features["label"].num_classes
h, w, c = info.features["image"].shape
num_pixels = h * w * c

train_images, train_labels = mnist_data["train"]["image"], mnist_data["train"]["label"]
train_images = np.reshape(train_images, (-1, num_pixels))
train_labels = one_hot(train_labels, num_labels)
print(f"Train: {train_images.shape=}, {train_labels.shape=}")

test_images, test_labels = mnist_data["test"]["image"], mnist_data["test"]["label"]
test_images = np.reshape(test_images, (-1, num_pixels))
test_labels = one_hot(test_labels, num_labels)
print(f"Test: {test_images.shape=}, {test_labels.shape=}")

# Model definition

def random_layer_params(m: int, n: int, key: Array, scale: float = 1e-2) -> Array:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes: Iterable[int], key: Array) -> List[Array]:
    keys = random.split(key, num=len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    
layers = [num_pixels, 512, 512, num_labels]
param_scale = 0.1
step_size = 1e-4
num_epochs = 10
batch_size = 128
params = init_network_params(layers, random.PRNGKey(0))

def relu(x):
    return np.maximum(0, x)

@jit
def predict(params: List[Array], image: Array) -> Array:
    activations = image
    for w, b in params[:-1]:
        out = np.dot(w, activations) + b
        activations = relu(out)
    w, b = params[-1]
    logits = np.dot(w, activations) + b
    return logits - jax.scipy.special.logsumexp(logits)

batch_predict = vmap(predict, in_axes=(None, 0))

def loss(params: List[Array], images: Array, targets: Array) -> float:
    preds = batch_predict(params, images)
    return -np.sum(preds*targets)

@jit
def update(params: List[Array], x: Array, y: Array) -> List[Array]:
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

def accuracy(params: List[Array], images: Array, targets: Array) -> Array:
    target_class = np.argmax(targets, axis=1)
    preds = batch_predict(params, images)
    predicted_class = np.argmax(preds, axis=1)
    return np.mean(predicted_class == target_class)

# Training loop

def get_batches() -> Generator[Tuple[Array], None, None]:
    num_batches = int(np.ceil(len(train_images) / batch_size))
    for i in range(num_batches):    
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        if end_idx >= len(train_images):
            end_idx = len(train_images) - 1
        yield train_images[start_idx:end_idx, ...], train_labels[start_idx:end_idx, ...]

for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in get_batches():
        params = update(params, x, y)
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1}: {epoch_time:0.2f} s")

    train_acc = accuracy(params, train_images, train_labels)
    print(f"Training set accuracy: {train_acc:0.5f}")
    
    test_acc = accuracy(params, test_images, test_labels)
    print(f"Test set accuracy: {test_acc:0.5f}")