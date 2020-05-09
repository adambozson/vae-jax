from typing import Generator, Iterable, List, Tuple

import jax
import jax.numpy as np
import tensorflow_datasets as tfds

Array = np.ndarray


def one_hot(x: Array, k: int) -> Array:
    return np.array(x[:, None] == np.arange(k))

class MNIST():
    def __init__(self):
        mnist_data, info = tfds.load(name="mnist", batch_size=-1, with_info=True)
        mnist_data = tfds.as_numpy(mnist_data)

        self.num_labels = info.features["label"].num_classes
        h, w, c = info.features["image"].shape
        self.num_pixels = h * w * c

        train_images, train_labels = (
            mnist_data["train"]["image"],
            mnist_data["train"]["label"],
        )
        self.train_images = np.reshape(train_images, (-1, self.num_pixels))
        self.train_labels = one_hot(train_labels, self.num_labels)
        print(f"Train: {train_images.shape=}, {train_labels.shape=}")

        test_images, test_labels = mnist_data["test"]["image"], mnist_data["test"]["label"]
        self.test_images = np.reshape(test_images, (-1, self.num_pixels))
        self.test_labels = one_hot(test_labels, self.num_labels)
        print(f"Test: {test_images.shape=}, {test_labels.shape=}")


    def get_batches(self, batch_size=128) -> Generator[Tuple[Array], None, None]:
        num_batches = int(np.ceil(len(self.train_images) / batch_size))
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx >= len(self.train_images):
                end_idx = len(self.train_images) - 1
            yield self.train_images[start_idx:end_idx, ...], self.train_labels[start_idx:end_idx, ...]
