import jax
import jax.numpy as np
import jax.random as random

Array = np.ndarray


def sample_z(key: Array, mu: Array, logvar: Array) -> Array:
    return mu + np.exp(logvar / 2) * random.normal(key, mu.shape)


def bernoulli_llh(logits: Array, x: Array) -> float:
    return -np.sum(np.logaddexp(0, logits * np.where(x, -1, 1)))


def gaussian_kl(mu: Array, logvar: Array) -> float:
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
