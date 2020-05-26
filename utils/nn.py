from typing import Iterable, List

import jax
import jax.numpy as np
from jax import random

Array = np.ndarray


def random_layer_params(m: int, n: int, key: Array, scale: float = 1e-1) -> Array:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes: Iterable[int], key: Array) -> List[Array]:
    keys = random.split(key, num=len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]
