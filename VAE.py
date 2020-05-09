import time
from typing import List, Tuple

import jax
import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap

from data import MNIST
from nn_utils import init_network_params

Array = np.ndarray


def relu(x):
    return np.maximum(0, x)


def encode(params: List[Tuple[Array]], x: Array) -> Tuple[Array]:
    for w, b in params[:-2]:
        x = np.dot(w, x) + b
        x = relu(x)

    w_mu, b_mu = params[-2]
    mu = np.dot(w_mu, x) + b_mu

    w_logvar, b_logvar = params[-1]
    logvar = np.dot(w_logvar, x) + b_logvar

    return mu, logvar


batch_encode = vmap(encode, in_axes=[None, 0])


def sample_z(key: Array, mu: Array, logvar: Array) -> Array:
    return mu + np.exp(logvar / 2) * random.normal(key, mu.shape)


def decode(params: List[Tuple[Array]], z):
    for w, b in params[:-1]:
        z = np.dot(w, z) + b
        z = relu(z)

    w_last, b_last = params[-1]
    logits = np.dot(w_last, z) + b_last

    return logits - jax.scipy.special.logsumexp(logits)


batch_decode = vmap(decode, in_axes=[None, 0])


def bernoulli_llh(logits: Array, x: Array) -> float:
    return -np.sum(np.logaddexp(0, logits * np.where(x, -1, 1)))


def gaussian_kl(mu: Array, logvar: Array) -> float:
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))


@jit
def loss(params, r, images):
    enc_params, dec_params = params
    mu, logvar = batch_encode(enc_params, images)
    z = mu + np.exp(logvar / 2) * r
    logits = batch_decode(dec_params, z)
    sum_loss = -bernoulli_llh(logits, images) + gaussian_kl(mu, logvar)
    return sum_loss / len(images)


@jit
def GD_update(key, params, images, step_size=1e-4):
    enc_params, dec_params = params

    latent_dim = enc_params[-1][0].shape[0]
    r = random.normal(key, (len(images), latent_dim))

    enc_grad, dec_grad = grad(loss)(params, r, images)

    enc_params = [
        (w - dw * step_size, b - db * step_size)
        for (w, b), (dw, db) in zip(enc_params, enc_grad)
    ]
    dec = [
        (w - dw * step_size, b - db * step_size)
        for (w, b), (dw, db) in zip(dec_params, dec_grad)
    ]
    return enc_params, dec_params


if __name__ == "__main__":
    data = MNIST()

    latent_dim = 10
    enc_layers = [data.num_pixels, 512, 512, latent_dim]
    dec_layers = [latent_dim, 512, 512, data.num_pixels]

    key = random.PRNGKey(0)
    enc_key, z_key, dec_key = random.split(key, 3)
    enc_params = init_network_params(enc_layers, enc_key)
    enc_params += init_network_params(
        [512, latent_dim], key=random.split(enc_key)[1]
    )  # also add a layer for the logvar net
    dec_params = init_network_params(dec_layers, dec_key)

    num_epochs = 50
    for epoch in range(num_epochs):
        start_time = time.time()
        for images, _ in data.get_batches():
            z_key, vae_key, data_key = random.split(z_key, 3)
            binary_images = random.bernoulli(data_key, images / 255)
            enc_params, dec_params = GD_update(
                vae_key, (enc_params, dec_params), binary_images
            )
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: {epoch_time:0.2f} s")
