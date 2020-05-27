import time
import os
from typing import Any, Sequence, Tuple

import haiku as hk
from haiku.nets import MLP

import jax
import jax.numpy as np
import jax.experimental.optimizers as optimizers
import jax.random as random

from utils.data import MNIST, input_output_figure
from utils.stats import bernoulli_llh, gaussian_kl

Array = np.ndarray
OptState = Any


class Encoder(hk.Module):
    def __init__(
        self, hidden_layers: Sequence[int] = [512, 512], latent_size: int = 10
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        h = MLP(self.hidden_layers, activate_final=True)(x)
        mean = hk.Linear(self.latent_size)(h)
        logvar = hk.Linear(self.latent_size)(h)
        return mean, logvar


class VAE(hk.Module):
    def __init__(self, latent_size: int = 10, output_size: int = 28 * 28):
        super().__init__()
        self.latent_size = latent_size
        self.output_size = output_size

    def __call__(self, x: Array) -> Tuple[Array, Array, Array]:
        mean, logvar = Encoder(latent_size=self.latent_size)(x)
        z = mean + np.exp(logvar / 2) * random.normal(hk.next_rng_key(), mean.shape)
        logits = MLP([512, 512, self.output_size])(z)
        return logits, mean, logvar


if __name__ == "__main__":
    data = MNIST()
    latent_size = 10
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 100

    def fwd(x):
        model = VAE(latent_size, data.num_pixels)
        return model(x)

    model = hk.transform(fwd, apply_rng=True)
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)

    keys = hk.PRNGSequence(0)

    params = model.init(next(keys), np.empty((batch_size, data.num_pixels)))
    opt_state = opt_init(params)

    @jax.jit
    def loss(params: hk.Params, key: random.PRNGKey, data: Array) -> Array:
        logits, mean, logvar = model.apply(params, key, data)
        llh = bernoulli_llh(logits, data)
        kl = gaussian_kl(mean, logvar)
        return (-llh + kl) / len(data)  # -ELBO

    @jax.jit
    def update(
        i: int, key: random.PRNGKey, opt_state: OptState, batch: Array
    ) -> OptState:
        g = jax.grad(loss)(get_params(opt_state), key, batch)
        return opt_update(i, g, opt_state)

    for epoch in range(num_epochs):
        start_time = time.time()

        for i, (batch, _) in enumerate(data.get_batches()):
            images = random.bernoulli(next(keys), batch / 255)
            opt_state = update(i, next(keys), opt_state, images)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: {epoch_time:0.2f} s")

        params = get_params(opt_state)
        train_sample = random.bernoulli(next(keys), data.train_images / 255)
        train_loss = loss(params, next(keys), train_sample)
        print(f"Training set loss: {train_loss}")

        test_sample = random.bernoulli(next(keys), data.test_images / 255)
        test_loss = loss(params, next(keys), test_sample)
        print(f"Training set loss: {test_loss}")

        test_outputs = jax.nn.sigmoid(model.apply(params, next(keys), test_sample)[0])

        n_images = 10
        out_dir = "images"
        path = os.path.join(out_dir, f"{epoch+1}.png")
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass  # Allow overwriting
        input_output_figure(data.test_images[:n_images], test_outputs[:n_images], path)
