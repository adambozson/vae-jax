import os
import time

import jax
import jax.numpy as np
import jax.random as random
from jax import grad, jit
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, FanOut, Relu

from utils.data import MNIST, input_output_figure
from utils.stats import bernoulli_llh, gaussian_kl, sample_z

Array = np.ndarray


def elbo(key, params, images):
    enc_params, dec_params = params
    mu, logvar = encode(enc_params, images)
    z = sample_z(key, mu, logvar)
    logits = decode(dec_params, z)
    elbo = bernoulli_llh(logits, images) - gaussian_kl(mu, logvar)
    return elbo, logits


if __name__ == "__main__":
    data = MNIST()
    latent_dim = 10

    encoder_init, encode = stax.serial(
        Dense(512),
        Relu,
        Dense(512),
        Relu,
        FanOut(2),
        stax.parallel(Dense(latent_dim), Dense(latent_dim)),
    )

    decoder_init, decode = stax.serial(
        Dense(512), Relu, Dense(512), Relu, Dense(data.num_pixels)
    )

    step_size = 1e-3
    num_epochs = 100
    batch_size = 128

    opt_init, opt_update, get_params = optimizers.nesterov(step_size, mass=0.9)

    # Initialisation
    key = random.PRNGKey(0)
    enc_init_key, dec_init_key, key = random.split(key, 3)
    _, enc_init_params = encoder_init(enc_init_key, (batch_size, data.num_pixels))
    _, dec_init_params = decoder_init(dec_init_key, (batch_size, latent_dim))
    init_params = (enc_init_params, dec_init_params)

    opt_state = opt_init(init_params)

    @jit
    def update(i, key, opt_state, images):
        loss = lambda p: -elbo(key, p, images)[0] / len(images)
        g = grad(loss)(get_params(opt_state))
        return opt_update(i, g, opt_state)

    @jit
    def evaluate(key, params, images):
        data_key, z_key = random.split(key)
        images_sample = random.bernoulli(data_key, images / 255)
        sum_elbo, logits = elbo(z_key, params, images_sample)
        mean_image = np.exp(-np.logaddexp(0, -logits))
        return sum_elbo / len(images), mean_image

    for epoch in range(num_epochs):
        start_time = time.time()

        for i, (images, _) in enumerate(data.get_batches(batch_size)):
            data_key, train_key, key = random.split(key, 3)
            binary_images = random.bernoulli(data_key, images / 255)
            opt_state = update(i, train_key, opt_state, binary_images)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}: {epoch_time:0.2f} s")

        train_key, test_key, key = random.split(key, 3)
        train_elbo, _ = evaluate(train_key, get_params(opt_state), data.train_images)
        print(f"Training set ELBO: {train_elbo}")
        test_elbo, test_outputs = evaluate(
            test_key, get_params(opt_state), data.test_images
        )
        print(f"Test set ELBO: {test_elbo}")

        n_images = 10
        out_dir = "images"
        path = os.path.join(out_dir, f"{epoch+1}.png")
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass  # Allow overwriting
        input_output_figure(data.test_images[:n_images], test_outputs[:n_images], path)
