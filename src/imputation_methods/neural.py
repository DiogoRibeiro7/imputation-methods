"""Neural-network imputers."""

from __future__ import annotations

import logging
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit
from sklearn.neural_network import MLPRegressor

from ._deprecation import renamed_parameters
from ._utils import OnError, check_on_error, raise_or_fall_back
from .base import BaseImputer, ImputationError
from .statistical import MeanImputer

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class AutoencoderImputer(BaseImputer):
    """Impute missing values using a simple autoencoder."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (10,),
        max_iter: int = 200,
        random_state: int | None = None,
        on_error: OnError = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            hidden_layer_sizes: Architecture of the ``MLPRegressor`` used as
                the autoencoder.
            max_iter: Maximum training iterations.
            random_state: Random seed controlling network initialization.
            on_error: What to do if the model can't be fitted: ``"raise"`` an
                :class:`~imputation_methods.ImputationError`, or ``"fallback"``
                to use mean imputation. The default, ``None``, falls back with a
                ``FutureWarning``; it will change to ``"raise"`` in 1.0.0.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=random_state,
        )
        self.on_error = check_on_error(on_error)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using an autoencoder reconstruction.

        Args:
            df: Dataframe with missing values.

        Returns:
            Dataframe with imputed values predicted by the autoencoder.

        Raises:
            RuntimeError: If autoencoder training or prediction fails.
        """
        df = self._ensure_numeric(df)
        try:
            filled = df.fillna(df.mean())
            # Handle case where mean might be NaN (all values missing)
            if filled.isna().any().any():
                filled = filled.fillna(0)
            target = filled.to_numpy()
            # scikit-learn expects a 1-D target when there is a single column.
            self._model.fit(filled, target.ravel() if target.shape[1] == 1 else target)
            reconstructed = pd.DataFrame(
                self._model.predict(filled),
                columns=df.columns,
                index=df.index,
            )
            return df.where(~df.isna(), reconstructed)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise_or_fall_back(
                self.on_error,
                imputer=type(self).__name__,
                error=e,
                fallback="mean imputation",
            )
            return MeanImputer().impute(df)
        except Exception as e:
            raise ImputationError(f"{type(self).__name__} failed: {e}") from e


class _MLP:
    """Fully connected network with ReLU hidden layers and a sigmoid output."""

    def __init__(self, sizes: list[int], rng: np.random.Generator) -> None:
        self.params: list[FloatArray] = []
        for fan_in, fan_out in pairwise(sizes):
            # Xavier-style initialisation, as in the GAIN reference implementation.
            std = np.sqrt(2.0 / fan_in)
            self.params.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.params.append(np.zeros(fan_out))

    @property
    def n_layers(self) -> int:
        return len(self.params) // 2

    def forward(self, x: FloatArray) -> tuple[FloatArray, list[FloatArray]]:
        """Return the output probabilities and the activations of every layer."""
        activations = [x]
        h = x
        for layer in range(self.n_layers):
            z = h @ self.params[2 * layer] + self.params[2 * layer + 1]
            h = expit(z) if layer == self.n_layers - 1 else np.maximum(z, 0.0)
            activations.append(h)
        return h, activations

    def backward(
        self, activations: list[FloatArray], grad_logits: FloatArray
    ) -> tuple[list[FloatArray], FloatArray]:
        """Backpropagate the gradient of the loss w.r.t. the output logits.

        Returns:
            Gradients for ``params`` (same order) and for the network input.
        """
        grads: list[FloatArray] = [np.empty(0)] * len(self.params)
        delta = grad_logits
        for layer in reversed(range(self.n_layers)):
            layer_input = activations[layer]
            grads[2 * layer] = layer_input.T @ delta
            grads[2 * layer + 1] = delta.sum(axis=0)
            delta = delta @ self.params[2 * layer].T
            if layer > 0:
                delta = delta * (layer_input > 0)
        return grads, delta


class _Adam:
    """Adam optimiser updating a list of arrays in place."""

    def __init__(
        self,
        params: list[FloatArray],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        self.step_count += 1
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        for param, grad, m, v in zip(params, grads, self.m, self.v, strict=True):
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad**2
            param -= self.learning_rate * (m / bias1) / (np.sqrt(v / bias2) + self.eps)


def _gain_impute(
    data: FloatArray,
    observed: FloatArray,
    *,
    batch_size: int,
    hint_rate: float,
    alpha: float,
    max_iter: int,
    learning_rate: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Train GAIN on ``data`` scaled to [0, 1] and return the completed matrix.

    ``observed`` is 1 where a value is observed and 0 where it is missing; missing
    entries of ``data`` may hold any value.
    """
    n_samples, n_features = data.shape
    generator = _MLP([2 * n_features, n_features, n_features, n_features], rng)
    discriminator = _MLP([2 * n_features, n_features, n_features, n_features], rng)
    g_optimizer = _Adam(generator.params, learning_rate)
    d_optimizer = _Adam(discriminator.params, learning_rate)
    batch = min(batch_size, n_samples)

    for _ in range(max_iter):
        idx = rng.permutation(n_samples)[:batch]
        mask = observed[idx]
        noise = rng.uniform(0.0, 0.01, size=(batch, n_features))
        x = mask * data[idx] + (1.0 - mask) * noise
        hint = mask * (rng.random((batch, n_features)) < hint_rate)
        g_input = np.hstack([x, mask])
        n_cells = batch * n_features

        # Discriminator: predict which entries were observed.
        g_out, _ = generator.forward(g_input)
        x_hat = mask * x + (1.0 - mask) * g_out
        d_prob, d_acts = discriminator.forward(np.hstack([x_hat, hint]))
        d_grads, _ = discriminator.backward(d_acts, (d_prob - mask) / n_cells)
        d_optimizer.step(discriminator.params, d_grads)

        # Generator: fool the discriminator on missing entries and reconstruct
        # the observed ones.
        g_out, g_acts = generator.forward(g_input)
        x_hat = mask * x + (1.0 - mask) * g_out
        d_prob, d_acts = discriminator.forward(np.hstack([x_hat, hint]))
        adversarial_logits = -(1.0 - mask) * (1.0 - d_prob) / n_cells
        _, grad_d_input = discriminator.backward(d_acts, adversarial_logits)
        grad_g_out = grad_d_input[:, :n_features] * (1.0 - mask)
        observed_fraction = max(float(mask.mean()), 1e-8)
        grad_g_out += alpha * 2.0 * mask * (g_out - x) / (n_cells * observed_fraction)
        g_grads, _ = generator.backward(g_acts, grad_g_out * g_out * (1.0 - g_out))
        g_optimizer.step(generator.params, g_grads)

    noise = rng.uniform(0.0, 0.01, size=data.shape)
    x = observed * data + (1.0 - observed) * noise
    g_out, _ = generator.forward(np.hstack([x, observed]))
    completed: FloatArray = observed * data + (1.0 - observed) * g_out
    return completed


class GAINImputer(BaseImputer):
    """Impute missing values with Generative Adversarial Imputation Nets (GAIN).

    A generator network fills in the missing entries, while a discriminator
    tries to tell which entries were observed and which were imputed. A hint
    vector reveals part of the missingness mask to the discriminator, and a
    reconstruction loss on the observed entries keeps the generator faithful to
    the data. Columns are min-max scaled to [0, 1] for training. Both networks
    have two hidden layers as wide as the number of columns and are trained
    with Adam, implemented directly on NumPy.

    GAIN needs a reasonable amount of data to train; on small datasets simpler
    methods such as :class:`~imputation_methods.MICEImputer` are usually more
    accurate. Columns with no observed values are left untouched.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imputation_methods import GAINImputer
        >>> df = pd.DataFrame(
        ...     {"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, np.nan, 6.0, 8.0]}
        ... )
        >>> imputed = GAINImputer(max_iter=100, random_state=0).impute(df)
        >>> bool(imputed.notna().all().all())
        True

    References:
        Yoon, J., Jordon, J., & van der Schaar, M. (2018). GAIN: Missing data
        imputation using generative adversarial nets. ICML, 5689-5698.
    """

    @renamed_parameters(iterations="max_iter")
    def __init__(
        self,
        batch_size: int = 128,
        hint_rate: float = 0.9,
        alpha: float = 100.0,
        max_iter: int = 10000,
        learning_rate: float = 0.001,
        random_state: int | None = None,
    ) -> None:
        """Initialize the imputer.

        The defaults are those of the reference implementation.

        Args:
            batch_size: Rows per training step (capped at the number of rows).
            hint_rate: Probability that each entry of the missingness mask is
                revealed to the discriminator.
            alpha: Weight of the reconstruction loss on observed entries.
            max_iter: Number of training steps.
            learning_rate: Adam learning rate for both networks.
            random_state: Seed for initialisation, batching, noise and hints.

        Raises:
            ValueError: If an argument is out of range.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0.0 <= hint_rate <= 1.0:
            raise ValueError(f"hint_rate must be in [0, 1], got {hint_rate}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train GAIN on ``df`` and fill its missing values.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe. Observed values are unchanged.
        """
        df = self._ensure_numeric(df)
        result = df.astype(float)
        modelled = result.columns[result.notna().any()]
        if modelled.empty or not result[modelled].isna().any().any():
            return result

        values = result[modelled].to_numpy(dtype=float, na_value=np.nan)
        observed = ~np.isnan(values)
        minimum = np.nanmin(values, axis=0)
        # As in the reference implementation, the small offset keeps constant
        # columns at their value instead of letting the generator move them.
        value_range = np.nanmax(values, axis=0) - minimum + 1e-6
        scaled = np.where(observed, (values - minimum) / value_range, 0.0)

        completed = _gain_impute(
            scaled,
            observed.astype(float),
            batch_size=self.batch_size,
            hint_rate=self.hint_rate,
            alpha=self.alpha,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            rng=np.random.default_rng(self.random_state),
        )
        result[modelled] = np.where(observed, values, completed * value_range + minimum)
        return result
