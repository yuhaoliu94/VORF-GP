import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


def _fwht(x):
    """x: input."""
    x = x.copy()
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    return x / np.sqrt(n)


def transform_all_fast(X, transformer, dense_output=True):
    """X: inputs; transformer: fitted map; dense_output: dense flag."""
    check_is_fitted(transformer, "random_weights_")
    X = check_array(X, accept_sparse=False)

    n_samples, n_features = X.shape
    n_stacks = transformer.random_weights_.shape[0]

    if hasattr(transformer, 'random_fourier') and transformer.random_fourier:
        if hasattr(transformer, 'use_offset') and not transformer.use_offset:
            n_features_padded = transformer.n_components // (2 * n_stacks)
        else:
            n_features_padded = transformer.n_components // n_stacks
    else:
        n_features_padded = transformer.n_components // n_stacks

    Z = np.zeros((n_samples, transformer.n_components))

    for sample_idx in range(n_samples):
        x = X[sample_idx]  # shape: (n_features,)

        for stack_idx in range(n_stacks):
            weights = transformer.random_weights_[stack_idx]  # shape: (n_features + 2*n_features_padded,)

            d1 = weights[:n_features]
            d2 = weights[n_features:n_features + n_features_padded]
            d3 = weights[n_features + n_features_padded:]

            cache = np.zeros(n_features_padded)
            cache[:n_features] = d1 * x

            # Step 2: H * (D1 * x) - Walsh-Hadamard Transform
            cache = _fwht(cache)

            # Step 3: D2 * (H * D1 * x)
            cache = d2 * cache

            # Step 4: H * (D2 * H * D1 * x)
            cache = _fwht(cache)

            # Step 5: D3 * (H * D2 * H * D1 * x)
            cache = d3 * cache

            # Step 6: H * (D3 * H * D2 * H * D1 * x)
            cache = _fwht(cache)

            output_segment = cache * np.sqrt(n_features_padded)

            start_idx = stack_idx * n_features_padded
            end_idx = start_idx + n_features_padded
            Z[sample_idx, start_idx:end_idx] = output_segment

    if hasattr(transformer, 'random_fourier') and transformer.random_fourier:
        gamma = getattr(transformer, 'gamma', 0.5)
        scale_weight = np.sqrt(2 * gamma)

        if hasattr(transformer, 'use_offset') and transformer.use_offset:
            # use_offset=True: cos(scale_weight * Z + offset) * sqrt(2)
            Z = np.cos(scale_weight * Z + transformer.random_offset_) * np.sqrt(2)
        else:
            # use_offset=False: [cos, sin] * sqrt(2)
            index_offset = transformer.n_components // 2
            Z_cos = np.cos(scale_weight * Z[:, :index_offset]) * np.sqrt(2)
            Z_sin = np.sin(scale_weight * Z[:, :index_offset]) * np.sqrt(2)
            Z = np.hstack([Z_cos, Z_sin])

    Z = Z / np.sqrt(transformer.n_components)

    return Z
