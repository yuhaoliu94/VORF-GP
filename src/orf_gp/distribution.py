import numpy as np

from abc import ABC
from orf_gp.utils import get_random_state
from scipy.stats import chi

class Distribution(ABC):

    def __init__(self):
        self.random_state = None


class Normal(Distribution):

    def __init__(self, random_state: int = 520) -> None:
        super().__init__()

        self.random_state = get_random_state(random_state)

    def sample_univariate(self, mean, scale, size):
        return self.random_state.normal(mean, scale, size)


class Uniform(Distribution):

    def __init__(self, random_state: int = 520) -> None:
        super().__init__()

        self.random_state = get_random_state(random_state)

    def sample_univariate(self, a, b, size):
        return self.random_state.uniform(a, b, size)


class HDD_Sampler(Distribution):
    def __init__(self, random_state: int = 520):
        super().__init__()
        self.random_state = get_random_state(random_state)

    def sample_directional_repulsion(self, J, D, num_steps=1000, sigma=0.01, epsilon=1e-6):
        U = self.random_state.randn(D, J)
        U /= np.linalg.norm(U, axis=0)

        def compute_log_det(U_mat):
            K = U_mat.T @ U_mat + epsilon * np.eye(J)
            try:
                L = np.linalg.cholesky(K)
                return 2 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                return -np.inf

        current_log_det = compute_log_det(U)

        for step in range(num_steps):
            idx = self.random_state.randint(0, J)
            u_old = U[:, idx].copy()

            u_proposal = u_old + sigma * self.random_state.randn(D)
            u_proposal /= np.linalg.norm(u_proposal)

            U_proposal = U.copy()
            U_proposal[:, idx] = u_proposal
            new_log_det = compute_log_det(U_proposal)

            if np.log(self.random_state.rand()) < (new_log_det - current_log_det):
                U[:, idx] = u_proposal
                current_log_det = new_log_det

        return U

    def sample_radial_magnitudes(self, J, D):
        k = D + 2
        r_standard = chi.rvs(df=k, size=J, random_state=self.random_state)
        r = r_standard / np.sqrt(D + 1)
        return r

    def generate_h_dpp_spectral_frequencies(self, J, D, **kwargs):
        U = self.sample_directional_repulsion(J, D, **kwargs)
        r = self.sample_radial_magnitudes(J, D)
        return U * np.expand_dims(r, axis=0)


class Maxwell_Sampler:
    def __init__(self, random_state: int = 520):
        self.random_state = get_random_state(random_state)

    def sample_directional_uniform(self, J, D):
        """J: features; D: dimensions."""
        U = self.random_state.randn(D, J)
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        return U

    def sample_radial_magnitudes(self, J, D):
        """J: features; D: dimensions."""
        k = D + 2
        r_standard = chi.rvs(df=k, size=J, random_state=self.random_state)
        r = r_standard / np.sqrt(D + 1)
        return r

    def generate_spectral_frequencies(self, J, D):
        """J: features; D: dimensions."""
        U = self.sample_directional_uniform(J, D)
        r = self.sample_radial_magnitudes(J, D)
        return U * np.expand_dims(r, axis=0)