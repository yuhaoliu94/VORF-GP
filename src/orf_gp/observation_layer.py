from abc import ABC
import numpy as np
from scipy.special import gammaln, logsumexp

from orf_gp.rff_layer import ORFGP, RFGP, MaxwellGP
from orf_gp.utils import symmetric_mult, force_symmetric

class ObservationLayer(ABC):
    def __init__(
            self,
            Dx,  # dimension of inputs
            J,  # number of random features
            Dy,  # dimension of outputs
            rff_config,
            sigma_l=1,  # prior kernel variance
            gamma=0.999, # usually 0.95-0.999, must be > 0.5
            rff_model='Orthogoanl',
    ):
        self.t = 0
        self.sigma_l = sigma_l
        self.gamma = gamma
        self.rff_config = rff_config
        self.rff_model = rff_model

        self.Dx = Dx
        self.J = J
        self.Dy = Dy

        self.M = None
        self.Sigma = None
        self.a = None
        self.b = None
        self.R = None
        self.noise_estimate = None

        self.RFF = None

    def initialize_structure(self):
        self.M = np.zeros((self.J, self.Dy))
        self.Sigma = np.eye(self.J) / self.sigma_l
        self.a = np.ones(self.Dy) * 2.0
        self.b = np.ones(self.Dy) * 0.5
        self.R = np.diag(self.b / self.a)
        self.noise_estimate = self.R * 1.0

        if self.rff_model == 'Orthogonal':
            self.RFF = ORFGP(self.Dx, self.J, self.Dy, **self.rff_config)
        elif self.rff_model == 'RBF':
            self.RFF = RFGP(self.Dx, self.J, self.Dy, **self.rff_config)
        elif self.rff_model == 'Maxwell':
            self.RFF = MaxwellGP(self.Dx, self.J, self.Dy, **self.rff_config)
        else:
            raise ValueError('Invalid rff_model: {}'.format(self.rff_model))

        self.RFF.initialize_structure()

        self.t = 0

    def reset_kalman_state(self):
        self.M = np.zeros((self.J, self.Dy))
        self.Sigma = np.eye(self.J) / self.sigma_l
        self.a = np.ones(self.Dy) * 2.0
        self.b = np.ones(self.Dy) * 0.5
        self.R = np.diag(self.b / self.a)
        self.noise_estimate = self.R * 1.0
        self.t = 0

    def predict(self, x_tilde_t):
        """x_tilde_t: input."""
        phi_t = self.RFF.get_phi(x_tilde_t)  # shape: (J,)

        s_t = 1.0 + symmetric_mult(phi_t, self.Sigma)
        s_t = np.maximum(s_t, 1e-6)

        return phi_t, s_t

    def predict_batch(self, X_tilde_t):
        """X_tilde_t: inputs."""
        phi_t = self.RFF.get_phi_batch(X_tilde_t)  # shape: (N, J)


        quad_form = np.einsum('nj,jk,nk->n', phi_t, self.Sigma, phi_t)

        s_t = 1.0 + quad_form
        s_t = np.maximum(s_t, 1e-6)  # Shape: (N,)

        return phi_t, s_t

    def predict_mc(self, x_tilde_t, MC):
        """x_tilde_t: input; MC: MC samples."""
        phi_samples = self.RFF.get_phi_mc(x_tilde_t, MC)

        y_samples = phi_samples @ self.M  # shape: (M, I)

        y_hat_mc = np.mean(y_samples, axis=0)

        var_epistemic = np.var(y_samples, axis=0)  # shape: (I,)

        prec_uncertainty_samples = np.einsum('mj,jk,mk->m', phi_samples, self.Sigma, phi_samples)
        var_aleatoric_scale = np.mean(prec_uncertainty_samples)

        noise_level = self.b / (self.a - 1)
        total_var = var_epistemic + (1.0 + var_aleatoric_scale) * noise_level

        return y_hat_mc, total_var

    def predict_batch_mc(self, X_tilde_t, y_test, MC, chunk_size=5):
        """X_tilde_t: inputs; y_test: targets; MC: MC samples; chunk_size: MC batch."""
        N, Dy = y_test.shape
        nu = 2 * self.a

        y_sum = np.zeros((N, Dy))
        y_sq_sum = np.zeros((N, Dy))
        aleatoric_sum = np.zeros(N)
        log_sum = np.full(N, -np.inf)

        processed = 0
        while processed < MC:
            chunk = min(chunk_size, MC - processed)

            phi_chunk = self.RFF.get_phi_batch_mc(X_tilde_t, chunk)
            y_chunk = phi_chunk @ self.M

            y_sum += y_chunk.sum(axis=0)
            y_sq_sum += (y_chunk ** 2).sum(axis=0)

            aleatoric_chunk = np.einsum('mnj,jk,mnk->mn', phi_chunk, self.Sigma, phi_chunk)
            aleatoric_sum += aleatoric_chunk.sum(axis=0)

            sigma_sq = (1.0 + aleatoric_chunk[:, :, np.newaxis]) * (self.b / self.a)
            e_sq = (y_test[np.newaxis, :, :] - y_chunk) ** 2
            c_t = (gammaln(self.a + 0.5) - gammaln(self.a) -
                   0.5 * np.log(np.pi * nu * sigma_sq))
            kernel_t = -(self.a + 0.5) * np.log1p(e_sq / (nu * sigma_sq))
            log_lik_chunk = np.sum(c_t + kernel_t, axis=2)  # (chunk, N)

            log_sum = np.logaddexp(log_sum, logsumexp(log_lik_chunk, axis=0))

            processed += chunk

        y_hat_mc = y_sum / MC
        y_sq_mean = y_sq_sum / MC
        var_epistemic = y_sq_mean - y_hat_mc ** 2  # (N, Dy)

        var_aleatoric_scale = aleatoric_sum / MC  # (N,)
        noise_level = self.b / (self.a - 1)
        total_var = var_epistemic + (1.0 + var_aleatoric_scale[:, np.newaxis]) * noise_level

        log_predictive_density = log_sum - np.log(MC)
        nll = -np.mean(log_predictive_density)


        mse = np.mean((y_test - y_hat_mc) ** 2)
        rmse = np.sqrt(mse)

        return y_hat_mc, total_var, nll, mse, rmse

    def get_nll_loss(self, y_t, phi_t, s_t):
        y_hat = self.M.T @ phi_t
        e_sq = (y_t - y_hat) ** 2

        nu = 2 * self.a
        sigma_sq = s_t * (self.b / self.a)
        c_t = gammaln(self.a + 0.5) - gammaln(self.a) - 0.5 * np.log(np.pi * nu * sigma_sq)

        kernel_t = -(self.a + 0.5) * np.log1p(e_sq / (nu * sigma_sq))

        log_likelihood = c_t + kernel_t

        return -np.sum(log_likelihood)

    def get_kl_sigma_loss_orf(self):
        d_sq = np.exp(2 * self.RFF.log_sigma)

        ln_d = self.RFF.log_sigma

        _lambda = (self.Dx - 1) / 2 + (self.J + 1) / self.Dx
        d_loss = np.sum(_lambda * d_sq - ln_d)

        return self.RFF.beta_kl * d_loss

    def get_kl_sigma_loss_rf(self):
        d_sq = np.exp(2 * self.RFF.log_sigma)

        ln_d_sq = 2 * self.RFF.log_sigma

        kl_sigma = 0.5 * np.sum(d_sq - ln_d_sq)

        return self.RFF.beta_kl * kl_sigma

    def get_kl_sigma_loss_maxwell(self):
        d_sq = np.exp(2 * self.RFF.log_sigma)

        ln_d = self.RFF.log_sigma

        _lambda = (self.Dx + 1) / 2
        d_loss = np.sum(_lambda * d_sq - ln_d)

        return self.RFF.beta_kl * d_loss

    def get_kl_sigma_loss(self):
        if self.rff_model == 'Orthogonal':
            return self.get_kl_sigma_loss_orf()
        elif self.rff_model == 'RBF':
            return self.get_kl_sigma_loss_rf()
        elif self.rff_model == 'Maxwell':
            return self.get_kl_sigma_loss_maxwell()
        else:
            raise ValueError('Invalid rff_model: {}'.format(self.rff_model))

    def get_kl_mu_loss_orf(self):
        gram = self.RFF.mu.T @ self.RFF.mu

        diff = gram - np.eye(self.J)
        ortho_penalty = 0.5 * np.sum(np.square(diff))

        return ortho_penalty * self.RFF.beta_orthogonality

    def get_kl_mu_loss_rf(self):
        mu_sq_sum = np.sum(np.square(self.RFF.mu))

        kl_mu = mu_sq_sum

        return self.RFF.beta_orthogonality * kl_mu

    def get_kl_mu_loss_maxwell(self):
        mu_sq_sum = np.sum(np.square(self.RFF.mu))

        kl_mu = (self.Dx + 1) / 2 * mu_sq_sum

        return self.RFF.beta_orthogonality * kl_mu

    def get_kl_mu_loss(self):
        if self.rff_model == 'Orthogonal':
            return self.get_kl_mu_loss_orf()
        elif self.rff_model == 'RBF':
            return self.get_kl_mu_loss_rf()
        elif self.rff_model == 'Maxwell':
            return self.get_kl_mu_loss_maxwell()
        else:
            raise ValueError('Invalid rff_model: {}'.format(self.rff_model))

    def get_all_losses(self, y_t, phi_t, s_t, y_hat_mc):
        nll = self.get_nll_loss(y_t, phi_t, s_t)
        kl_sigma = self.get_kl_sigma_loss()
        kl_mu = self.get_kl_mu_loss()
        mse_loss = np.mean((y_t - y_hat_mc) ** 2)

        total = nll + kl_sigma + kl_mu

        return total, nll, kl_sigma, kl_mu, mse_loss

    def get_grad_phi(self, y_t, phi_t, s_t):
        y_hat = self.M.T @ phi_t
        e_t = y_t - y_hat

        sigma_sq_t = s_t * (self.b / self.a)
        nu = 2 * self.a

        robust_weight = (nu + 1) / (nu * sigma_sq_t + e_t**2)
        grad_y_hat = - (robust_weight * e_t)

        grad_phi = self.M @ grad_y_hat

        return grad_phi

    def update(self, y_t, phi_t, s_t):
        y_hat = self.M.T @ phi_t
        e_t = y_t - y_hat  # Error vector, shape: (I,)

        tmp = self.Sigma @ phi_t
        k_t = tmp / s_t  # shape: (J,)

        new_Sigma = self.Sigma - np.outer(k_t, tmp)
        if self.t % 100 == 0:
            self.Sigma = force_symmetric(new_Sigma)

        self.M = self.M + np.outer(k_t, e_t)

        self.a = self.gamma * self.a + 0.5

        self.b = self.gamma * self.b + 0.5 * (np.square(e_t) / s_t)
        self.b = np.maximum(self.b, 1e-6)

        self.R = np.diag(self.b / self.a)
        self.noise_estimate = np.sqrt(self.R * s_t)