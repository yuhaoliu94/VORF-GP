from abc import ABC
import numpy as np
from orf_gp.distribution import Normal, Uniform, HDD_Sampler, Maxwell_Sampler
from orf_gp.utils import l2_normalize, dynamic_clip

class ORFGP(ABC):
    def __init__(
            self,
            Dx,  # Inputs
            J,  # Features
            Dy,  # Outputs
            beta_kl=1e-4,  # KL weight
            beta_orthogonality=1e-4,  # Orthogonality weight
            meta_learning_rate=None,  # Parameter learning rates
            alpha_meta = 1e-2,  # Hyper learning rate
            fixed_step=-1,  # Warmup steps
            model_seed=520,  # RNG seed
            init_gamma=None,
            init_log_l=None,
            init_sigma=None,
    ):
        self.t = 0

        self.beta_kl = beta_kl
        self.beta_orthogonality = beta_orthogonality
        self.fixed_step = fixed_step
        self.model_seed = model_seed
        self.init_gamma = init_gamma
        self.init_log_l = init_log_l
        self.init_sigma = init_sigma
        self.normal_generator = Normal(self.model_seed)

        self.alpha_meta = alpha_meta

        default_lr = {
            'mu': 1e-2,
            'b': 1e-2,
            'l': 1e-2,
            'sigma': 1e-2
        }

        if meta_learning_rate is None:
            lr_config = default_lr
        elif isinstance(meta_learning_rate, dict):
            lr_config = {**default_lr, **meta_learning_rate}
        else:
            lr_config = {k: meta_learning_rate for k in default_lr.keys()}

        self.lr_mu = lr_config['mu']
        self.lr_b = lr_config['b']
        self.lr_l = lr_config['l']
        self.lr_sigma = lr_config['sigma']

        self.Dx = Dx
        self.J = J
        self.Dy = Dy

        self.mu = None  # Mean of Omega: (D, J)
        self.log_sigma = None  # Log-std of Omega: (D, J)
        self.b = None  # Phase shift: (J,)
        self.log_l = None  # ARD Length-scale: (D,)

        self.grad_mu_old = None
        self.grad_b_old = None
        self.grad_l_old = None
        self.grad_sigma_old = None

        self.reparam_epsilon = None  # Stochastic noise for reparameterization

    def _effective_gamma_at_log_l_zero(self):
        mu_norm_sq = np.sum(np.square(self.mu), axis=0)
        sigma_norm_sq = np.sum(np.exp(2 * self.log_sigma), axis=0)
        return float(np.mean(mu_norm_sq + sigma_norm_sq) / (2.0 * self.Dx))

    def _initialize_log_l(self):
        if self.init_log_l is not None:
            return np.ones(self.Dx) * float(self.init_log_l)

        if self.init_gamma is not None:
            init_gamma = float(self.init_gamma)
            if init_gamma <= 0:
                raise ValueError("init_gamma must be positive.")
            current_gamma = self._effective_gamma_at_log_l_zero()
            init_log_l = -0.5 * np.log(init_gamma / current_gamma)
            return np.ones(self.Dx) * init_log_l

        return np.zeros(self.Dx)

    def _initialize_log_sigma(self, default_sigma):
        if self.init_sigma is None:
            sigma = float(default_sigma)
        else:
            sigma = float(self.init_sigma)
            if sigma <= 0:
                raise ValueError("init_sigma must be positive.")
        return np.ones((self.Dx, self.J)) * np.log(sigma)

    def initialize_structure(self):
        sampler = HDD_Sampler(self.model_seed)
        self.mu = sampler.generate_h_dpp_spectral_frequencies(self.J, self.Dx, num_steps=1000, sigma=0.01, epsilon=1e-6)
        self.mu = l2_normalize(self.mu)

        lambd = (self.Dx - 1) / 2 + (self.J + 1) / self.Dx
        sigma_star = np.sqrt(1.0 / (2 * lambd))
        self.log_sigma = self._initialize_log_sigma(sigma_star)

        self.log_l = self._initialize_log_l()

        uniform_generator = Uniform(self.model_seed)
        self.b = uniform_generator.sample_univariate(0, 2*np.pi, size=self.J)
        self.b = self.b % (2 * np.pi)

        self.grad_mu_old = np.zeros((self.Dx, self.J))
        self.grad_sigma_old = np.zeros((self.Dx, self.J))
        self.grad_b_old = np.zeros(self.J)
        self.grad_l_old = np.zeros(self.Dx)

        self.reparam_epsilon = self.normal_generator.sample_univariate(0, 1, (self.Dx, self.J))

        self.t = 0

    def get_Omega_scaled(self):
        Omega_norm = self.mu + np.exp(self.log_sigma) * self.reparam_epsilon
        inv_l = np.exp(-self.log_l)
        Omega_scaled = Omega_norm * inv_l[:, np.newaxis]

        return Omega_scaled

    def get_z(self, x_tilde_t):
        z = self.get_Omega_scaled().T @ x_tilde_t + self.b
        return z

    def get_phi(self, x_tilde_t):
        z = self.get_z(x_tilde_t)
        phi = np.cos(z) / np.sqrt(self.J)
        return phi

    def get_z_batch(self, X_tilde_t):
        """X_tilde_t: inputs."""
        z = X_tilde_t @ self.get_Omega_scaled() + self.b
        return z

    def get_phi_batch(self, X_tilde_t):
        """X_tilde_t: inputs."""
        z = self.get_z_batch(X_tilde_t)
        phi = np.cos(z) / np.sqrt(self.J)
        return phi

    def get_omega_mc(self, MC):
        eps = self.normal_generator.sample_univariate(0, 1, (MC, self.Dx, self.J))
        omega_norm = self.mu + np.exp(self.log_sigma) * eps
        inv_l = np.exp(-self.log_l)
        return omega_norm * inv_l[np.newaxis, :, np.newaxis]  # (MC, D, J)

    def get_z_mc(self, x_tilde_t, MC):
        """x_tilde_t: input; MC: MC samples."""
        omega_scaled = self.get_omega_mc(MC)

        z = np.einsum('mdj,d->mj', omega_scaled, x_tilde_t) + self.b

        return omega_scaled, z

    def get_phi_mc(self, x_tilde_t, MC):

        _, z = self.get_z_mc(x_tilde_t, MC)
        phi_samples = np.cos(z) / np.sqrt(self.J)

        return phi_samples

    def get_z_batch_mc(self, X_tilde_t, MC):
        """X_tilde_t: inputs; MC: MC samples."""
        omega_scaled = self.get_omega_mc(MC)

        z = np.einsum('mdj,nd->mnj', omega_scaled, X_tilde_t) + self.b

        return omega_scaled, z

    def get_phi_batch_mc(self, X_tilde_t, MC):

        _, z = self.get_z_batch_mc(X_tilde_t, MC)
        phi_samples = np.cos(z) / np.sqrt(self.J)

        return phi_samples

    def get_z_and_gz(self, x_tilde_t, grad_phi):
        """x_tilde_t: input."""
        z = self.get_z(x_tilde_t)
        g_z = grad_phi * (-np.sin(z) / np.sqrt(self.J))  # Shape: (J,)
        return z, g_z

    def get_mu_nll_gradient(self, x_tilde_t, g_z):
        """x_tilde_t: input."""
        inv_l = np.exp(-self.log_l)
        return np.outer(x_tilde_t * inv_l, g_z) # (D, J)

    def get_mu_kl_gradient(self):
        gram = self.mu.T @ self.mu
        gram = (gram + gram.T) / 2.0  # Force symmetry here
        diff = gram - np.eye(self.J)
        return 4.0 * (self.mu @ diff)

    def get_mu_gradient(self, x_tilde_t, g_z):
        nll_gradient = self.get_mu_nll_gradient(x_tilde_t, g_z)
        kl_gradient = self.get_mu_kl_gradient()
        all_gradient = nll_gradient + self.beta_orthogonality * kl_gradient
        inner_product = np.sum(self.mu * all_gradient, axis=0)

        all_gradient -= inner_product[np.newaxis, :] * self.mu
        return all_gradient, nll_gradient

    def get_log_sigma_nll_gradient(self, grad_mu_nll):
        sigma = np.exp(self.log_sigma)
        nll_grad = grad_mu_nll * self.reparam_epsilon * sigma
        return nll_grad

    def get_log_sigma_kl_gradient(self):
        sigma_2 = np.exp(2 * self.log_sigma)
        _lambda = (self.Dx - 1) / 2 + (self.J + 1 ) / self.Dx
        kl_grad = 2 * _lambda * sigma_2 - 1
        return kl_grad

    def get_log_sigma_gradient(self, grad_mu_nll):
        nll_grad = self.get_log_sigma_nll_gradient(grad_mu_nll)
        kl_grad = self.get_log_sigma_kl_gradient()

        return nll_grad + self.beta_kl * kl_grad  # (D, J)

    def get_b_gradient(self, g_z):
        return g_z # (J,)

    def get_log_l_nll_gradient(self, g_mu_nll):

        g_log_l = - np.sum(g_mu_nll * self.mu, axis=1)

        return g_log_l

    def get_log_l_gradient(self, g_mu_null):
        nll_grad = self.get_log_l_nll_gradient(g_mu_null)

        return nll_grad

    def update(self, x_tilde_t, grad_phi):
        """x_tilde_t: input."""
        z, g_z = self.get_z_and_gz(x_tilde_t, grad_phi)

        g_mu, g_mu_nll = self.get_mu_gradient(x_tilde_t, g_z)
        g_b = self.get_b_gradient(g_z)
        g_log_sigma = self.get_log_sigma_gradient(g_mu_nll)
        g_log_l = self.get_log_l_gradient(g_mu_nll)

        g_log_l = dynamic_clip(g_log_l, sigma_multiplier=3.0, min_threshold=1.0)
        g_log_sigma = dynamic_clip(g_log_sigma, sigma_multiplier=3.0, min_threshold=1.0)

        delta_l_raw = np.clip(self.alpha_meta * np.sum(g_log_l * self.grad_l_old), -0.5, 0.5)
        delta_sigma_raw = np.clip(self.alpha_meta * np.sum(g_log_sigma * self.grad_sigma_old), -0.5, 0.5)
        if self.t > self.fixed_step:
            delta_mu_raw = np.clip(self.alpha_meta * np.sum(g_mu * self.grad_mu_old), -0.5, 0.5)
            delta_b_raw = np.clip(self.alpha_meta * np.sum(g_b * self.grad_b_old), -0.5, 0.5)


        MIN_LR = 1e-7
        MAX_LR = 1.0
        self.lr_l = np.clip(self.lr_l * np.exp(delta_l_raw), MIN_LR, MAX_LR)
        self.lr_sigma = np.clip(self.lr_sigma * np.exp(delta_sigma_raw), MIN_LR, MAX_LR)
        if self.t > self.fixed_step:
            self.lr_mu = np.clip(self.lr_mu * np.exp(delta_mu_raw), MIN_LR, MAX_LR)
            self.lr_b = np.clip(self.lr_b * np.exp(delta_b_raw), MIN_LR, MAX_LR)

        self.log_l -= self.lr_l * g_log_l
        self.log_sigma -= self.lr_sigma * g_log_sigma
        if self.t > self.fixed_step:
            self.mu -= self.lr_mu * g_mu
            self.b -= self.lr_b * g_b

        self.log_l = np.clip(self.log_l, -5.0, 5.0)
        self.log_sigma = np.clip(self.log_sigma, -10.0, 2.0)
        if self.t > self.fixed_step:
            self.mu = l2_normalize(self.mu)  # Project back to unit sphere
            self.b %= (2 * np.pi)  # Maintain phase range

        self.grad_l_old = g_log_l
        self.grad_sigma_old = g_log_sigma
        if self.t > self.fixed_step:
            self.grad_mu_old = g_mu
            self.grad_b_old = g_b

        self.reparam_epsilon = self.normal_generator.sample_univariate(0, 1, (self.Dx, self.J))


class RFGP(ORFGP):

    def initialize_structure(self):
        self.mu = self.normal_generator.sample_univariate(0, 1, (self.Dx, self.J))

        self.log_sigma = self._initialize_log_sigma(1.0)

        self.log_l = self._initialize_log_l()

        uniform_generator = Uniform(self.model_seed)
        self.b = uniform_generator.sample_univariate(0, 2*np.pi, size=self.J)
        self.b = self.b % (2 * np.pi)

        self.grad_mu_old = np.zeros((self.Dx, self.J))
        self.grad_sigma_old = np.zeros((self.Dx, self.J))
        self.grad_b_old = np.zeros(self.J)
        self.grad_l_old = np.zeros(self.Dx)

        self.reparam_epsilon = self.normal_generator.sample_univariate(0, 1, (self.Dx, self.J))

        self.t = 0

    def get_mu_kl_gradient(self):
        return 2 * self.mu

    def get_mu_gradient(self, x_tilde_t, g_z):
        nll_gradient = self.get_mu_nll_gradient(x_tilde_t, g_z)
        kl_gradient = self.get_mu_kl_gradient()
        all_gradient = nll_gradient + self.beta_orthogonality * kl_gradient
        return all_gradient, nll_gradient

    def get_log_sigma_kl_gradient(self):
        sigma_2 = np.exp(2 * self.log_sigma)
        kl_grad = sigma_2 - 1
        return kl_grad

    def update(self, x_tilde_t, grad_phi):
        """x_tilde_t: input."""
        z, g_z = self.get_z_and_gz(x_tilde_t, grad_phi)

        g_mu, g_mu_nll = self.get_mu_gradient(x_tilde_t, g_z)
        g_b = self.get_b_gradient(g_z)
        g_log_sigma = self.get_log_sigma_gradient(g_mu_nll)
        g_log_l = self.get_log_l_gradient(g_mu_nll)

        g_log_l = dynamic_clip(g_log_l, sigma_multiplier=3.0, min_threshold=1.0)
        g_log_sigma = dynamic_clip(g_log_sigma, sigma_multiplier=3.0, min_threshold=1.0)

        delta_l_raw = np.clip(self.alpha_meta * np.sum(g_log_l * self.grad_l_old), -0.5, 0.5)
        delta_sigma_raw = np.clip(self.alpha_meta * np.sum(g_log_sigma * self.grad_sigma_old), -0.5, 0.5)
        if self.t > self.fixed_step:
            delta_mu_raw = np.clip(self.alpha_meta * np.sum(g_mu * self.grad_mu_old), -0.5, 0.5)
            delta_b_raw = np.clip(self.alpha_meta * np.sum(g_b * self.grad_b_old), -0.5, 0.5)


        MIN_LR = 1e-7
        MAX_LR = 1.0
        self.lr_l = np.clip(self.lr_l * np.exp(delta_l_raw), MIN_LR, MAX_LR)
        self.lr_sigma = np.clip(self.lr_sigma * np.exp(delta_sigma_raw), MIN_LR, MAX_LR)
        if self.t > self.fixed_step:
            self.lr_mu = np.clip(self.lr_mu * np.exp(delta_mu_raw), MIN_LR, MAX_LR)
            self.lr_b = np.clip(self.lr_b * np.exp(delta_b_raw), MIN_LR, MAX_LR)

        self.log_l -= self.lr_l * g_log_l
        self.log_sigma -= self.lr_sigma * g_log_sigma
        if self.t > self.fixed_step:
            self.mu -= self.lr_mu * g_mu
            self.b -= self.lr_b * g_b

        self.log_l = np.clip(self.log_l, -5.0, 5.0)
        self.log_sigma = np.clip(self.log_sigma, -10.0, 2.0)
        if self.t > self.fixed_step:
            self.b %= (2 * np.pi)  # Maintain phase range

        self.grad_l_old = g_log_l
        self.grad_sigma_old = g_log_sigma
        if self.t > self.fixed_step:
            self.grad_mu_old = g_mu
            self.grad_b_old = g_b

        self.reparam_epsilon = self.normal_generator.sample_univariate(0, 1, (self.Dx, self.J))

class MaxwellGP(ORFGP):

    def initialize_structure(self):
        sampler = Maxwell_Sampler(self.model_seed)
        self.mu = sampler.generate_spectral_frequencies(self.J, self.Dx)
        self.mu = l2_normalize(self.mu)

        lambd = (self.Dx + 1) / 2
        sigma_star = np.sqrt(1.0 / (2 * lambd))
        self.log_sigma = self._initialize_log_sigma(sigma_star)

        self.log_l = self._initialize_log_l()

        uniform_generator = Uniform(self.model_seed)
        self.b = uniform_generator.sample_univariate(0, 2 * np.pi, size=self.J)
        self.b = self.b % (2 * np.pi)

        self.grad_mu_old = np.zeros((self.Dx, self.J))
        self.grad_sigma_old = np.zeros((self.Dx, self.J))
        self.grad_b_old = np.zeros(self.J)
        self.grad_l_old = np.zeros(self.Dx)

        self.reparam_epsilon = self.normal_generator.sample_univariate(0, 1, (self.Dx, self.J))

        self.t = 0

    def get_mu_kl_gradient(self):
        return 0

    def get_log_sigma_kl_gradient(self):
        sigma_2 = np.exp(2 * self.log_sigma)
        _lambda = (self.Dx + 1) / 2
        kl_grad = 2 * _lambda * sigma_2 - 1
        return kl_grad
