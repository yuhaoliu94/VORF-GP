from abc import ABC

import numpy as np
from orf_gp.config import GPConfig
from orf_gp.observation_layer import ObservationLayer


class OnlineGP(ABC):
    def __init__(
            self,
            config: GPConfig,
            monitor,
    ):
        self.t = 0
        self.config = config
        self.J = config.J
        self.Dx = config.Dx
        self.Dy = config.Dy

        self.observation_layer = None

        self.monitor = monitor

    def initialize_structure(self):
        rff_config = {
            'beta_kl': self.config.beta_kl,
            'beta_orthogonality': self.config.beta_ortho,
            'meta_learning_rate': self.config.meta_lr,
            'alpha_meta': self.config.alpha_meta,
            'fixed_step': self.config.fixed_step,
            'model_seed': self.config.model_seed,
            'init_gamma': self.config.init_gamma,
            'init_log_l': self.config.init_log_l,
            'init_sigma': self.config.init_sigma,
        }

        self.observation_layer = ObservationLayer(
            Dx=self.Dx, J=self.J, Dy=self.Dy,
            sigma_l=self.config.sigma_l,
            gamma=self.config.gamma_obs,
            rff_config=rff_config,
            rff_model=self.config.rff_model,
        )

        self.observation_layer.initialize_structure()

        self.t = 0

    def step(self, y_t, x_t, MC):
        """MC: MC samples."""
        phi_t, s_t = self.observation_layer.predict(x_t)
        y_hat_mc, total_var = self.observation_layer.predict_mc(x_t, MC)

        losses = self.observation_layer.get_all_losses(y_t, phi_t, s_t, y_hat_mc)
        self.monitor.check(self, losses)

        grad_phi = self.observation_layer.get_grad_phi(y_t, phi_t, s_t)
        self.observation_layer.RFF.update(x_t, grad_phi)

        self.observation_layer.update(y_t, phi_t, s_t)

        self.t += 1
        self.observation_layer.t = self.t
        self.observation_layer.RFF.t = self.t

        return losses, y_hat_mc, total_var

    def step_kalman_only(self, y_t, x_t, MC):
        """MC: MC samples."""
        phi_t, s_t = self.observation_layer.predict(x_t)
        y_hat_mc, total_var = self.observation_layer.predict_mc(x_t, MC)
        losses = self.observation_layer.get_all_losses(y_t, phi_t, s_t, y_hat_mc)
        self.monitor.check(self, losses)
        self.observation_layer.update(y_t, phi_t, s_t)
        self.t += 1
        self.observation_layer.t = self.t
        self.observation_layer.RFF.t = self.t
        return losses, y_hat_mc, total_var

    def reset_kalman(self):
        self.observation_layer.reset_kalman_state()
        self.t = 0
