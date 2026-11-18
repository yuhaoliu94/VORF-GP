import time
from copy import deepcopy

import numpy as np
from datetime import datetime


class OnlineMonitor:
    def __init__(self, log_interval=500, verbal=True):
        self.log_interval = log_interval
        self.verbal = verbal
        self.start_time = time.time()

        self.loss_buffer = []

    def check(self, model, losses):
        current_t = model.t
        self.loss_buffer.append(losses)

        if current_t % self.log_interval == 0:
            self._display_report(model, current_t)
            self.loss_buffer = []

    def _display_report(self, model, current_t):
        avg_losses = np.mean(self.loss_buffer, axis=0)
        total_loss = avg_losses[0]
        nll_loss = avg_losses[1]
        kl_sigma = avg_losses[2]
        kl_mu = avg_losses[3]
        mse = avg_losses[4]
        rmse = np.sqrt(mse)

        total_steps_processed = current_t
        elapsed = time.time() - self.start_time
        ips = total_steps_processed / elapsed if elapsed > 0 else 0
        wall_time = datetime.now().strftime("%H:%M:%S")

        if self.verbal:
            print(f"[{wall_time}] t: {current_t} | {ips:.2f} steps/s")
            print(f"  > Loss     | Total: {total_loss:.4f}  | NLL: {nll_loss:.4f}")
            print(f"  > Reg      | KL_Sig: {kl_sigma:.4f} | KL_Ortho: {kl_mu:.4f}")
            print(f"  > Accuracy | MSE: {mse:.4f}    | RMSE: {rmse:.4f}")
            print(f"  > Physics  | R_mean: {np.mean(model.observation_layer.R):.4f} | Noise_est_mean: {np.mean(model.observation_layer.noise_estimate):.4f}")
            print("-" * 90)
        else:
            print(f"t: {current_t} | RMSE: {rmse:.4f}  | NLL: {nll_loss:.4f}")


class OnlineMonitorValidate:
    def __init__(self, validation_x, validation_y, log_interval=500, verbal=True):
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.min_loss = np.inf
        self.best_model = None
        self.log_interval = log_interval
        self.verbal = verbal
        self.start_time = time.time()

        self.loss_buffer = []

    def check(self, model, losses):
        current_t = model.t
        self.loss_buffer.append(losses)

        if current_t % self.log_interval == 0:
            self._display_report(model, current_t)
            self.loss_buffer = []

    def _display_report(self, model, current_t):
        avg_losses = np.mean(self.loss_buffer, axis=0)
        total_loss = avg_losses[0]
        nll_loss = avg_losses[1]
        kl_sigma = avg_losses[2]
        kl_mu = avg_losses[3]
        mse = avg_losses[4]
        rmse = np.sqrt(mse)

        _, _, nll_std, _, rmse_std = model.observation_layer.predict_batch_mc(
            self.validation_x, self.validation_y, MC=100
        )

        if rmse_std < self.min_loss:
            self.min_loss = rmse_std
            self.best_model = deepcopy(model)

        total_steps_processed = current_t
        elapsed = time.time() - self.start_time
        ips = total_steps_processed / elapsed if elapsed > 0 else 0
        wall_time = datetime.now().strftime("%H:%M:%S")

        if self.verbal:
            print(f"[{wall_time}] t: {current_t} | {ips:.2f} steps/s")
            print(f"  > Loss     | Total: {total_loss:.4f}  | NLL: {nll_loss:.4f}")
            print(f"  > Reg      | KL_Sig: {kl_sigma:.4f} | KL_Ortho: {kl_mu:.4f}")
            print(f"  > Accuracy | MSE: {mse:.4f}    | RMSE: {rmse:.4f}")
            print(f"  > Physics  | R_mean: {np.mean(model.observation_layer.R):.4f} | Noise_est_mean: {np.mean(model.observation_layer.noise_estimate):.4f}")
            print("-" * 90)
        else:
            print(f"t: {current_t} | RMSE: {rmse:.4f}  | Validation NLL: {nll_std:.4f}  | NLL: {nll_loss:.4f}")