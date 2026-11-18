import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions_from_mc(y_hat_mc_history, total_var_history, actual_data, channels=None, last_steps=None,
                             show_ci=True):
    """y_hat_mc_history: predicted means; total_var_history: predicted variances; actual_data: observations; channels: output indices; last_steps: tail length; show_ci: 95% interval."""
    pred_mean = np.array(y_hat_mc_history)  # (T, Dy)
    pred_var = np.array(total_var_history)  # (T, Dy)
    pred_std = np.sqrt(pred_var)  # (T, Dy)

    T_total, total_channels = actual_data.shape

    if channels is None:
        target_channels = list(range(total_channels))
    elif isinstance(channels, int):
        target_channels = list(range(min(channels, total_channels)))
    else:
        target_channels = channels

    t_axis = np.arange(T_total)
    if last_steps is not None:
        pred_mean = pred_mean[-last_steps:]
        pred_std = pred_std[-last_steps:]
        actual_plot = actual_data[-last_steps:]
        t_axis = t_axis[-last_steps:]
    else:
        actual_plot = actual_data

    num_plots = len(target_channels)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for i, ch_idx in enumerate(target_channels):
        ax = axes[i]
        ax.plot(t_axis, actual_plot[:, ch_idx], label='Actual',
                color='black', linestyle='--', alpha=0.7)
        ax.plot(t_axis, pred_mean[:, ch_idx], label='Predicted Mean',
                color='blue', lw=1.5)

        if show_ci:
            lower = pred_mean[:, ch_idx] - 1.96 * pred_std[:, ch_idx]
            upper = pred_mean[:, ch_idx] + 1.96 * pred_std[:, ch_idx]
            ax.fill_between(t_axis, lower, upper, color='blue',
                            alpha=0.2, label='95% CI')

        ax.set_ylabel(f'Channel {ch_idx}')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper right', ncol=3)

    axes[-1].set_xlabel('Time Steps (t)')
    plt.tight_layout()
    plt.show()


def plot_rff_lr_history(history_list, param_name='mu', last_steps=None):
    """history_list: history; param_name: parameter; last_steps: tail length."""
    data = np.array(history_list)
    total_t = data.shape[0]
    t_axis = np.arange(total_t)

    if last_steps is not None:
        data = data[-last_steps:]
        t_axis = t_axis[-last_steps:]

    if data.ndim > 1:
        num_dim = data.shape[1]
        data_plot = data.reshape(-1, num_dim)
    else:
        num_dim = 1
        data_plot = data.reshape(-1, 1)

    plt.figure(figsize=(10, 5))
    for d in range(num_dim):
        plt.plot(t_axis, data_plot[:, d], label=f'Dim {d}', alpha=0.7)

    plt.title(f'Evolution of RFF Learning Rate: {param_name}')
    plt.xlabel('Online Steps (t)')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rff_parameters(history_list, param_type='l', last_steps=None):
    """history_list: history; param_type: parameter; last_steps: tail length."""
    data = np.array(history_list)
    if param_type == 'l':
        data = np.exp(data)
        ylabel = "Lengthscale ($l$)"
        title = "Evolution of RFF Lengthscales"
    elif param_type == 'b':
        ylabel = "Bias ($b$)"
        title = "Evolution of RFF Bias"
    elif param_type == 'mu':
        ylabel = "Frequency Mean ($\\mu$)"
        title = "Evolution of RFF Frequency Means"
    elif param_type == 'sigma':
        data = np.exp(data)
        ylabel = "Frequency Std ($\\sigma$)"
        title = "Evolution of RFF Frequency Standard Deviations"
    else:
        raise ValueError(f"Unknown param_type: {param_type}. Use 'l', 'b', 'mu', or 'sigma'.")

    total_t = data.shape[0]
    t_axis = np.arange(total_t)

    if last_steps is not None:
        data = data[-last_steps:]
        t_axis = t_axis[-last_steps:]

    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    num_dim = data.shape[1] if data.ndim > 1 else 1
    data_plot = data.reshape(-1, num_dim)

    plt.figure(figsize=(10, 5))
    for d in range(min(num_dim, 20)):
        plt.plot(t_axis, data_plot[:, d], label=f'Dim {d}', alpha=0.8)

    plt.title(title)
    plt.xlabel('Online Steps (t)')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_loss_evolution(loss_history, mva_step=50, last_steps=None):
    """mva_step: window; last_steps: tail length."""
    losses = np.array(loss_history)
    mse_phys = losses[:, 4]
    rmse_phys = np.sqrt(mse_phys)

    plot_data = np.column_stack([losses, rmse_phys])
    loss_names = ['Total Loss', 'NLL', 'KL Sigma', 'KL Mu', 'MSE', 'RMSE']

    total_t = losses.shape[0]
    t_axis = np.arange(total_t)

    if last_steps is not None:
        plot_data = plot_data[-last_steps:]
        t_axis = t_axis[-last_steps:]

    fig, axes = plt.subplots(len(loss_names), 1, figsize=(10, 3 * len(loss_names)), sharex=False)
    if len(loss_names) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data = plot_data[:, i]

        ax.plot(t_axis, data, color='gray', alpha=0.3, label='Raw')

        if len(data) >= mva_step:
            mva_data = pd.Series(data).rolling(window=mva_step, min_periods=1).mean().values
            line_color = 'blue' if i == len(loss_names) - 1 else 'red'
            ax.plot(t_axis, mva_data, color=line_color, lw=1.5, label=f'{mva_step}-step MVA')

        ax.set_ylabel(loss_names[i])
        ax.grid(True, linestyle='--', alpha=0.5)

        if i >= len(loss_names) - 2:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        if i == 0:
            ax.legend(loc='upper right')
            ax.set_title("Model Training Loss Evolution", fontsize=14)

        ax.set_xlabel('Online Steps (t)')
    plt.tight_layout()
    plt.show()