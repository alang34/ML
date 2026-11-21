import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from diagnostics import LogCol, ValLogCol
from config import params


def render_training_monitor(diagnoser, save_path, residual, latest_fields):
    data = diagnoser._filled()
    if data.shape[0] == 0:
        return

    epochs = data[:, LogCol.EPOCH]
    val_data = diagnoser._filled_val()
    pde = epochs > params.PRETRAIN_EPOCHS

    fig = plt.figure(figsize=(20, 24))
    grid = fig.add_gridspec(6, 3)
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(3)]
                     for row in range(5)])
    sampling_grid = grid[5, :].subgridspec(1, 4, wspace=.4)
    sampling_axes = [fig.add_subplot(sampling_grid[0, col]) for col in range(4)]

    ax = axes[0, 0]
    ax.plot(epochs, data[:, LogCol.U_LOSS] + data[:, LogCol.V_LOSS] + data[:, LogCol.W_LOSS] + data[:,LogCol.P_LOSS], label="data (raw)")
    weighted_pde = data[:, LogCol.LAMBDA_PDE] * data[:, LogCol.PDE_LOSS]
    weighted_boundary = data[:, LogCol.WEIGHTED_BOUNDARY_LOSS]
    weighted_vorticity = data[:, LogCol.WEIGHTED_VORTICITY_LOSS]
    ax.plot(epochs, data[:, LogCol.TOTAL_LOSS] - weighted_pde - weighted_boundary - weighted_vorticity,
            label="data (scaled objective)")
    ax.plot(epochs, weighted_boundary, label="boundary (weighted)")
    if weighted_vorticity.any():
        ax.plot(epochs, weighted_vorticity, label="wake vorticity (weighted)")
    ax.plot(epochs, data[:, LogCol.PDE_LOSS], label="pde (raw)")
    ax.plot(epochs, weighted_pde, label="pde (weighted)")
    ax.plot(epochs, data[:, LogCol.TOTAL_LOSS], label="total (optimized)")
    ax.set_yscale("log"); ax.set_title("Training objectives"); ax.set_xlabel("epoch"); ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, data[:, LogCol.U_LOSS], label="u")
    ax.plot(epochs, data[:, LogCol.V_LOSS], label="v")
    ax.plot(epochs, data[:, LogCol.W_LOSS], label="w")
    ax.plot(epochs, data[:, LogCol.P_LOSS], label="p")
    ax.plot(epochs, data[:, LogCol.PDE_LOSS], label = "pde (raw)")
    ax.plot(epochs, data[:, LogCol.PDE_LOSS]*data[:, LogCol.LAMBDA_PDE], label = "pde (weighted)")
    ax.set_yscale("log"); ax.set_title("Component data losses"); ax.set_xlabel("epoch"); ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, data[:, LogCol.LAMBDA_U], label="lambda_u")
    ax.plot(epochs, data[:, LogCol.LAMBDA_V], label="lambda_v")
    ax.plot(epochs, data[:, LogCol.LAMBDA_W], label="lambda_w")
    ax.plot(epochs, data[:, LogCol.LAMBDA_P], label="lambda_p")
    ax.plot(epochs, data[:, LogCol.LAMBDA_PDE], label="lambda_pde")
    ax.set_yscale("log"); ax.set_title("Adaptive lambdas"); ax.set_xlabel("epoch"); ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, data[:, LogCol.COV], label="cov")
    ax.plot(epochs, data[:, LogCol.STD], label="std")
    ax.plot(epochs, data[:, LogCol.MEAN], label="mean")
    ax.plot(epochs, data[:, LogCol.SLOPE], label="slope")
    relative_slope = (data[:, LogCol.SLOPE] / data[:, LogCol.MEAN])
    ax.plot(epochs, relative_slope, label="slope/mean")
    ax.set_yscale("log")
    ax.set_title("PDE residual stability; Avg over last 20: ")
    ax.set_xlabel("epoch")
    ax.legend()

    ax = axes[1, 1]
    if val_data.shape[0] > 0:
        vepochs = val_data[:, ValLogCol.EPOCH]
        ax.plot(vepochs, val_data[:, ValLogCol.L2_U], label=f"u ({val_data[-1, ValLogCol.L2_U]*100:.1f}%)")
        ax.plot(vepochs, val_data[:, ValLogCol.L2_V], label=f"v ({val_data[-1, ValLogCol.L2_V]*100:.1f}%)")
        ax.plot(vepochs, val_data[:, ValLogCol.L2_W], label=f"w ({val_data[-1, ValLogCol.L2_W]*100:.1f}%)")
        ax.plot(vepochs, val_data[:, ValLogCol.L2_P], label=f"p ({val_data[-1, ValLogCol.L2_P]*100:.1f}%)")
        ax.set_yscale("log"); ax.legend()
    ax.set_title("Validation relative L2"); ax.set_xlabel("epoch")

    ax = axes[1, 2]
    if val_data.shape[0] > 0:
        vepochs = val_data[:, ValLogCol.EPOCH]
        ax.plot(vepochs, val_data[:, ValLogCol.WAKE_ERROR], label="wake")
        ax.plot(vepochs, val_data[:, ValLogCol.FREESTREAM_ERROR], label="freestream")
        ax.set_yscale("log"); ax.legend()
    ax.set_title(f"Wake vs. freestream error; ({val_data[-1, ValLogCol.WAKE_ERROR]/val_data[-1, ValLogCol.FREESTREAM_ERROR]}) is the W/F ratio"); ax.set_xlabel("epoch")
    
    max_points = 5000
    
    ax = axes[2, 0]
    latest_val_extras = diagnoser.latest_val_extras
    pairs = (latest_val_extras["error_deficit_pairs"].detach().cpu().numpy())
    if len(pairs) > max_points:
        pairs = pairs[np.random.choice(len(pairs), max_points, replace=False)]
    ax.scatter(pairs[:, 0],pairs[:, 1],s=1,alpha=0.25,rasterized=True,)
    ax.set_yscale("log")
    ax.set_xlabel("Velocity deficit")
    ax.set_ylabel("Prediction error")
    ax.set_title("Prediction Error vs. Velocity Deficit")
    ax.grid(alpha=0.3)
    
    ax = axes[2,1]
    mag = residual.detach().abs().cpu().numpy().flatten()
    mag = mag[mag > 0]
    if mag.size:
        mag.sort()
        ax.step(mag, np.arange(mag.size, 0, -1), where="post", linewidth=1)
        for percentile in (50, 90, 99):
            value = np.percentile(mag, percentile)
            ax.axvline(value, linestyle="--", linewidth=0.8, label=f"P{percentile}: {value:.2g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|PDE residual|")
    ax.set_ylabel("number of residuals at or above magnitude")
    ax.set_title("PDE residual exceedance")
    ax.grid(which="both", alpha=0.2)
    if mag.size: ax.legend()
    
    ax = axes[2,2]
    save_parity_plots(latest_fields, "u", ax)
    
    ax = axes[3,0]
    save_parity_plots(latest_fields, "v", ax)
    
    ax = axes[3,1]
    save_parity_plots(latest_fields, "w", ax)
    
    ax = axes[3,2]
    save_parity_plots(latest_fields, "p", ax)
    
    ax = axes[4,0]
    cosine = data[pde, LogCol.GRAD_COSINE]
    ax.plot(epochs[pde], cosine, label=f"EMA cosine ({cosine[-1]:.3g})" if cosine.size else "EMA cosine")
    ax.axhline(0.1, color="green", linestyle="--", label="increase threshold")
    ax.axhline(-0.05, color="red", linestyle="--", label="decrease threshold")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Data/PDE gradient alignment"); ax.set_xlabel("epoch"); ax.set_ylim(-1.05, 1.05); ax.legend()

    ax = axes[4,1]
    data_norm, pde_norm = data[pde, LogCol.DATA_GRAD_NORM], data[pde, LogCol.PDE_GRAD_NORM]
    weighted_norm = pde_norm*data[pde, LogCol.LAMBDA_PDE]
    ax.plot(epochs[pde], data_norm, label="data gradient")
    ax.plot(epochs[pde], pde_norm, label="raw PDE gradient")
    ax.plot(epochs[pde], weighted_norm, label="weighted PDE gradient")
    ax.set_yscale("symlog", linthresh=1e-12); ax.set_title("Objective gradient norms"); ax.set_xlabel("epoch"); ax.legend()

    ax = axes[4,2]
    pde_epochs = epochs[pde]
    progress = np.clip((pde_epochs-params.PRETRAIN_EPOCHS)/params.PDE_RAMP_EPOCHS, 0, 1)
    ratio_cap = params.PDE_MAX_GRAD_RATIO*0.5*(1-np.cos(np.pi*progress))
    actual_ratio = data[pde, LogCol.LAMBDA_PDE]*pde_norm/np.maximum(data_norm, 1e-12)
    ax.plot(pde_epochs, ratio_cap, label="scheduled maximum")
    ax.plot(pde_epochs, data[pde, LogCol.TARGET_GRAD_RATIO], label="target ratio")
    ratio_label = f"actual weighted ratio ({actual_ratio[-1]:.3g})" if actual_ratio.size else "actual weighted ratio"
    ax.plot(pde_epochs, actual_ratio, label=ratio_label)
    ratio_limit = float(ratio_cap[-1]) if ratio_cap.size else params.PDE_MAX_GRAD_RATIO
    ax.set_title("PDE gradient influence"); ax.set_xlabel("epoch"); ax.set_ylim(0, max(ratio_limit, 1e-6)); ax.legend()

    plot_sampling_distribution(sampling_axes, diagnoser.data)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def plot_sampling_distribution(axes, dataset):
    if dataset is None:
        for ax in axes:
            ax.text(.5, .5, "Sampling distribution unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
        return

    def numpy(tensor):
        return tensor.detach().cpu().numpy()

    reference = getattr(dataset, "_sampling_plot_reference", None)
    if reference is None:
        targets = numpy(dataset._all_targets)
        eligible = ~numpy(dataset._val_mask)
        crossflow = numpy(dataset._crossflow_sampling_scores)
        probabilities = numpy(dataset._sampling_probabilities)[eligible]
        pressure_low = targets[eligible, 3].min()
        pressure_span = targets[eligible, 3].max() - pressure_low
        pressure_span = pressure_span if pressure_span > 0 else 1.
        columns = [np.clip(1 - targets[:, 0], 0, 1), crossflow[:, 0], crossflow[:, 1],
                   (targets[:, 3] - pressure_low) / pressure_span]
        edges = [np.linspace(0, 1, 41), np.linspace(-1, 1, 41),
                 np.linspace(-1, 1, 41), np.linspace(0, 1, 41)]
        reference = {"edges": edges, "pressure_low": pressure_low, "pressure_span": pressure_span,
                     "population": [], "proposal": []}
        for values, bins in zip(columns, edges):
            population, _ = np.histogram(values[eligible], bins=bins)
            proposal, _ = np.histogram(values[eligible], bins=bins, weights=probabilities)
            reference["population"].append(100 * population / population.sum())
            reference["proposal"].append(100 * proposal / proposal.sum())
        dataset._sampling_plot_reference = reference

    targets = numpy(dataset.train_targets)
    crossflow = numpy(dataset.train_crossflow_scores)
    columns = [np.clip(1 - targets[:, 0], 0, 1), crossflow[:, 0], crossflow[:, 1],
               (targets[:, 3] - reference["pressure_low"]) / reference["pressure_span"]]
    names = ["u deficit", "v signed log", "w signed log", "pressure"]
    labels = ["1 - u/Uref", "Signed-log sampling score", "Signed-log sampling score",
              "p (scaled eligible range)"]
    for i, (ax, values, bins, name, label) in enumerate(zip(axes, columns, reference["edges"], names, labels)):
        selected, _ = np.histogram(values, bins=bins)
        selected = 100 * selected / max(selected.sum(), 1)
        ax.stairs(reference["population"][i], bins, color="0.6", linewidth=1,
                  label="Eligible CFD")
        ax.stairs(reference["proposal"][i], bins, color="tab:green", linestyle="--",
                  linewidth=1.2, label="Mixed draw weights")
        ax.stairs(selected, bins, color="tab:blue", linewidth=1.5, label="Selected training")
        ax.set_title(f"Sampling: {name}\n{len(targets):,} training points")
        ax.set_xlabel(label)
        ax.set_ylabel("Percent per display bin")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_yscale("linear")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        if i == 0:
            ax.legend(fontsize=8)
    
def save_parity_plots(fields, name, ax):
    pred, true = fields[name]

    pred_np = pred.detach().cpu().numpy().ravel()
    true_np = true.detach().cpu().numpy().ravel()

    ss_res = np.sum((true_np - pred_np) ** 2)
    ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)

    if ss_tot > 1e-12:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    max_points = 5000

    if len(true_np) > max_points:
        idx = np.random.choice(
            len(true_np),max_points,replace=False)
        true_plot = true_np[idx]
        pred_plot = pred_np[idx]
    else:
        true_plot = true_np
        pred_plot = pred_np

    ax.scatter(true_plot,pred_plot,s=2,alpha=0.4,rasterized=True,label=f"$R^2$ = {r2:.4f}")

    low = min(true_np.min(),pred_np.min())

    high = max(true_np.max(),pred_np.max())

    ax.plot([low, high],[low, high],"r--",linewidth=1.0)

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    ax.set_xlabel("CFD truth")
    ax.set_ylabel("PINN prediction")
    ax.set_title(f"{name} parity")
    ax.grid(alpha=0.25)
    ax.legend()
