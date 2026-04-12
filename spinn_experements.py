import pandas as pd
import matplotlib.pyplot as plt
from IPython import display


def disply_sweep_results(results, param_name):
    display_results = []
    for ri in results:
        if param_name in ri['config'].get('model_args', {}):
            param_val = ri['config']['model_args'][param_name]
        else:
            param_val = ri['config'][param_name]
        display_results.append({param_name: param_val,
                                "MSE": round(ri['mse'], 6),
                                "L2": round(ri['l2_norm'], 6)})

    display.display(pd.DataFrame(display_results))


def draw_sweep_results(results, param_name, log_iter=100):
    fig, axes = plt.subplots(2, 2, figsize=(22, 22))

    for ri in results:
        if param_name in ri['config'].get('model_args', {}):
            param_val = ri['config']['model_args'][param_name]
        else:
            param_val = ri['config'][param_name]

        axes[0, 0].semilogy(ri["history"]["loss"], label=f"{param_name}={param_val}")
        axes[0, 1].semilogy(ri["history"]["residual"], label=f"{param_name}={param_val}")
        axes[1, 0].semilogy(ri["history"]["boundary"], label=f"{param_name}={param_val}")

        iters = range(0, len(ri["history"]["error"]) * log_iter,
                      log_iter)
        axes[1, 1].semilogy(list(iters), ri["history"]["error"], label=f"{param_name}={param_val}")

    axes[0, 0].set_title('loss')
    axes[0, 1].set_title('pde')
    axes[1, 0].set_title('bc')
    axes[1, 1].set_title('l2')

    for ax in axes:
        for axx in ax:
            axx.legend()
