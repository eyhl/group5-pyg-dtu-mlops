import matplotlib.pyplot as plt


def visualise_drift(results, kernels, ood):
    cells = [[results[kernel_name][i] for kernel_name in kernels] for i in range(2)]
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 3, 4)
    ax3 = plt.subplot(2, 3, 5)
    ax4 = plt.subplot(2, 3, 6)
    axes = [ax1, ax2, ax3, ax4]
    _ = axes[0].table(cellText=cells,
                      rowLabels=["Score", "p-value"],
                      colLabels=kernels,
                      loc="center",
                      )
    i = 1
    for key, value in results.items():
        base_embedded = value[2]
        features_embedded = value[3]
        axes[i].scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
        axes[i].scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
        axes[i].title.set_text(f'{key}: score {value[0]:.2f} p-value {value[1]:.2f}')
        i += 1
    axes[0].axis("off")
    if ood:
        plt.savefig("./reports/figures/drift_results_ood.png")
    else:
        plt.savefig("./reports/figures/drift_results.png")
    plt.show()
