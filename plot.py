import alipy
import os
from alipy.experiment.state_io import StateIO
import numpy as np


methods_lstyle = ['-', ':', '-.',
                  '--', '--',
                  '--', '--', '--', '--', '-']
methods_color = ['#F71E35',
                 '#274c5e', '#0080ff',
                 '#bf209f', '#79bd9a', '#4013af', 'gray', 'black']
methods_marker = ["D", "d", "d",
                  "^", "^",
                  "o", "o", "^", "^", "o"]


def plot_beta(dataset, target_domain, title):
    analyser = alipy.experiment.ExperimentAnalyser(x_axis='num_of_queries')

    for beta in [0.1, 1, 10, 100]:
        saver_arr = []
        for fold in np.arange(10):
            try:
                saver = StateIO.load(f"./beta_results/{dataset}_{target_domain}/{target_domain}_fold{fold}_beta{beta}_ini{0}.save")
            except FileNotFoundError:
                saver = StateIO.load(f"./beta_results/{dataset}_{target_domain}/{target_domain}_fold{fold}_beta{beta}.save")
            saver_arr.append(saver)
        analyser.add_method(method_name=f'beta_{beta}', method_results=saver_arr)

    print(analyser)
    plt = analyser.plot_learning_curves(title='', std_area=False, show=False, saving_path=None, plot_interval=5)
    ax = plt.gca()
    for id, line in enumerate(ax.lines):
        plt.setp(line, linestyle=methods_lstyle[id], color=methods_color[id], marker=methods_marker[id])
    plt.legend(fancybox=True, framealpha=0.5)
    plt.savefig(os.path.join(f'./{title}.png'))
    # plt.show()
    plt.close()
    print()



if __name__ == "__main__":
    from active_learning import paths_PIE, paths_OC

    dataset = "PIE"
    for target_domain in paths_PIE:
        plot_beta(dataset, target_domain, target_domain)

    dataset = "OfficeCaltech"
    for target_domain in [
        "amazon.mat",
        "caltech10.mat"
    ]:
        plot_beta(dataset, target_domain, target_domain)
