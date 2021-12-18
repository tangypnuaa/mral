import os
from alipy.experiment.state_io import StateIO
from alipy.experiment.experiment_analyser import StateIOContainer
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
import re
import copy

methods_res = ['our', 'our_incons', 'our_trans', 'least_conf',
               'LS', 'ADDA', 'img_trans', 'uncertain_map', 'margin_avg', 'click_spvs' ,'random']
methods_label = ['QBox', 'QBox-Incons.', 'QBox-Trans.',
                 'Least Conf.', 'Local. Stability',
                 'AADA', 'Image Transf.', 'Uncertain Map', 'Margin Average', 'Center Click', 'Random']


def get_one_dataset_AP(dataset, target_domain, beta=1, fold=10, length=150, smooth=False):
    saver_arr = []
    for fo in range(fold):
        try:
            saver = StateIO.load(f"./beta_results/{dataset}_{target_domain}/{target_domain}_fold{fo}_beta{beta}_ini{0}.save")
        except FileNotFoundError:
            saver = StateIO.load(f"./beta_results/{dataset}_{target_domain}/{target_domain}_fold{fo}_beta{beta}.save")
        saver.recover_workspace(length)
        saver_arr.append(saver)

    iocont = StateIOContainer(f"{dataset}_{target_domain}", saver_arr)
    perf_mat = iocont.extract_matrix()
    return perf_mat


def print_tex(dataset, data_mat):
    global methods_label, methods_res
    class_fn = f"{dataset}{'_tgt' if dataset != 'voc' else ''}.names"
    with open('/home/tangyingpeng/yolov3/data/'+class_fn, 'r') as f:
        class_arr = f.read().splitlines()

    assert data_mat.shape[0] == len(class_arr)
    assert data_mat.shape[1] == len(methods_res)
    bold_idx = np.argmax(data_mat, axis=1)

    # latex codes
    print("""\\begin{tabular}{c|ccccccccccc}
    \\toprule
    \hline
    & [1]& [2]& [3]& [4]& [5]& [6]& [7]& [8]& [9]& [10]& [11] \\\\
    \hline""")

    for id, val in enumerate(bold_idx):
        print(f"\t{class_arr[id]} ", end='')
        for col in range(data_mat.shape[1]):
            if col != val:
                print(f"& {data_mat[id][col]*100:.1f} ", end='')
            else:
                print('& \\textbf{'+ f"{data_mat[id][col]*100:.1f}"+'} ', end='')
        print('\\\\')

          
    print("""\t\hline
    \\bottomrule
\end{tabular}
    """
    )


if __name__ == "__main__":
    from active_learning import paths_PIE, paths_OC

# for ini in [0.05, 0.1, 0.15, 0.2]:
    dataset = "PIE"
    for target_domain in paths_PIE:
        perf_mat = get_one_dataset_AP(dataset, target_domain)
        raw_arr = np.mean(perf_mat, axis=1)

    dataset = "OfficeCaltech"
    for target_domain in [
        "amazon.mat",
        "caltech10.mat"
    ]:
        perf_mat = get_one_dataset_AP(dataset, target_domain)
        raw_arr = np.mean(perf_mat, axis=1)
    