import os
import copy
from alipy import ToolBox, query_strategy
from alipy.utils.misc import nlargestarg, nsmallestarg
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import sklearn
import TMR_s as stf
from utils import train_plain_target_model, get_proba_pred, calc_linear_entropy
import scipy.io as scio
from sklearn.decomposition import PCA
from alipy.experiment.state_io import StateIO
from alipy.query_strategy.query_labels import QueryInstanceQUIRE, QueryInstanceUncertainty, QueryInstanceCoresetGreedy, QueryInstanceRandom

paths_OC = [
    "amazon.mat",
    "caltech10.mat",
    "dslr.mat",
    "webcam.mat",
]
paths_PIE = [
    "PIE05.mat",
    "PIE07.mat",
    "PIE09.mat",
    "PIE27.mat"
]


def getXY(path):
    mm = scio.loadmat(path)
    if 'PIE' in path:
        X = mm['fea']
        Y = mm['gnd'].reshape((1, -1))[0]
    else:
        X = mm['fts']
        Y = mm['labels'].reshape((1, -1))[0]
    if Y.shape == 2:
        Y = np.array([i.tolist().index(max(i)) + 1 for i in Y])
    return X, Y


def select_MRLL(WT_unlab_pred, WTp_unlab_pred, beta=1, batch_size=1):
    norm_values = np.linalg.norm(WT_unlab_pred - WTp_unlab_pred, axis=1)
    scores = norm_values + np.asarray([beta * calc_linear_entropy(item_pred) for item_pred in WT_unlab_pred])
    return nlargestarg(scores, batch_size)


def main_loop(dataset, target_domain, fold, beta=1, ini_lab_num=0.1, strategy_name=''):
    if ini_lab_num != 0 and ini_lab_num < 1:
        ini_lab_num = ini_lab_num
    else:
        ini_lab_num = 10 if dataset == 'OfficeCaltech' else 100
    saver_fname = f"./results/{dataset}_{target_domain}/{target_domain}_fold{fold}_beta{beta}_ini{ini_lab_num}_qs{strategy_name}.save"
    budget = 150 if dataset == 'OfficeCaltech' else 250
    paths = paths_OC if dataset == 'OfficeCaltech' else paths_PIE
    dimension_reduction = 100 if dataset == 'OfficeCaltech' else 200
    k_dict = 20 if dataset == 'OfficeCaltech' else 60
    # lambda_value = 1 if dataset == 'OfficeCaltech' else 100
    # mu_value = 10 if dataset == 'OfficeCaltech' else 0.1
    lambda_value = 100 if dataset == 'OfficeCaltech' else 100
    mu_value = 0.1 if dataset == 'OfficeCaltech' else 0.1

    src_X = []
    src_Y = []
    src_models = []
    # train source models
    for source_domain in paths:
        if source_domain != target_domain:
            X, Y = getXY(f"{dataset}/{source_domain}")
            pca = PCA(n_components=dimension_reduction)
            X = pca.fit_transform(X=X, y=Y)
            src_X.append(X)
            src_Y.append(Y)
            Ws = train_plain_target_model(data=X, labelo=Y, lamda=1)
            # # test src models (test ok)
            pred_results = stf.predict_w(WT=Ws, data=X)
            print(balanced_accuracy_score(Y, pred_results))
            src_models.append(copy.deepcopy(Ws))

    WS = np.hstack((src_models[0], src_models[1], src_models[2]))

    # train init target model
    X, Y = getXY(f"{dataset}/{target_domain}")
    pca = PCA(n_components=dimension_reduction)
    X = pca.fit_transform(X=X, y=Y)

    os.makedirs(f'./results/{dataset}_{target_domain}/', exist_ok=True)
    alibox = ToolBox(X=X, y=Y, query_type='AllLabels', saving_path=f'./results/{dataset}_{target_domain}/')

    # Split data
    if ini_lab_num > 1:
        alibox.split_AL(test_ratio=0.3, initial_label_rate=ini_lab_num / (X.shape[0] * 0.7), split_count=10)
    else:
        alibox.split_AL(test_ratio=0.3, initial_label_rate=ini_lab_num, split_count=10)

    # The cost budget is 50 times querying
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', budget)

    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(fold)

    # get query strategy object. If None, use the proposed method
    if len(strategy_name) > 0:
        query_strategy = alibox.get_query_strategy(strategy_name, X=X, y=Y, train_idx=train_idx)
    else:
        query_strategy = None


    # Get intermediate results saver for one fold experiment
    if os.path.exists(saver_fname):
        saver = StateIO.load(saver_fname)
        train_idx, test_idx, label_ind, unlab_ind = saver.get_workspace()
        stopping_criterion.update_information(saver)
    else:
        saver = alibox.get_stateio(fold, saving_path=saver_fname)

        print("initial balanced_accuracy_score:")
        # Set initial performance point
        D, VT, VS, WT = stf.fit(data=np.mat(X[label_ind, :]), label=Y[label_ind], ws=WS,
                                LAMBDA=lambda_value, MU=mu_value, k=k_dict)
        score = stf.score(data=X[test_idx, :], label=Y[test_idx], WT=WT)
        print(f"transfer lambda={lambda_value}", score)

        saver.set_initial_point(score)

    # If the stopping criterion is simple, such as query 50 times. Use `for i in range(50):` is ok.
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        WT_plain = train_plain_target_model(data=np.mat(X[label_ind, :]), labelo=Y[label_ind])
        WT_pred = get_proba_pred(WT=WT, data=X[unlab_ind, :])
        WTp_pred = get_proba_pred(WT=WT_plain, data=X[unlab_ind, :])
        if query_strategy is None:
            select_ind = select_MRLL(WT_unlab_pred=WT_pred, WTp_unlab_pred=WTp_pred, beta=beta, batch_size=1)
            select_ind = [unlab_ind[idx] for idx in select_ind]
        else:
            if hasattr(query_strategy, 'select_by_prediction_mat'):
                select_ind = query_strategy.select_by_prediction_mat(unlabel_index=unlab_ind, predict=WT_pred, batch_size=1)
            else:
                select_ind = query_strategy.select(label_index=label_ind, unlabel_index=unlab_ind, model=None, batch_size=1)
            
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        D, VT, VS, WT = stf.fit(data=np.mat(X[label_ind, :]), label=Y[label_ind], ws=WS,
                                LAMBDA=lambda_value, MU=mu_value, k=k_dict)
        score = stf.score(data=X[test_idx, :], label=Y[test_idx], WT=WT)
        print(score)

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=score)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    return saver


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PIE', help="[PIE, OfficeCaltech]")
    parser.add_argument('--ini', type=float, default=0)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--AL', type=str, default='', help="[QueryInstanceQUIRE, QueryInstanceUncertainty, QueryInstanceCoresetGreedy, QueryInstanceRandom]")
    args = parser.parse_args()

    dataset = args.dataset
    target_domains = ["amazon.mat", "caltech10.mat"] if dataset == "OfficeCaltech" else paths_PIE

    for target_domain in target_domains:
        # for ini in [0.05, 0.1, 0.15, 0.2]:
        for fold in np.arange(10):
            print(target_domain, args.ini, args.beta, fold)
            main_loop(dataset, target_domain, int(fold), beta=args.beta, ini_lab_num=args.ini, strategy_name=args.AL)

# analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
# analyser.add_method(method_name='QBC', method_results=qbc_result)
# analyser.add_method(method_name='Unc', method_results=unc_result)
# analyser.add_method(method_name='EER', method_results=eer_result)
# analyser.add_method(method_name='Random', method_results=rnd_result)
# analyser.add_method(method_name='QUIRE', method_results=quire_result)
# analyser.add_method(method_name='Density', method_results=density_result)
# analyser.add_method(method_name='LAL', method_results=lal_result)
# if _I_have_installed_the_cvxpy:
#     analyser.add_method(method_name='BMDR', method_results=bmdr_result)
#     analyser.add_method(method_name='SPAL', method_results=spal_result)
# print(analyser)
# analyser.plot_learning_curves(title='Example of alipy', std_area=False)
