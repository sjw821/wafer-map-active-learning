import pickle
import os
import numpy as np
# from read_data import read_resize, random_split_in_trainingset, make_balanced_initial_trainingset
import ActiveModel
import time

data_directory='/home/woong/WFmap/'

with open('{}allrandom_data_1_with_val.pickle'.format(data_directory), 'rb') as f:
    train_x, train_y, val_x, val_y, test_x, test_y, unlabeled_x, unlabeled_y= pickle.load(f)
    print('read data')

print('train shape {}, val shape {}, unlabeled shape {}, test shape {}'.format(train_x.shape, val_x.shape, unlabeled_x.shape,
                                                                                  test_x.shape))


save_path='/home/woong/WFmap/finetune6_try1_with_val_with_rambda/'

# initial model training
start_model='start_model_allrandom_1_with_val.h5'
# initial_model= ActiveModel.Model(save_path= save_path, data_directory=data_directory)
# initial_model.train_from_scratch2(train_x,train_y,val_x,val_y, start_model, epoch=150)

def run_simulation(active_method_name):
    auc_mean = {}
    auc_class = {}
    to_train_dict = {}
    pred_probs = {}
    total_epoch = {}

    active_model = ActiveModel.Model(save_path=save_path,  data_directory=data_directory)
    # variation_ratio_model.train_from_scratch(train_x,train_y, test_x, test_y, 'start_model', epoch=70)
    active_model.load_model('{}{}'.format(data_directory, start_model))
    auc_mean[0], auc_class[0], pred_prob = active_model.test(test_x, test_y)
    print('----------------------------phase 0 end, AUC: {}'.format(auc_mean[0]))

    unlabeled_x_c = np.copy(unlabeled_x)
    unlabeled_y_c = np.copy(unlabeled_y)
    train_x_c = np.copy(train_x)
    train_y_c = np.copy(train_y)

    for i in range(1, 101):
        start_time = time.time()

        if active_method_name =='variation_ratio':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_variation_ratio(unlabeled_x_c, unlabeled_y_c, n_iter=30, top_k=9)
        elif active_method_name =='predictive_entropy':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_predictive_entropy(unlabeled_x_c, unlabeled_y_c, n_iter=30, top_k=9)
        elif active_method_name =='mean_STD':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_meanSTD(unlabeled_x_c, unlabeled_y_c, n_iter=30, top_k=9)
        elif active_method_name == 'BALD':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_BALD(unlabeled_x_c, unlabeled_y_c, n_iter=30, top_k=9)
        elif active_method_name == 'confidence':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_confidence(unlabeled_x_c, unlabeled_y_c, top_k=9)
        elif active_method_name == 'margin':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_margin(unlabeled_x_c, unlabeled_y_c, top_k=9)
        elif active_method_name == 'entropy':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_entropy(unlabeled_x_c, unlabeled_y_c, top_k=9)
        elif active_method_name == 'variation_ratio_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_variation_ratio_div(unlabeled_x_c, unlabeled_y_c, n_iter=30)
        elif active_method_name == 'predictive_entropy_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_predictive_entropy_div(unlabeled_x_c, unlabeled_y_c, n_iter=30)
        elif active_method_name == 'mean_STD_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_meanSTD_div(unlabeled_x_c, unlabeled_y_c, n_iter=30)
        elif active_method_name == 'BALD_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_BALD_div(unlabeled_x_c, unlabeled_y_c, n_iter=30)
        elif active_method_name == 'confidence_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_confidence_div(unlabeled_x_c, unlabeled_y_c)
        elif active_method_name == 'margin_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_margin_div(unlabeled_x_c, unlabeled_y_c)
        elif active_method_name == 'entropy_div':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_entropy_div(unlabeled_x_c, unlabeled_y_c)
        elif active_method_name == 'batch_BALD':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_batchBALD(unlabeled_x_c, unlabeled_y_c, top_k=9, n_iter=30)
        elif active_method_name == 'random':
            to_train_x, to_train_y, mask_unlabeled = active_model.select_random(unlabeled_x_c, unlabeled_y_c, top_k=9)

        print(np.unique(to_train_y, return_counts=True))
        # data update
        unlabeled_x_c = unlabeled_x_c[mask_unlabeled]
        unlabeled_y_c = unlabeled_y_c[mask_unlabeled]

        print('finetune data : train {}, add data {}, unlabeled shape {}'.format(train_x_c.shape, to_train_x.shape,
                                                                                 unlabeled_x_c.shape))
        total_epoch[i] = active_model.train_finetune6(train_x_c, train_y_c, to_train_x, to_train_y, val_x,
                                                               val_y,
                                                               '{}_finetune_phase{}'.format(active_method_name,i),
                                                               phase=i, start_epoch=100)
        train_x_c = np.concatenate((train_x_c, to_train_x))
        train_y_c = np.concatenate((train_y_c, to_train_y))

        auc_mean[i], auc_class[i], pred_prob = active_model.test(test_x, test_y)
        if (i % 10) == 0:
            pred_probs[i] = pred_prob

        to_train_dict['phase{}'.format(i)] = [to_train_x, to_train_y] # to_train_x 지우기?
        end_time = time.time()
        minutes = (end_time - start_time) / 60
        print('----------------------------phase {} end, AUC: {}, time: {}min'.format(i, auc_mean[i], minutes))

    with open(save_path + 'measures_{}.pickle'.format(active_method_name), 'wb') as f:
        pickle.dump((auc_mean, auc_class, to_train_dict, pred_probs, total_epoch), f)

    ActiveModel.Model.clear_model()


active_model_name_list=[
                # 'variation_ratio',
                # 'predictive_entropy',
                # 'mean_STD',
                # 'BALD',
                # 'confidence',
                # 'margin',
                # 'entropy',
                # 'variation_ratio_div',
                # 'predictive_entropy_div',
                # 'mean_STD_div',
                # 'BALD_div',
                # 'confidence_div',
                # 'margin_div',
                # 'entropy_div',
                # 'batch_BALD',
                'random'
                ]

for active_model_name in active_model_name_list:
    start_time=time.time()
    run_simulation(active_model_name)
    end_time=time.time()
    minutes = (end_time - start_time) / 60
    print('-----------------------------------------------------------------------------total elapsed time for {} : {}min'.format(active_model_name, minutes))

