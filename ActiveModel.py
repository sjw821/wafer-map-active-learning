import numpy as np
import tensorflow as tf
import scipy as sc
from sklearn.utils import class_weight
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
import KerasDropoutPrediction2 as KDP
import pandas as pd
# from numpy import ma
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import torch

class Model(object):

    def __init__(self, save_path, data_directory):

        # a gpu growth
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        self.save_path= save_path
        self.data_directory = data_directory

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu',
                                         input_shape=(64, 64, 1)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        self.model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(9, activation='softmax'))

        adam = tf.keras.optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=adam)



    def print_structure(self):
        print(self.model.summary())

    def train_from_scratch(self, train_x, train_y, test_x, test_y, model_name, epoch= 100, phase=0):
        train_y = tf.keras.utils.to_categorical(train_y, 9)
        test_y = tf.keras.utils.to_categorical(test_y, 9)

        tb_hist = tf.keras.callbacks.TensorBoard(log_dir='{}graph/{}'.format(self.save_path, model_name), histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)

        print('start train for {}'.format(model_name))
        start_time = time.time()

        # TODO : weight re-initializing
        session = tf.keras.backend.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)

        self.model.fit(train_x,
                       train_y,
                       batch_size=128,
                       epochs=epoch,
                       validation_data=(test_x, test_y), #                            skip for time
                       callbacks=[tb_hist],
                       verbose=0
                       )

        end_time = time.time()
        minutes = (end_time - start_time) / 60

        if phase % 10 == 0:
            self.model.save('{}{}'.format(self.data_directory, model_name))
        print('save {} model : {} min'.format(model_name,minutes))

    def train_from_scratch2(self, train_x, train_y, val_x, val_y, model_name, epoch=150, phase=0): # with val
        train_y = tf.keras.utils.to_categorical(train_y, 9)
        val_y = tf.keras.utils.to_categorical(val_y, 9)

        tb_hist = tf.keras.callbacks.TensorBoard(log_dir='{}graph/{}'.format(self.save_path, model_name),
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
        mc = ModelCheckpoint('{}{}'.format(self.data_directory, model_name), monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        print('start train for {}'.format(model_name))
        start_time = time.time()

        # TODO : weight re-initializing
        session = tf.keras.backend.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)

        history=self.model.fit(train_x,
                       train_y,
                       batch_size=128,
                       epochs=epoch,
                       validation_data=(val_x, val_y),  # skip for time
                       callbacks=[tb_hist, es, mc],
                       verbose=0
                       )

        minepoch = np.argmin(history.history['val_loss'])
        print('used epoch : {}'.format(minepoch+1))

        end_time = time.time()
        minutes = (end_time - start_time) / 60

        self.model = tf.keras.models.load_model('{}{}'.format(self.data_directory, model_name))

        print('save {} model : {} min'.format(model_name, minutes))


    def load_model(self, modelfile):
        self.model = tf.keras.models.load_model(modelfile)



    def train_finetune6(self, train_x, train_y, train_x_add, train_y_add, test_x, test_y, model_name, phase=1,
                        start_epoch=100):  # with validation set !

        sample_weight = np.concatenate((np.ones(train_x.shape[0]), np.ones(train_x_add.shape[0])*10))

        train_x = np.concatenate((train_x, train_x_add))
        train_y = np.concatenate((train_y, train_y_add))

        train_y = tf.keras.utils.to_categorical(train_y, 9)
        test_y = tf.keras.utils.to_categorical(test_y, 9)

        # adam = tf.keras.optimizers.Adam(lr=0.0001)
        # self.model.compile(loss='categorical_crossentropy',
        #                    optimizer=adam)

        start_time = time.time()

        tb_hist = tf.keras.callbacks.TensorBoard(log_dir='{}graph/{}'.format(self.save_path, model_name),
                                                 histogram_freq=0,
                                                 write_graph=False,
                                                 write_images=False)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20) # patience setting
        mc = ModelCheckpoint('{}{}.h5'.format(self.save_path, model_name), monitor='val_loss', mode='min', verbose=0,
                             save_best_only=True)

        # initial_epoch = start_epoch + 10 * (phase - 1)
        # last_epochs = initial_epoch + 50

        print('start train for {}'.format(model_name))
        # start_time = time.time()

        history= self.model.fit(train_x,
                       train_y,
                       batch_size=128,
                       epochs = 30, # 50->30
                       # epochs=last_epochs,
                       sample_weight=sample_weight,
                       validation_data=(test_x, test_y),
                       callbacks=[tb_hist, es, mc],
                       verbose=0,
                       # initial_epoch=initial_epoch
                       )

        minepoch=np.argmin(history.history['val_loss'])
        print('used epoch : {}'.format(minepoch+1))

        self.model = tf.keras.models.load_model('{}{}.h5'.format(self.save_path, model_name))

        end_time = time.time()
        minutes = (end_time - start_time) / 60

        # if phase%10 ==0 :
        #     self.model.save('{}{}.h5'.format(self.save_path, model_name))

        print('phase{} finetune time : {} min'.format(phase, minutes))

        return minepoch+1

        

    def auc_one_a_rest(self, y_test, y_score, n_classes=9):
        #one versus rest
        y_test = tf.keras.utils.to_categorical(y_test, n_classes)
        fpr = dict()
        tpr = dict()
        auc_class = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            auc_class[i] = auc(fpr[i], tpr[i])
        auc_mean = np.array(list(auc_class.values())).mean()
        return auc_mean, auc_class


    def test(self, test_x, test_y):

        pred_y = self.model.predict(test_x, batch_size= 512)
        # pred_y_class = pred_y.argmax(axis=1)
        #
        # cfm = confusion_matrix(test_y, pred_y_class, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        # accuracy = (pred_y_class == test_y).sum() / test_y.shape[0] * 100
        # f1 = f1_score(test_y, pred_y_class, average='macro')
        # precision = precision_score(test_y, pred_y_class, average='macro')
        # recall = recall_score(test_y, pred_y_class, average='macro')
        auc_mean, auc_class = self.auc_one_a_rest(test_y, pred_y, 9)

        # return cfm, accuracy, f1, precision, recall, auc_mean, auc_class, pred_y
        return auc_mean, auc_class, pred_y



    def select_random(self, unlabeled_x, unlabeled_y, top_k=30):
        idx = np.random.permutation(unlabeled_y.shape[0])[:top_k]

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_entropy(self, unlabeled_x, unlabeled_y, top_k=30):

        pred_y = self.model.predict(unlabeled_x, batch_size= 512)
        pred_y_class = pred_y.argmax(axis=1)

        # uncertainty = np.apply_along_axis(sc.stats.entropy, 1, pred_y)
        uncertainty =  -np.sum(pred_y * my_log(pred_y), axis=1)
        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled

    def select_confidence(self, unlabeled_x, unlabeled_y, top_k=30):

        pred_y = self.model.predict(unlabeled_x, batch_size= 512)
        pred_y_class = pred_y.argmax(axis=1)

        uncertainty= 1- pred_y.max(axis=1)
        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_margin(self, unlabeled_x, unlabeled_y, top_k=30):

        pred_y = self.model.predict(unlabeled_x, batch_size= 512)
        pred_y_class = pred_y.argmax(axis=1)

        pred_y.sort(axis=1)
        margin = pred_y[:, -1]-pred_y[:, -2]
        uncertainty= -margin
        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_predictive_entropy(self, unlabeled_x, unlabeled_y, top_k=30, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size=512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        norm_pred_y = pred_y / pred_y.sum(axis=1, keepdims=True)
        uncertainty = -np.sum(norm_pred_y * my_log(norm_pred_y), axis=1)
        # uncertainty = np.apply_along_axis(sc.stats.entropy, 1, pred_y)

        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_variation_ratio(self, unlabeled_x, unlabeled_y, top_k=30, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        maxclass=result.argmax(axis=2)
        classes, frequency= sc.stats.mode(maxclass,axis=0)
        uncertainty=1- frequency.flatten() /float(result.shape[0])

        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled



    def select_meanSTD(self, unlabeled_x, unlabeled_y, top_k=30, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        std=result.std(axis=0)
        uncertainty = std.sum(axis=1)

        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_BALD(self, unlabeled_x, unlabeled_y, top_k=30, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        norm_pred_y = pred_y / pred_y.sum(axis=1, keepdims=True)
        entropy = -np.sum(norm_pred_y * my_log(norm_pred_y), axis=1)
        # entropy = np.apply_along_axis(sc.stats.entropy, 1, pred_y)

        entropy_iter = -np.sum(result * my_log(result), axis=2)
        # entropy_iter = np.apply_along_axis(sc.stats.entropy, 2, result)
        entropy_m = entropy_iter.mean(axis=0)
        uncertainty = entropy-entropy_m

        idx = uncertainty.argsort()[-top_k:][::-1]  # top entropy idx

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled

    def select_batchBALD(self, unlabeled_x, unlabeled_y, top_k=30, n_iter=50):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size=512)

        pool_num = unlabeled_y.shape[0]
        num_samples = 500
        num_samples_per_ws = num_samples / n_iter
        num_class = 9

        index_list = np.arange(0, pool_num)
        An = []

        # 첫번째 acqusition
        pred_y = result.mean(axis=0)
        norm_pred_y = pred_y / pred_y.sum(axis=1, keepdims=True)
        entropy = -np.sum(norm_pred_y * my_log(norm_pred_y), axis=1)
        # entropy= np.apply_along_axis(stats.entropy,1,pred_y)

        entropy_iter = -np.sum(result * my_log(result), axis=2)
        # entropy_iter = np.apply_along_axis(sc.stats.entropy, 2, result)
        m_entropy = entropy_iter.mean(axis=0)

        bald = entropy - m_entropy
        top_id_temp = bald.argmax()  # 순서상 id
        top_id = index_list[top_id_temp]  # 순서상id -> original id
        An.append(top_id)
        index_list = np.delete(index_list, top_id_temp)

        P_prev = result[:, top_id, :].T  # C^n, K  only run at first

        c_entropy = m_entropy.copy()
        for i in range(top_k-1):

            c_entropy = c_entropy + m_entropy[top_id]
            c_entropy_current = c_entropy[index_list]

            j_entropy = np.ones(pool_num, )

            if num_class ** (i + 1) <= num_samples:  # exact j entropy
                # print('with exact')
                for j in range(pool_num):
                    Pn = result[:, j, :].T  # C, K

                    # joint entropy 계산
                    temp_prob = (1 / n_iter) * (P_prev @ Pn.T)
                    j_entropy[j] = -np.sum(temp_prob * my_log(temp_prob))

                j_entropy_current = j_entropy[index_list]

                bald = j_entropy_current - c_entropy_current
                top_id_temp = bald.argmax()
                top_id = index_list[top_id_temp]
                An.append(top_id)
                index_list = np.delete(index_list, top_id_temp)

                # P recursive with top_id
                P_prev = P_prev.T[:, :, None]  # K M 1
                Pn = result[:, top_id, :].T  # C, K
                Pn = Pn.T[:, None, :]  # K 1 C
                P_prev = P_prev * Pn
                P_prev = P_prev.reshape((n_iter, -1, 1))
                P_prev = P_prev.squeeze(2).T

            else:  # approximate
                # print('with approx')
                probs_K_N_C = result[:, An, :]
                probs_N_K_C = np.swapaxes(probs_K_N_C, 0, 1)

                P_prev = sample_M_K( probs_N_K_C, ss = num_samples_per_ws )


                for j in range(pool_num):
                    Pn = result[:, j, :].T  # C, K

                    temp_prob = (1 / n_iter) * (P_prev @ Pn.T)
                    entropy_temp = temp_prob * my_log(temp_prob)

                    P_M_1 = P_prev.mean(axis=1, keepdims=True)
                    entropy_temp = entropy_temp / P_M_1
                    j_entropy[j] = - (1 / num_samples) * np.sum(entropy_temp)

                j_entropy_current = j_entropy[index_list]

                bald = j_entropy_current - c_entropy_current
                top_id_temp = bald.argmax()
                top_id = index_list[top_id_temp]
                An.append(top_id)
                index_list = np.delete(index_list, top_id_temp)

        to_train_x = unlabeled_x[An, :, :, :]
        to_train_y = unlabeled_y[An]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[An] = False

        return to_train_x, to_train_y, mask_unlabeled

#-----------------------------------------------------------------------------------------------------------------------

    def diverse_selector(self, uncertainty, pred_y_class, classnum=9):
        T = pd.DataFrame({'U': uncertainty, 'pred': pred_y_class, 'check': 0})

        for i in range(classnum):
            if sum(T['pred'] == i) != 0:
                classT = T[T['pred'] == i]
                classmaxidx = classT['U'].idxmax()
                T.loc[classmaxidx, 'check'] = 1

        remain = classnum - T['check'].sum()
        while remain > 0:
            notcheckT = T[T['check'] == 0]
            notcheckmaxidx = notcheckT['U'].idxmax()
            T.loc[notcheckmaxidx, 'check'] = 1
            remain = remain - 1

        idx = T[T['check'] == 1].index.values
        return idx



    def select_entropy_div(self, unlabeled_x, unlabeled_y):

        pred_y = self.model.predict(unlabeled_x, batch_size= 512)
        pred_y_class = pred_y.argmax(axis=1)

        uncertainty = -np.sum(pred_y * my_log(pred_y), axis=1)
        # uncertainty = np.apply_along_axis(sc.stats.entropy, 1, pred_y)
        idx= self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled

    def select_confidence_div(self, unlabeled_x, unlabeled_y):

        pred_y = self.model.predict(unlabeled_x, batch_size= 512)
        pred_y_class = pred_y.argmax(axis=1)

        uncertainty= 1- pred_y.max(axis=1)
        idx = self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_margin_div(self, unlabeled_x, unlabeled_y):

        pred_y = self.model.predict(unlabeled_x, batch_size= 512)
        pred_y_class = pred_y.argmax(axis=1)

        pred_y.sort(axis=1)
        margin = pred_y[:, -1]-pred_y[:, -2]
        uncertainty= -margin
        idx = self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_predictive_entropy_div(self, unlabeled_x, unlabeled_y, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        norm_pred_y = pred_y / pred_y.sum(axis=1, keepdims=True)
        uncertainty = -np.sum(norm_pred_y * my_log(norm_pred_y), axis=1)
        # uncertainty = np.apply_along_axis(sc.stats.entropy, 1, pred_y)


        idx = self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_variation_ratio_div(self, unlabeled_x, unlabeled_y, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        maxclass=result.argmax(axis=2)
        classes, frequency= sc.stats.mode(maxclass,axis=0)
        uncertainty=1-  frequency.flatten() / float(result.shape[0])

        idx = self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled



    def select_meanSTD_div(self, unlabeled_x, unlabeled_y, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        std=result.std(axis=0)
        uncertainty = std.sum(axis=1)

        idx = self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled


    def select_BALD_div(self, unlabeled_x, unlabeled_y, n_iter=50 ):

        kdp = KDP.DropoutPrediction(self.model)
        result = kdp.predict(unlabeled_x, n_iter, batch_size= 512)

        pred_y = result.mean(axis=0)
        pred_y_class = pred_y.argmax(axis=1)

        norm_pred_y = pred_y / pred_y.sum(axis=1, keepdims=True)
        entropy = -np.sum(norm_pred_y * my_log(norm_pred_y), axis=1)
        # entropy = np.apply_along_axis(sc.stats.entropy, 1, pred_y)

        entropy_iter = -np.sum(result * my_log(result), axis=2)
        # entropy_iter = np.apply_along_axis(sc.stats.entropy, 2, result)
        entropy_m = entropy_iter.mean(axis=0)
        uncertainty = entropy-entropy_m

        idx = self.diverse_selector(uncertainty, pred_y_class, classnum=9)

        to_train_x = unlabeled_x[idx, :, :, :]
        to_train_y = unlabeled_y[idx]

        mask_unlabeled = np.ones(unlabeled_x.shape[0], dtype=bool)
        mask_unlabeled[idx] = False

        return to_train_x, to_train_y, mask_unlabeled



    @staticmethod
    def clear_model():
        tf.keras.backend.clear_session()


def gather_expand(data, dim, index):
    #     if DEBUG_CHECKS:
    #         assert len(data.shape) == len(index.shape)
    #         assert all(dr == ir or 1 in (dr, ir) for dr, ir in zip(data.shape, index.shape))

    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)  # new shape으로 바꾸는데 복제하면서
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)

def sample_M_K( probs_N_K_C , ss=1000 ):

    probs_N_K_C = torch.from_numpy(probs_N_K_C)

    probs_N_K_C = probs_N_K_C.double()

    K = probs_N_K_C.shape[1]

    choices_N_K_S = batch_multi_choices(probs_N_K_C, ss).long()

    expanded_choices_N_K_K_S = choices_N_K_S[:, None, :, :]
    expanded_probs_N_K_K_C = probs_N_K_C[:, :, None, :]

    probs_N_K_K_S = gather_expand(expanded_probs_N_K_K_C, dim=-1, index=expanded_choices_N_K_K_S)
    # exp sum log seems necessary to avoid 0s?
    probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
    samples_K_M = probs_K_K_S.reshape((K, -1))

    samples_M_K = samples_K_M.t()
    samples_M_K = samples_M_K.numpy()
    return samples_M_K

def batch_multi_choices(probs_b_C, M):
    """
    probs_b_C: Ni... x C
    Returns:
        choices: Ni... x M
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))

    # samples: Ni... x draw_per_xx
    choices = torch.multinomial(probs_B_C, num_samples=int(M), replacement=True)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [int(M)])
    return choices_b_M

def my_log(x):
    a=np.log(x)
    a[np.isinf(a)]=0
    return a