# import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import scipy as sc

class DropoutPrediction:

    # dropout mask fixed....maybe fix random seed for dropout ?
    def __init__(self, model):
        self.f = tf.keras.backend.function(
                [model.layers[0].input,
                 tf.keras.backend.learning_phase()],
                [model.layers[-1].output])

    def predict(self, x, n_iter=50, batch_size=128):

        # def predictive_entropy(a):
        #     return - np.sum(np.log(a) * a)

        result= []

        print('predict with dropout....')
        quitient, remainder = x.shape[0] // batch_size, x.shape[0] % batch_size
        for i in range(n_iter):
            temp = []
            for j in range(quitient):
                start_idx= j*batch_size
                batch_x= x[start_idx:start_idx+batch_size]
                temp.append(self.f([batch_x,1]))
            temp_result=np.array(temp).reshape(batch_size*quitient,9)

            if remainder == 0:
                epoch_result=temp_result
            else:
                last_batch_x= x[quitient*batch_size:] # 딱 떨어질때 empty 라서 inference할때 error
                epoch_result=np.concatenate((temp_result, self.f([last_batch_x,1])[0]), axis=0)

            result.append(epoch_result)

            # if i%10 == 0 :
            #     print(str(i) + 'iter predict complete')

        result = np.array(result).reshape(n_iter, len(x), 9 )
        # pred_y = result.mean(axis=0)
        # #entropy cal
        # entropy = np.apply_along_axis(sc.stats.entropy, 1, pred_y)
        # print('predict with dropout end')
        #
        # return pred_y, entropy

        return result