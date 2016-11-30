import numpy as np
from scipy.stats import pearsonr
import models, bookkeeping, utils
import matplotlib.pyplot as plt


# comparing GLM values feature importance values

def train_glm_xgb_models(X_train,y_train,X_test):
    glm_model = models.GLM_poisson(X_train, y_train, X_test,verbose=False,return_model=True)
    xgb_model = models.XGB_poisson(X_train, y_train, X_test,return_model=True)
    return glm_model,xgb_model


def get_weights(glm_model,xgb_model,plot=True,importance_type='weight'):

    n_neurons = glm_model.get_input_shape_at(0)
    glm_weights = glm_model.get_weights()[0]
    score_dict = xgb_model.get_score(importance_type=importance_type)

    cleaned_scores = np.zeros(n_neurons)
    for i in xrange(n_neurons):
        if 'f'+str(i) in score_dict.keys():
            cleaned_scores[i] = score_dict['f'+str(i)]

    if plot==True:
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(glm_weights, cleaned_scores, '.')
        ax1.set_title("Relationship between GLM weights and XGB feature importance scores for 1 neuron. r = " +str(pearsonr(cleaned_scores,np.squeeze(glm_weights))))
        ax1.set_xlabel("GLM weight")
        ax1.set_ylabel("XGB feature "+importance_type+" importance score")
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')

    return glm_weights, cleaned_scores