# %%
'''
Offline classification. Tested classifiers so far: k-fold LDA

Author:
    Ayca Kepce
    Nathan van Beelen

'''

# !%matplotlib qt

import numpy as np

import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import process_features, epoching, k_fold_LDA

# %%
# Alternatively load the concatenated data
file = r'..\data\sub_epochs\sub_epochs_trial.pkl'
with open(file, 'rb') as handle:
    sub_epochs = pickle.load(handle)

# %%

# Now we extract the data to a Numpy array of the format: (n_epochs, n_channels, n_times)
dataset_epochs, dataset_ndarray, eeg_labels = epoching(sub_epochs)

# %%
###################################
# Test the performance using K-Fold
###################################

# Running the k-fold LDA on the data
k_ldas, scores, selected_feature_idx = k_fold_LDA(dataset_epochs, dataset_ndarray, eeg_labels, n_features_to_select=20)

# Selecting the best LDA features given the scores of every LDA
best_lda_features = k_ldas[np.argmax(scores)][1]

# %%

#######################
# Train on all the data
#######################
lda = LinearDiscriminantAnalysis()

X, csp = process_features(dataset_ndarray, dataset_epochs, eeg_labels)
fitted_lda = lda.fit(X[:,best_lda_features], eeg_labels)

# Pickle the LDA classifier for use in the online classification.
#with open(r'C:\Users\AYCA\PycharmProjects\biosignal_stars\models\fitted_lda.pkl', 'wb') as file:
#    pickle.dump(fitted_lda, file)

# Pickle the csp for use in the online classification.
#with open(r'C:\Users\AYCA\PycharmProjects\biosignal_stars\models\csp.pkl', 'wb') as file:
#    pickle.dump(csp, file)
# %%

