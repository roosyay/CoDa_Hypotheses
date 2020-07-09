import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import ampligraph
import itertools

triples = np.load('triples.npy')
np.load('seen_triples.npy')
np.load('unseen_triples.npy')
np.load('unseen_triples_hasEffectOnly.npy')


#  Train, test, valid split 
from ampligraph.evaluation import train_test_split_no_unseen 

X = triples
X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2000) # DisGeNET: 8% = 2794
X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2000)

print('Train set size: ', X_train.shape)
print('Test set size: ', X_test.shape)
print('Valid set size: ', X_valid.shape)


#  Define ComplEx model 
from ampligraph.latent_features import ComplEx

model = ComplEx(batches_count=100, 
                seed=555, 
                epochs=100, 
                k=200, 
                eta=15,
                loss='multiclass_nll', 
                embedding_model_params = {'negative_corruption_entities': 'all'},
                regularizer='LP', 
                regularizer_params={'p':1, 'lambda':1e-5}, 
                initializer= 'xavier', 
                initializer_params= {'uniform': False},
                optimizer= 'adam',
                optimizer_params = {'lr': 0.0005}, 
                verbose=True)

positives_filter = X

#  Fit model on the training data 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

model.fit(X_train, early_stopping = False)

from ampligraph.latent_features import save_model, restore_model
save_model(model, './best_model.pkl')
model = restore_model('./best_model.pkl')


#  Sanity check 
if model.is_fitted:
    print('The model is fit!')
else:
    print('The model is not fit! Did you skip a step?')


#  Evaluate performance on the test set 
from ampligraph.evaluation import evaluate_performance
ranks = evaluate_performance(X_test, 
                             model=model, 
                             filter_triples=positives_filter,   # Corruption strategy filter defined above 
                             use_default_protocol=True, # corrupt subj and obj separately while evaluating
                             verbose=True)


#  See evaluation scores
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score

mr = mr_score(ranks)
print("MR: %.2f" % (mr))
mrr = mrr_score(ranks)
print("MRR: %.2f" % (mrr))

hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))