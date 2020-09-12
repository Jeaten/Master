import gzip
import cPickle
import sys
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from multi_layer_km import test_SdC
from cluster_acc import acc
from Function import DataModule
from Function import Process
result = []
path ='./File/'
trials = 1
filename = 'train.pkl.gz'
dataset = path + filename
data = ['fin_idx.npy', 'cluster_data.npy', 'fin_centers.npy']
for K in range( 13, 14 ):
    config = {'Init': '',
              'lbd': 1,  # reconstruction
              'beta': 0,
              'output_dir': 'MNIST_results',
              'save_file': 'mnist_pre.pkl.gz',
              'pretraining_epochs': 25,
              'pretrain_lr_base': 0.0001,
              'mu': 0.9,
              'finetune_lr': 0.0001,
              'training_epochs': 5,
              'dataset': dataset,
              'batch_size': 96,
              'nClass': K,
              'hidden_dim': [500, 500, 2000, 10],
              'diminishing': False}
    results = []
    for i in range( trials ):
        res_metrics = test_SdC( **config )
        results.append( res_metrics )
    results_SAEKM = np.zeros( (trials, 3) )
    results_DCN = np.zeros( (trials, 3) )
    N = config['training_epochs'] / 5
    ##training_epochs is 250,so N is 50
    for i in range( trials ):
        results_SAEKM[i] = results[i][0]
        results_DCN[i] = results[i][N]
    SAEKM_mean = np.mean( results_SAEKM, axis=0 )
    SAEKM_std = np.std( results_SAEKM, axis=0 )
    DCN_mean = np.mean( results_DCN, axis=0 )
    DCN_std = np.std( results_DCN, axis=0 )
    results_all = np.concatenate( (DCN_mean, DCN_std, SAEKM_mean, SAEKM_std),axis=0 )
    print( results_all )
    config = {'Init': 'mnist_pre.pkl.gz',
              'lbd': 1,  # reconstruction
              'beta': 1,
              'output_dir': 'MNIST_results',
              'save_file': 'mnist_10.pkl.gz',
              'pretraining_epochs': 25,
              'pretrain_lr_base': 0.0001,
              'mu': 0.9,
              'finetune_lr': 0.0001,
              'training_epochs': 5,
              'dataset': dataset,
              'batch_size': 96,
              'nClass': K,
              'hidden_dim': [500, 500, 2000, 10],
              'diminishing': False}
    results = []
    for i in range( trials ):
        res_metrics = test_SdC( **config )
        results.append( res_metrics )
    results_SAEKM = np.zeros( (trials, 3) )
    results_DCN = np.zeros( (trials, 3) )
    N = config['training_epochs'] / 5
    ##training_epochs is 250,so N is 50
    for i in range( trials ):
        results_SAEKM[i] = results[i][0]
        results_DCN[i] = results[i][N]
    SAEKM_mean = np.mean( results_SAEKM, axis=0 )
    SAEKM_std = np.std( results_SAEKM, axis=0 )
    DCN_mean = np.mean( results_DCN, axis=0 )
    DCN_std = np.std( results_DCN, axis=0 )
    results_all = np.concatenate( (DCN_mean, DCN_std, SAEKM_mean, SAEKM_std),axis=0 )
    print( results_all )
    SSE = Process().elbow(path,data)
    result.append( SSE )
    DataModule().save_array( np.array( result ), path+'result.npy' )
    print( 'K=', K )
    print( 'SSE is:', SSE )
    print( result )