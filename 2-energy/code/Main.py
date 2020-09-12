# -*- coding: utf-8 -*-
"""
Created on Sunday,3 November 17:32 2019
@author:Jeaten
"""
import os,time
import Function as F
path='./File/'
F.Model_path='./File/Model/'
file=path+'Trajectory.pkl'
window=10
test_day=8
time_start=time.time()
########################### cluster module ###################################
print('start generating training data...')
Trajectory=F.Trajectory=F.DataModule().load_pkl(file)
'''
stand,test=F.DataModule().distribution(stand,test_day=test_day)
norm=F.DataModule().normalize(stand)
train=F.DataModule().generate_train(norm)
train=F.DataModule().trans_pkl_python2(train,path+'train.pkl')
time_temp1=time.time()
print('training data has generated...','cost:',time_temp1-time_start,'s')
os.system("gzip ./File/train.pkl")
print('start clustering...')
time_temp1=time.time()
os.system("THEANO_FLAGS='floatX=float32' python2 Cluster.py")
time_temp2=time.time()
print('clustering has been done... this part cost:',(time_temp2-time_temp1)/60,'mins')
######################### train Lstm module #########################################
fin_id=F.DataModule().load_list(path+'fin_idx.npy')
stand=F.DataModule().load_pkl(path+'train_stand.pkl')
query=F.Process().catalogue(fin_id,stand)
Fit=F.Process().generate_fit(stand,query,window)
# F.DataModule().save_pkl(Fit,path+'Fit.pkl')
time_temp1=time.time()
print('Start training lstm ...')
for Class in Fit:
    F.Lstm().train(Fit[Class]['feature'],Fit[Class]['label'],Class)
time_temp2=time.time()
print('Lstm has been trained ... this part cost:',(time_temp2-time_temp1)/60,'mins')

################################# predict module #################################
test=F.DataModule().load_pkl(path+'test_set.pkl')
query=F.DataModule().load_pkl(path+'Query.pkl')
print('loading lstm model...')
time_temp1=time.time()
model={i:F.Lstm().load(i) for i in range(0,13)}
time_temp2=time.time()
print('lstm models have loaded... this part cost:',(time_temp2-time_temp1)/60,'mins')
print('starting predict...')
time_temp1=time.time()
Original,Prediction=F.Lstm().predict(test,model,query,window)
F.DataModule().save_pkl(Original,path+'Original_res.pkl')
F.DataModule().save_pkl(Prediction,path+'Prediction_res.pkl')
time_temp2=time.time()
print('prediction has finished... this part cost:',(time_temp2-time_temp1)/60,'mins')
'''
############################### correspondence module ############################
Original=F.DataModule().load_pkl(path+'Original_res.pkl')
Prediction=F.DataModule().load_pkl(path+'Prediction_res.pkl')
stand=F.DataModule().load_pkl(path+'Tu_Veh.pkl')
F.Query=F.DataModule().load_pkl(path+'Query.pkl')
Can_send=F.Correspondence().allocation(Original,Prediction)
F.Correspondence().direct(Original,stand)
F.Correspondence().strategy(Original,Can_send,test_day,stand)
# print('Sent trajectories:',len(F.Sta_sent))
# rate=[]
# for key in F.Sta_sent:
#     print(F.Sta_sent[key])
#     rate.append(len(set(F.Sta_sent[key]))/len(Trajectory[8][key[0]][key[1]]))
# # for k in F.Sta_sent_clu:
# #     print(k,':',len(set(F.Sta_sent_clu[k])),end=',')
# print()
# print("Rate:",sum(rate)/123027)
time_end=time.time()
print('total time:',(time_end-time_start)/60,'mins')