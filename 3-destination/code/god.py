# -*- coding: utf-8 -*-
"""
this is the main function used to implement "Segmented Trajectory Clustering-Based
Destination Prediction in IoVs"
created at 20:03, March 24,2020
@author: Jeaten
@email: ljt_IT@163.com
"""
import time
import multiprocessing
import library as fun
dat=fun.Data()
seg=fun.Segment()
clu=fun.Cluster()
pre=fun.Predict()
Com=6
Layer=1
path='./file/'
path_model='./file/model_god/com_'+str(Com)+'/'
Len_padding=35 # the maximum length of feature,can be gotten form pre.get_len_padding(fea)
Record=[] # to record the return values of multiprocess
def extract():
    ### extract data from csv file
    print("extracting trajectory from csv file...")
    test_ext=dat.extract_data(path+'test.csv')
    dat.save_pkl(test_ext,path+'test_ext',make_dir=path)
    train_ext=dat.extract_data(path+'train.csv')
    dat.save_pkl(train_ext,path+'train_ext',make_dir=path)
def filter():
    ### filter trajectory by length
    print("filtering trajectory by length( in rang of",dat.num_min_tra,"and",dat.num_max_tra,")...")
    test_p=dat.load_pkl(path+'test_ext')
    test_p=dat.filter_length(test_p)
    dat.save_pkl(test_p,path+'test_p')
    train_p=dat.load_pkl(path+'train_ext')
    train_p=dat.filter_length(train_p)
    dat.save_pkl(train_p,path+'train_p')
def segment():
    ### segment trajectory
    print("segmenting trajectories...")
    seg.threshold=200/97082.78749247293
    test_p=dat.load_pkl(path+'test_p')
    test_seg=seg.segmenting(test_p)
    dat.save_pkl(test_seg,path+'test_seg')
    train_p=dat.load_pkl(path+'train_p')
    train_seg=seg.segmenting(train_p)
    dat.save_pkl(train_seg,path+'train_seg')
def cluster():
    ### trajectory cluster
    print("clustering trajectories...")
    clu.dis_threshold = 80 / 97082.78749247293
    ### test is used for debugging, you don't need to cluster test set in reality
    # test_seg = dat.load_pkl( path + 'test_seg' )
    # res_clu_test = clu.cluster( test_seg )
    # dat.save_pkl( res_clu_test, path + 'res_clu_test' )
    train_seg = dat.load_pkl( path + 'train_seg' )
    res_clu_train = clu.cluster( train_seg )
    dat.save_pkl( res_clu_train, path + 'res_clu_train' )
    return
def record(value_re):
    '''
    to record the return value while classifying
    :param value_re: the returned classification result
    :return: None
    '''
    Record.append(value_re)
def save_model(combination):
    '''
    to save the model we have trained
    :param combination: returned combination of the model name and model itself
    :return: None
    '''
    model,name=combination
    dat.save_pkl(model,path_model+name,make_dir=path_model)
def classify_multi(tra_seg,unit,res_clu):
    '''
    to classify with multiprocess
    :param tra_seg: the segmented trajectories
    :param unit: the number of trajectories in each task
    :param res_clu: the result of clustering
    :return: the classified trajectories
    '''
    ### divide it into multi tasks
    mask = []
    temp_dic = {}
    for k in tra_seg:
        temp_dic[k] = tra_seg[k]
        if (k + 1) % unit == 0 or k == list( tra_seg.keys() )[-1]:
            mask.append( temp_dic )
            temp_dic = {}
    pool = multiprocessing.Pool( mask.__len__() )
    for m in mask:
        pool.apply_async( clu.classify, args=(m, res_clu,), callback=record )
    pool.close()
    pool.join()
    ### merge multi tasks
    tra_class_temp={}
    for tra in Record:
        try:
            for k in tra:
                tra_class_temp[k]=tra[k]
        except:
            print(k,tra[k])
    tra_class={}
    for k in sorted(tra_class_temp):
        tra_class[k]=tra_class_temp[k]
    return tra_class
    #####
def completion(tra_ori,center,prefix):
    '''
    split the trajectory with completion
    :param tra_ori: the original trajectory data
    :param center: the clustering center
    :param prefix: the data prefix('train' or 'test')
    :return: None
    '''
    tra_com_p=[]
    percentage=[0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for per in percentage:
        tra_per={}
        for k in tra_ori:
            len=int(tra_ori[k].__len__()*per)
            tra_per[k]=tra_ori[k][:len]
        tra_com_p.append(tra_per)
    dat.save_pkl(tra_com_p,path+prefix+'_completion_p')
    tra_com_p=dat.load_pkl(path+prefix+'_completion_p')
    seg.threshold = 200 / 97082.78749247293
    tra_com_seg=[]
    for tra in tra_com_p:
        tra_com_seg.append(seg.segmenting(tra))
    dat.save_pkl( tra_com_seg, path + prefix+'_completion_s' )
    tra_com_c=[]
    for com_seg in tra_com_seg:
        tra_com_c.append(classify_multi(com_seg,1500,center))
    dat.save_pkl(tra_com_c,path + prefix+'_completion_c')
def train(train_class,center,lab_true,layer=1,com=''):
    '''
    train the model with multiprocess or the final model 
    :param train_class: the trajectories represented with classed
    :param center: the clustering center
    :return: None
    '''
    epoch_ini = 1000
    batch_ini = 64
    layer_ini = layer
    neuron_ini = 128
    validation_ini = 0
    act_ini='tanh'
    loss_ini='mse'
    print( 'training model...' )
    train_rep=pre.represent(train_class,center)
    # fea, lab = pre.get_fea_lab(train_rep) ### when completion is 100%
    ###
    fea=pre.get_feature(train_rep) ### when completion is less than 100%
    lab=lab_true
    ###
    fea=pre.padding(fea,Len_padding )
    ### train the model you want
    # model, name = pre.train_model( (fea, lab), epoch_ini, batch_ini, layer_ini, neuron_ini, validation_ini, act_ini,loss_ini )
    # dat.save_pkl( model, path_model +diff+'_'+ name, make_dir=path_model )
    # print(path_model +diff+'_')
    ##### start: the multiprocess part
    ## monitor the batch size with multiprocess
    # batch = [10, 32, 64, 100, 200]
    # pool = multiprocessing.Pool( batch.__len__() )
    # for i in range( batch.__len__() ):
    #     arg = ((fea, lab), epoch_ini, batch[i], layer_ini, neuron_ini, validation_ini, act_ini, loss_ini)
    #     pool.apply_async( pre.train_model, args=arg, callback=save_model )
    ### monitor layer with multiprocess
    # layer=[1,2,3,4,5]
    # pool = multiprocessing.Pool( layer.__len__() )
    # for i in range( layer.__len__() ):
    #     arg = ((fea, lab), epoch_ini, batch_ini, layer[i], neuron_ini, validation_ini, act_ini, loss_ini)
    #     pool.apply_async( pre.train_model, args=arg, callback=save_model )
    ### monitor the number of neurons with multiprocess
    neuron=[16,32,64,128,256]
    pool = multiprocessing.Pool(neuron.__len__() )
    for i in range( neuron.__len__() ):
        arg = ((fea, lab), epoch_ini, batch_ini, layer_ini, neuron[i], validation_ini, act_ini, loss_ini)
        pool.apply_async( pre.train_model, args=arg, callback=save_model )
    # ### if you use the multiprocess, the following statements are needed
    pool.close()
    pool.join()
    ##### end: the multiprocess part
def predict(tra_p,tra_class,center,model,des_true):
    '''
    to predict the location of final destination position and calculate the mean error between them
    :param tra_p: the completed trajectory 
    :param tra_class: the trajectory represented with classes
    :param center: the clustering center
    :param model: the trained prediction model
    :param des_true: the actual destination position of each trajectory
    :return: the mean error(km)
    '''
    tra_rep=pre.represent(tra_class,center)
    fea=pre.get_feature(tra_rep)
    fea=pre.padding( fea, Len_padding )
    res=model.model.predict(fea)
    lab_pre = pre.reverse(res,center )
    error = []
    count = 0
    for k in tra_p:
        cur=tra_p[k][-1]
        c=lab_pre[count]
        dis=[]
        ### find the nearest location as the predicted destination
        for p in center[c][1]:
            dis.append(pre.mean_hav_dis(cur,p))
        index=dis.index(min(dis))
        ###
        des_act=des_true[k] # the actual destination position
        des_pre=center[c][1][index] # the predicted destination position
        error.append(pre.mean_hav_dis(des_act,des_pre))
        count += 1
    return sum( error )/error.__len__()
if __name__ == '__main__':
    time_s=time.time()
    ##################################################################
    # Each part of the following 4 functions is a whole,
    # which can be executed separately, but must be executed in order,
    # otherwise the necessary files will be missing
    ### extract and filter data, segment and cluster trajectories
    # extract()
    # filter()
    # segment()
    # cluster()
    # ### classify trajectories with multiprocess
    # tra_seg_all=dat.load_pkl(path+'train_seg')
    # tra_seg={}
    # for k in tra_seg_all:
    #     tra_seg[k]=tra_seg_all[k]
    #     if k>30000:break # we only choose 30000 trajectories to simulate
    # center=dat.load_pkl(path+'res_clu_train_30000')
    # test_seg=dat.load_pkl(path+'test_seg')
    # # about test set
    # test_class=classify_multi(test_seg,30,center)
    # dat.save_pkl( test_class, path + 'test_class_30000' )
    # # about train set
    # tra_class=classify_multi(tra_seg,1500,center)
    # dat.save_pkl(tra_class,path+'tra_class_30000')
    ### extract trajectories with completion
    # tra_p = dat.load_pkl( path + 'test_p' )#train
    # center = dat.load_pkl( path + 'res_clu_train_30000' )#train
    # completion( tra_p, center, 'test' )
    ### train the model when the completion is 100%
    # tra_class = dat.load_pkl( path + 'tra_class_30000' )
    # center = dat.load_pkl( path + 'res_clu_train_30000' )
    # path_label_fin=path+'lab_fin_train'
    # lab_true=dat.load_pkl(path_label_fin)
    # # train(tra_class,center,path_label_fin,1)
    # # # ### train the model when the completion is less than 100%
    # com=8
    # tra_class_c=dat.load_pkl(path+'com/train_completion_c_'+str(com))
    # train(tra_class_c,center,lab_true,layer=1,com=str(com))

    ### predict destination with different models when completion is 100%
    test_class = dat.load_pkl( path + 'tra_class_30000' )
    com=Com
    tra_c_com={}
    for k in test_class:
        split=int(test_class[k].__len__()*(com/10)+0.5)
        if split>0:
            tra_c_com[k]=test_class[k][:split]
            print(k,test_class[k].__len__(),split,test_class[k][:split],test_class[k])
        else:
            tra_c_com[k] =test_class[k][:1]
            # print(k,test_class[k].__len__(),split,test_class[k][:1],test_class[k])
    center = dat.load_pkl( path + 'res_clu_train_30000' )
    path_label_fin = path + 'lab_fin_train'
    lab_true = dat.load_pkl( path_label_fin )
    # # ### train the model when the completion is less than 100%
    ## tra_class_c = dat.load_pkl( path + 'com/train_completion_c_' + str( com ) )
    tra_class_c=tra_c_com
    train( tra_class_c, center, lab_true, layer=Layer, com=str( com ) )
    ### predict destination with the final model when completion is less than 100%
    # test_class = dat.load_pkl( path + 'test_class_30000' )
    # center = dat.load_pkl( path + 'res_clu_train_30000' )
    # des_true = dat.load_pkl( path + 'des_test' )
    # test_p = dat.load_pkl( path + 'test_p' )
    # test_com_p=dat.load_pkl(path+'test_completion_p')
    # test_com_c=dat.load_pkl(path+'test_completion_c')
    # name_model = path + 'model_batch/' + '1_128_64'
    # model = dat.load_pkl( name_model )
    # com=[5,6,7]
    # for i in com:
    #     error = predict(test_com_p[i-3],test_com_c[i-3], center, model,des_true )  # when the trajectory completion is less than 100%
    #     print( 'trajectory completion ' + str(i * 10 ) + '%, mean error: ', error, 'km' )
    # error = predict( test_p, test_class, center, model, des_true )  # when the trajectory completion is 100%
    # print('trajectory completion 100%, mean error:', error, 'km' )
    ##################################################################
    time_e=time.time()
    print('time you used: '+str((time_e-time_s)/60)+' minutes')
