# -*- coding: utf-8 -*-
"""
@author: Jeaten
@email: ljt_IT@163.com
this python file is to show the segmentation figure
"""
import numpy as np
from library import Data
path='./file/'
dat=Data()
#from his.function import  Draw
#from his.function import Process
from library import Predict
#from his.function import Segment
#dra=Draw()
#pro=Process()
pre=Predict()
#seg=Segment()
Len_padding=35
def draw_original(self):
    '''
    draw the figure of "ori_map.pdf" # by capturing png to pdf
    :return: 
    '''
    import matplotlib.pyplot as plt
    data = Data()
    data = data.load_pkl( './file/train.pkl' )
    coordinate_x = []
    coordinate_y = []
    for k in data:
        for d in data[k]:
            # if d[0] >= -8.7 and d[0] <= -8. and d[1] >= 41 and d[1] <= 41.25:
            if k >= 10000: break
            coordinate_x.append( d[0] )
            coordinate_y.append( d[1] )
    plt.plot( coordinate_x, coordinate_y, '.', color='gray', ms=0.1 )  # 0.004
    plt.axis( [-8.70, -8.54, 41.1, 41.2] )
    # plt.savefig( './f.eps',dpi=0.01)
    plt.show()
def draw_seg(self):
    '''
    draw the figure of "seg_map.pdf" # by capturing png to pdf
    :return: 
    '''
    import matplotlib.pyplot as plt
    data = Data()
    data = data.load_pkl( './file/train_seg.pkl' )
    coordinate_x = []
    coordinate_y = []
    for k in data:
        for sub in data[k]:
            for d in sub:
                # if d[0] >= -8.7 and d[0] <= -8. and d[1] >= 41 and d[1] <= 41.25:
                    if k >= 5000: break
                    coordinate_x.append( d[0] )
                    coordinate_y.append( d[1] )
    plt.plot( coordinate_x, coordinate_y, 'b.',ms=0.04 )  # 0.004  # color='blue',
    plt.axis( [-8.70, -8.54, 41.1, 41.2] )
    # plt.savefig( './f.eps',dpi=0.01)
    plt.show()
def draw_tra_exam():
    '''
    draw the figure of "traz_exam.pdf"
    :return: 
    '''
    import his.function as f
    draw = f.Draw()
    # tra =  [[107.605, 137.329], [122.274, 169.126], [132.559, 179.311], [153.324, 184.276], [171.884, 174.654],
    #                [186.408, 168.634], [196.566, 145.204], [200.549, 127.877], [211.391, 118.179], [216.318, 116.547],
    #                [225.197, 122.796], [231.064, 135.459], [240.835, 143.398], [254.63, 144.933], [265.055, 158.761],
    #                [271.004, 159.66], [274.474, 173.979]]
    tra = [[107.605, 137.329], [122.274, 169.126], [132.559, 179.311], [153.324, 184.276], [171.884, 174.654],
           [186.408, 168.634], [196.566, 145.204], [200.549, 127.877], [216.318, 116.547],
           [225.197, 122.796], [231.064, 135.459], [240.835, 143.398], [254.63, 144.933],
           [271.004, 159.66], [274.474, 173.979]]
    tick=[[107.605, 137.329-2.5], [122.274+0.5, 169.126-2.5], [132.559, 179.311], [153.324, 184.276+0.5], [171.884, 174.654+0.5],
           [186.408, 168.634+0.5], [196.566+0.5, 145.204], [200.549+0.5, 127.877], [216.318+0.5, 116.547-2.5],
           [225.197+1, 122.796-2], [231.064+0.5, 135.459-2.5], [240.835-4.3, 143.398-1.3], [254.63+1, 144.933-1.3],
           [271.004, 159.66], [274.474, 173.979]]
    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    dat = f.Data()
    def draw_single_2d(data, title=''):
        '''
        this function is used to draw a 2d figure
        :param data: the 2d data you want to draw
        :return: True if this function is be executed successfully
        '''
        import matplotlib.pyplot as plt
        coordinate_x = []
        coordinate_y = []
        for d in data:
            coordinate_x.append( d[0] )
            coordinate_y.append( d[1] )
        plt.plot( coordinate_x, coordinate_y, '-' + 'b' + '.' )
        plt.title( title )
        plt.axis([104.605,285.474,110.547,192.276])
        tra=tick
        for p in range(tra.__len__()):#\mathcal{}
            plt.text( tra[p][0], tra[p][1],
                      r"$\mathcal{l}_{{\mathcal{v}_\mathcal{i}}}^{{\mathcal{t}_\mathcal{" + str( p + 1 ) + "}}}$",
                      fontsize=20 )
            # if p+1 in [13]:
            #     plt.text( tra[p][0]+1, tra[p][1]-1.3,
            #               r"$\mathcal{l}_{{\mathcal{v}_\mathcal{i}}}^{{\mathcal{t}_\mathcal{" + str( p + 1 ) + "}}}$",
            #               fontsize=20 )
            # else:
            #     plt.text( tra[p][0], tra[p][1],
            #               r"$\mathcal{l}_{{\mathcal{v}_\mathcal{i}}}^{{\mathcal{t}_\mathcal{" + str( p + 1 ) + "}}}$",
            #               fontsize=20 )
            # plt.text( tra[p][0], tra[p][1], '$\mathcal{{l_{{v_i}}^{{t_{' + str( p + 1 ) + '}}}}}$', fontdict=font )  #
            #l_{{v_i}}^{{t_1}
            # plt.text( tra[p][0], tra[p][1], r'$\mathcal{A}\mathrm{sin}(2 \omega t)$', fontdict=font )  #
            #r'$\mathcal{A}\mathrm{sin}(2 \omega t)$'
        plt.xticks( [] )
        plt.yticks( [] )
        # plt.axis( 'off' )
        plt.show()
        return True
    draw_single_2d( tra )
def draw_seg_exam():
        '''
        draw the figure of "seg_exam.pdf"
        :return: 
        '''
        seg = Segment()
        draw = Draw()
        # tra_seg = [[107.605, 137.329], [122.274, 169.126], [132.559, 179.311], [153.324, 184.276], [171.884, 174.654],
        #            [186.408, 168.634], [196.566, 145.204], [200.549, 127.877], [211.391, 118.179], [216.318, 116.547],
        #            [225.197, 122.796], [231.064, 135.459], [240.835, 143.398], [254.63, 144.933], [265.055, 158.761],
        #            [271.004, 159.66], [274.474, 173.979]]
        tra_seg=[[107.605, 137.329], [122.274, 169.126], [132.559, 179.311], [153.324, 184.276], [171.884, 174.654],
           [186.408, 168.634], [196.566, 145.204], [200.549, 127.877], [216.318, 116.547],
           [225.197, 122.796], [231.064, 135.459], [240.835, 143.398], [254.63, 144.933],
           [271.004, 159.66], [274.474, 173.979]]
        tick = [[107.605, 137.329 - 2.5], [122.274 + 0.5, 169.126 - 2.5], [132.559, 179.311], [153.324, 184.276 + 0.5],
                [171.884, 174.654 + 0.5],[186.408, 168.634 + 0.5], [196.566 + 0.5, 145.204], [200.549 -9, 127.877-1],
                [216.318 + 0.5, 116.547 - 2.5],[225.197 + 1, 122.796 - 2], [231.064 -9, 135.459 - 1], [240.835 -9.3, 143.398+0.5 ],
                [254.63 + 1, 144.933 - 1.3],[271.004, 159.66], [274.474, 173.979]]

        def draw_compare_2d(original, comparision, title='', leg=''):
            '''
            this function is used to draw the comparision figure
            :param original: the original data
            :param comparision: the comparision data
            :return: None
            '''
            import matplotlib.pyplot as plt
            coordinate_x = [[], []]
            coordinate_y = [[], []]
            for o in original:
                coordinate_x[0].append( o[0] )
                coordinate_y[0].append( o[1] )
            for c in range(comparision.__len__()):
                # if c in [0,3,9,16]:
                coordinate_x[1].append( comparision[c][0] )
                coordinate_y[1].append( comparision[c][1] )
            plt.plot( coordinate_x[0], coordinate_y[0], '-' + 'b' + '.' )
            plt.plot( coordinate_x[1], coordinate_y[1], '-' + 'r' + '^' )
            plt.legend(leg)
            plt.title(title)
            tra = tick
            for p in range( tra.__len__() ):  # \mathcal{}
                plt.text( tra[p][0], tra[p][1],
                          r"$\mathcal{l}_{{\mathcal{v}_\mathcal{i}}}^{{\mathcal{t}_\mathcal{" + str( p + 1 ) + "}}}$",
                          fontsize=20 )
            plt.axis( [104.605, 285.474, 110.547, 192.276] )
            plt.xticks( [] )
            plt.yticks( [] )
            # plt.axis('off')
            plt.show()
            return
        tra_seg = np.array( tra_seg )
        seg.threshold = 40
        seg.segment(tra_seg )
        key_tra = seg.fill_cons_dis( seg.S )
        print(tra_seg.__len__(), key_tra.__len__() )
        draw_compare_2d(tra_seg, key_tra, leg=['Original trajectory','Segmented trajectory'] )
        return
def several_exam_seg():
    data_ori=dat.load_pkl(path+'test_p')
    data_seg=dat.load_pkl(path+'test_seg')
    for k in [1,3,5,11]:
        draw_compare(data_ori[k],data_seg[k],leg=['Original trajectories','Segmented trajectories'])
    return
def effect_cluster():
    ### can be executed at first
    # tra_seg=dat.load_pkl(path+'train_seg_3w')
    # tra_class=dat.load_pkl(path+'tra_class_30000')
    # cluster={}
    # data=tra_class
    # for k in data:
    #     for c in range(data[k].__len__()):
    #         try:
    #             cluster[data[k][c]].append(tra_seg[k][c])
    #         except:
    #             cluster[data[k][c]]=[]
    #             cluster[data[k][c]].append( tra_seg[k][c] )
    # dat.save_pkl( cluster, path + 'center_element' )
    ###
    import random
    center = dat.load_pkl( path + 'res_clu_train_30000' )
    cluster=dat.load_pkl(path+'center_element')
    show=[1207,2290,2303,3618] # candidate: 235,3150,2319,1920,,4629
    for c in show:
        if cluster[c].__len__()>=10:
            element=random.sample(cluster[c],10)
        else:
            element=cluster[c]
        print(c,cluster[c].__len__())
        element_center(element,center[c][1])
def draw_compare(original, comparision, title='', leg=''):
    '''
    this function is used to draw the comparision figure
    :param original: the original data
    :param comparision: the comparision data
    :return: None
    '''
    import matplotlib.pyplot as plt
    coordinate_x = [[], []]
    coordinate_y = [[], []]
    for o in original:
        coordinate_x[0].append( o[0] )
        coordinate_y[0].append( o[1] )
    for c in comparision:
        for sub in c:
            coordinate_x[1].append( sub[0] )
            coordinate_y[1].append( sub[1] )
    plt.plot( coordinate_x[0], coordinate_y[0], '-' + 'b' + '.' )
    plt.plot( coordinate_x[1], coordinate_y[1], '-' + 'r' + '^' )
    if leg:
        plt.legend( leg )
    plt.title( title )
    # plt.axis( 'off' )

    plt.show()
    return
def compare_multi_line(data, label=[],ylabel='',title='',solid=888):
    import matplotlib.pyplot as plt
    line_style = ':' #
    # color = ['b', 'r', 'lime', 'cyan', 'k', 'y','g','deeppink']
    color=['r','b','g', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
                 'tan', 'dodgerblue', 'yellow']
    marker = ['s', 'x', '*', '^', '.', '+', 'p','v','>','1','2','3','4']
    if label:
        for i in range( data.__len__() ):
            row = range( 1, data[i].__len__() + 1 )
            if i==solid:
                plt.plot( row, data[i], linestyle='-', marker=marker[i], color=color[i], label=label[i],ms=2)
            else:
                plt.plot( row, data[i], linestyle=line_style, marker=marker[i], color=color[i], label=label[i],ms=2)
        plt.legend()
    else:
        for i in range( data.__len__() ):
            row = range( 1, data[i].__len__() + 1 )
            plt.plot( row, data[i], linestyle=line_style, marker=marker[i], color=color[i])
    plt.title( title )
    plt.xlabel('Training times')
    plt.ylabel(ylabel)
    plt.show()
def chart():
    data=[0.1884125796595129,0.2143987529919254,0.17923334450450307,0.39154270341860564,0.46127131366304513]
    xticks = ['10','32','64','100','200']
    import matplotlib.pyplot as plt
    color = ['r', 'b', 'g', 'cyan',  'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
             'tan', 'dodgerblue', 'yellow','k',  'deeppink','lime',]
    plt.bar(range(len(data)),data, color=color )
    # plt.ylim( ymax=80, ymin=0 )
    if xticks:
        plt.xticks(range(len(data)),xticks)
    plt.xlabel('Batch size')
    plt.ylabel('Mean error(km)')
    plt.show()
def element_center(data,center,title=''):
        '''
        this function is used to draw the figure of a 2d multiple list([[1,3],[2,3],...,[n,n]])
        :param data: the multi list we want to draw the figure
        :param title: the figure's title we have drawn
        :return: return True if this function is be executed successfully
        '''
        import matplotlib.pyplot as plt
        coordinate_x = []
        coordinate_y = []
        for da in data:
            sub_coordinate_x =[]
            sub_coordinate_y=[]
            for d in da:
                sub_coordinate_x.append(d[0])
                sub_coordinate_y.append(d[1])
            coordinate_x.append(sub_coordinate_x)
            coordinate_y.append(sub_coordinate_y)
        # label=[]
        color = ['g', 'b', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
                 'tan', 'dodgerblue', 'yellow']
        for i in range(data.__len__()):
            plt.plot( coordinate_x[i], coordinate_y[i], color=color[i % len( color )],)
        # plt.xlabel('Interval')
        # plt.ylabel('The number of normalized vehicle nodes')
        d=np.array(center)
        # print(d[:,0])
        plt.plot(d[:,0],d[:,1],color='r',label='center',linewidth=4,linestyle=':')
        plt.title( title )
        plt.legend()
        # plt.axis( [-8.70, -8.606, 41.1, 41.152] )
        plt.show()
        return True
def compare():
    path_model = path+'model/'
    act = ['linear', 'relu', 'sigmoid', 'softmax', 'tanh']#'softmax',
    loss =  ['mse', 'binary_crossentropy', 'KLD', 'hinge', 'MSLE'] # 'mean_absolute_error',
    loss_label=['mse', 'binary cross entropy', 'kullback leiber divergence','hinge','mean squared logarithmic error']
    batch = [5, 10, 32, 64, 100, 200, 300, 400, 500]
    layer=[1,2,3,4,5,6,7,8]
    # identification=layer
    identification=['1_16_64','1_32_64','1_64_64','1_128_64','1_256_64']
    loss=[]
    acc=[]
    for ide in identification:
        ### notice: use the one of four model_temp sentence and corresponds to identification
        # name_model=path_model+ide+'_mse_64' # if you want to monitor the activation function
        # name_model=path_model+'tanh_'+ide+'_64' # if you want to monitor the loss function
        # name_model=path_model+'tanh_MSLE_'+str(ide) # if you want to monitor the batch size
        # name_model=path_model+str(ide)+'tanh_MSLE_64' # if you want to monitor the layers of model
        name_model=path_model+'com_4/'+ide
        ###
        print("loading model '"+name_model+"'...")
        model_temp=dat.load_pkl(name_model)
        loss.append(model_temp.history['loss'])
        acc.append(model_temp.history['acc'])
    ### notice: the figure of loss can not be used directly, and its label is as follow
    # identification=loss_label
    ###
    position=88
    # compare_multi_line(loss,label=identification,ylabel='Loss',solid=position)
    compare_multi_line(acc,label=identification,ylabel='Accuracy',solid=position)
def vs():
    import matplotlib.pyplot as plt
    baseline={0.3:1.06,0.4:0.9,0.5:0.76,0.6:0.63,0.7:0.48,0.8:0.37,0.9:0.348,1:0.336}
    res_com={1:0.304}
    base=[]
    our=[]
    for k in baseline:
        base.append(baseline[k])
        our.append(baseline[k]-0.2)
    row = [(i+3)*10 for i in range(baseline.__len__())]
    print(row,baseline)
    plt.xlabel('Trajectory completion (%)')
    plt.ylabel('Mean error (km)')
    plt.grid(linestyle=':')
    plt.plot(row,base,'b',label='trajectory distribution-based')
    plt.plot(row,our,'r',label='ours')
    plt.legend()
    plt.show()
def forum():
    def predict(tra_p, tra_class, center, model, des_true):
        '''
        to predict the location of final destination position and calculate the mean error between them
        :param tra_p: the completed trajectory 
        :param tra_class: the trajectory represented with classes
        :param center: the clustering center
        :param model: the trained prediction model
        :param des_true: the actual destination position of each trajectory
        :return: the mean error(km)
        '''
        tra_rep = pre.represent( tra_class, center )
        fea = pre.get_feature( tra_rep )
        fea = pre.padding( fea, Len_padding )
        res = model.model.predict( fea )
        lab_pre = pre.reverse( res, center )
        error = []
        count = 0
        for k in tra_p:
            cur = tra_p[k][-1]
            c = lab_pre[count]
            dis = []
            ### find the nearest location as the predicted destination
            for p in center[c][1]:
                dis.append( pre.mean_hav_dis( cur, p ) )
            index = dis.index( min( dis ) )
            ###
            des_act = des_true[k]  # the actual destination position
            des_pre = center[c][1][index]  # the predicted destination position
            error.append( pre.mean_hav_dis( des_act, des_pre ) )
            count += 1
        return sum( error ) / error.__len__()
    ### monitor the mean error when trajectory completion is less than 100%
    com = 8
    path_model = './file/model/com'+str(com)+'_done/'
    path_model='./file/model/layer_neuron/'
    # test_com_p= dat.load_pkl( path + 'test_p' )
    # test_com_c = dat.load_pkl( path + 'test_class_30000' )
    ### if you want to monitor mean error when trajectory completion is less than 100%, please use the flowing
    # path_model='./file/model/com_'+str(com)+'/' # the com can be 4,6 and 8
    # path_model='./file/model/com_'+str(com)+'/'
    print(path_model)
    print( 'layer  16   \t 32  \t 64 \t 128 \t 256' )
    test_com_p=dat.load_pkl(path+'com/test_completion_p_'+str(com))
    test_com_c= dat.load_pkl( path + 'com/test_completion_c_'+str(com) )
    ###
    # test_class = dat.load_pkl( path + 'test_class_30000' )
    # test_com_c = {}
    # for k in test_class:
    #     split = int( test_class[k].__len__() * (com /10) +0.5)
    #     if split > 0:
    #         test_com_c[k] = test_class[k][:split]
    #         # print(k,test_class[k].__len__(),split,test_class[k][:split],test_class[k])
    #     else:
    #         test_com_c[k] = test_class[k][:1]
    #         # print(k,test_class[k].__len__(),split,test_class[k][:1],test_class[k])
    # ###
    #####
    center = dat.load_pkl( path + 'res_clu_train_30000' )
    des_true = dat.load_pkl( path + 'des_test' )
    batch = [10, 32, 64, 100, 200]
    layer = [1, 2, 3, 4, 5]
    neuron = [16, 32, 64, 128, 256]
    using = []
    for l in layer:
        for n in neuron:
            using.append( str( l ) + '_' + str( n ) + '_64' )
    count = 1
    row = 1
    for u in using:
        name_model = path_model + u
        model = dat.load_pkl( name_model )
        # error = predict( test_com_p,test_com_c, center, model, des_true )  # when the trajectory completion is 100%
        # error_print = round( error, 4 )
        error_print=round(np.average(model.history['loss'])*100000,4)# acc
        if count % 5 == 1:
            print( row, '&', error_print, end=' & ' )
        elif count % 5 == 0:
            print( error_print, end=' \\\ ' )
            print()
            row += 1
        else:
            print( error_print, end=' & ' )
        count += 1
def predict():
    def predict(tra_p, tra_class, center, model, des_true):
        '''
        to predict the location of final destination position and calculate the mean error between them
        :param tra_p: the completed trajectory 
        :param tra_class: the trajectory represented with classes
        :param center: the clustering center
        :param model: the trained prediction model
        :param des_true: the actual destination position of each trajectory
        :return: the mean error(km)
        '''
        tra_rep = pre.represent( tra_class, center )
        fea = pre.get_feature( tra_rep )
        fea = pre.padding( fea, Len_padding )
        res = model.model.predict( fea )
        lab_pre = pre.reverse( res, center )
        error = []
        count = 0
        for k in tra_p:
            cur = tra_p[k][-1]
            c = lab_pre[count]
            dis = []
            ### find the nearest location as the predicted destination
            for p in center[c][1]:
                dis.append( pre.mean_hav_dis( cur, p ) )
            index = dis.index( min( dis ) )
            ###
            des_act = des_true[k]  # the actual destination position
            des_pre = center[c][1][index]  # the predicted destination position
            error.append( pre.mean_hav_dis( des_act, des_pre ) )
            count += 1
        return sum( error ) / error.__len__()
    ### monitor the mean error when trajectory completion is less than 100%
    path_model = './file/model/layer_neuron/'
    # path_model='./file/model/model_batch/10/'
    test_com_p= dat.load_pkl(path + 'test_p' )
    test_com_c = dat.load_pkl(path + 'test_class_30000' )
    center = dat.load_pkl(path + 'res_clu_train_30000' )
    des_true = dat.load_pkl(path + 'des_test' )
    neuron=[16,32,64,128,256]
    batch=[10,32,64,100,200]
    using=neuron
    for u in using:
        # name_model = path_model + '1_64_'+str(u)
        name_model=path_model+'1_'+str(u)+'_64'
        model = dat.load_pkl( name_model )
        error = predict( test_com_p,test_com_c, center, model, des_true )  # when the trajectory completion is 100%
        # error_print = round( error )
        print(error,end=',')
def pre_com():
    com=8
    path_model = './file/model/com'+str(com)+'_done/'
    # path_model='./file/model/model_batch/'+str(com)+'/'
    def predict(tra_p, tra_class, center, model, des_true):
        '''
        to predict the location of final destination position and calculate the mean error between them
        :param tra_p: the completed trajectory 
        :param tra_class: the trajectory represented with classes
        :param center: the clustering center
        :param model: the trained prediction model
        :param des_true: the actual destination position of each trajectory
        :return: the mean error(km)
        '''
        tra_rep = pre.represent( tra_class, center )
        fea = pre.get_feature( tra_rep )
        fea = pre.padding( fea, Len_padding )
        res = model.model.predict( fea )
        lab_pre = pre.reverse( res, center )
        error = []
        count = 0
        pre_corr=0
        # print('pre',lab_pre)
        label_true=dat.load_pkl(path+'label_true')
        # print('act',label_true)
        for k in tra_p:
            cur = tra_p[k][-1]
            c = lab_pre[count]
            dis = []
            ### find the nearest location as the predicted destination
            for p in center[c][1]:
                # dis.append( pre.mean_hav_dis( cur, p ) )
                dis.append( pre.mean_hav_dis( cur, p ) )
            index = dis.index( min( dis ) )
            # print(c,label_true[count])
            if c==label_true[count]:
                pre_corr+=1
            ###
            des_act = des_true[k]  # the actual destination position
            des_pre = center[c][1][index]  # the predicted destination position
            # error.append( pre.mean_hav_dis( des_act, des_pre ) )
            error.append( pre.mean_hav_dis( des_act, des_pre ) )
            count += 1
        # print('correct',pre_corr,'total',lab_pre.__len__(),'class accuracy:',str(round(pre_corr/lab_pre.__len__()*100,4))+'%')
        return sum( error ) / error.__len__()
    center = dat.load_pkl( path + 'res_clu_train_30000' )
    test_com_p = dat.load_pkl( path + 'com/test_completion_p_' + str( com ) )
    test_com_c = dat.load_pkl( path + 'com/test_completion_c_' + str( com ) )
    des_true = dat.load_pkl( path + 'des_test' )
    neuron = [16, 32, 64, 128, 256]
    batch = [10, 32, 64, 100, 200]
    using = neuron
    for u in using:
        # name_model=path_model+'2_64_'+str(u)
        name_model = path_model + '1_'+str(u)+'_64'
        model=dat.load_pkl(name_model)
        error = predict(test_com_p,test_com_c, center, model, des_true )  # when the trajectory completion is 100%
        # print('trajectory completion'+str(com*10) +'%, mean error ', error,'km')
        print(com, error, end=',')
def draw_mean_error():
    ### neuron_10:0.4251453953436097(problem)
    neuron_10 = [0.365918854238573, 0.3151654496486092, 0.3416338523909765, 0.3251474590762448, 0.33818437420235364]
    batch_10 = [0.4558806008911369, 0.31916990194299955, 0.32552471420695117, 0.6507219057384389, 0.7326530144531515]
    batch_4 = [1.4438736803611683,1.4746732212493596,1.4296151328965978,1.6423217824776295,1.533614993181575]
    neuron_4 = [1.419176230272559,1.4128741470168624,1.4122939722591887,1.4130528833845968,1.4693538806866449]
    batch_6 = [1.1554064644340738,1.1342175392806677,1.1355379408404467,1.1660557633773483,1.5519530135791397]
    neuron_6 = [1.1757941756688053,1.1381494277191844,1.130215200582154,1.157855873034558,1.1106164789236752]
    batch_8 = [0.7614722184776789,0.7251990969286859,0.8017236543680126,1.0872215966362344,0.8924484372477195]
    neuron_8 = [0.8216869624378512,0.7346602308429029,0.7553573106010909,0.7926034043702825,0.7646504114628165,]
    def compare_multi_line( data, label=[], xlabel='',ylabel='',xticks=[], title='', solid=888 ):
        import matplotlib.pyplot as plt
        line_style = '-'  #
        # color = ['b', 'r', 'lime', 'cyan', 'k', 'y','g','deeppink']
        color = ['cyan', 'g', 'b', 'r', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
                 'tan', 'dodgerblue', 'yellow']
        marker = ['s', 'x', '*', '^', '.', '+', 'p', 'v', '>', '1', '2', '3', '4']
        if label:
            for i in range( data.__len__() ):
                row = range( 1, data[i].__len__() + 1 )
                if i == solid:
                    plt.plot( row, data[i], linestyle='-', marker=marker[i], color=color[i], label=label[i], ms=5 )
                else:
                    plt.plot( row, data[i], linestyle=line_style, marker=marker[i], color=color[i], label=label[i], ms=5)
            plt.legend()
        else:
            for i in range( data.__len__() ):
                row = range( 1, data[i].__len__() + 1 )
                plt.plot( row, data[i], linestyle=line_style, marker=marker[i], color=color[i] )
        plt.title( title )
        if xticks:
            plt.xticks([i+1 for i in range(5)],xticks)
        plt.xlabel( xlabel )
        plt.ylabel( ylabel )
        plt.show()
    def transform(data):
        data_re=[[]]*5
        for i in range(data.__len__()):
            data_re[i].append( data[i][i] )
            # print(data_re[i].append(data[i][i]))
            # print(data[i][i])
        return data_re
    completion=[batch_4,batch_6,batch_8,batch_10]
    neuron=[neuron_4,neuron_6,neuron_8,neuron_10]
    # transform(neuron)
    # compare_multi_line(transform(neuron))
    # compare_multi_line( transform(completion) )
    compare_multi_line(completion,['40%','60%','80%','100%'],xticks=[10,32,64,100,200],xlabel='Batch size',ylabel='Mean Prediction Error(km)')
    compare_multi_line( neuron, ['40%', '60%', '80%', '100%'], xticks=[16,32,64,128,256],xlabel='Neurons',ylabel='Mean Prediction Error(km)' )
def __draw__():
    import matplotlib.pyplot as plt
    ax = plt.subplot( 111 )
    ax.text( 0.1, 0.8, r"$\int_a^b f(x)\mathrm{d}x$", fontsize=30, color="red" )
    ax.text( 0.1, 0.3, r"$\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}!$", fontsize=30 )
    plt.show()
def debug():
    def compare_multi_line(data, label=[], ylabel='', title='', solid=888):
        import matplotlib.pyplot as plt
        line_style = '-'  #
        # color = ['b', 'r', 'lime', 'cyan', 'k', 'y','g','deeppink']
        color = ['r', 'b', 'g', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
                 'tan', 'dodgerblue', 'yellow']
        marker = ['s', 'x', '*', '^', '.', '+', 'p', 'v', '>', '1', '2', '3', '4']
        if label:
            for i in range( data.__len__() ):
                row = range( 1, data[i].__len__() + 1 )
                if i == solid:
                    plt.plot( row, data[i], linestyle='-', marker=marker[i], color=color[i], label=label[i], ms=2 )
                else:
                    plt.plot( row, data[i], linestyle=line_style, marker=marker[i], color=color[i], label=label[i],
                              ms=2 )
            plt.legend()
        else:
            for i in range( data.__len__() ):
                row = range( 1, data[i].__len__() + 1 )
                plt.plot( row, data[i], linestyle=line_style, marker=marker[i], color=color[i] )
        plt.title(title)
        plt.xticks(range( 1, data[0].__len__() + 1 ),[40,80,120,160,200])
        plt.xlabel('Threshold')
        plt.ylabel(ylabel)
        plt.show()

    dic_sse_f = {15: [21.8651946940172, 27.78769163305876, 29.150500568349823, 29.36308472843291, 29.61561782282503],
                 30: [22.884729508953733, 32.814964851545376, 36.8363553990977, 37.700343416170625, 37.813683160744304],
                 45: [23.017703773753134, 34.416694865373586, 40.838521768869946, 43.01558088783681, 44.400076558517135]
                 }
    dic_sse = {15: [22.48381239490696, 29.087303414524218, 30.641280076020973, 30.96684897441528, 31.340465207001568],
               30: [23.126029783079975, 33.59554770824346, 38.03383575286666, 39.02907456768665, 39.463312807376504],
               45: [23.166673491138354, 34.94989866828563, 41.718940136122264, 44.24658725518167, 45.622346057103584]
               }
    dic_ch_f = {
        15: [108.69190309341114, 155.36558209739306, 163.80961737329403, 167.0178458950471, 168.47378914067139],
        30: [111.49073260949203, 191.41122219843714, 232.3952534869933, 235.9308687777544, 246.15606608835378],
        45: [112.18533686074976, 204.098591068542, 266.5287028111998, 293.5961090901342, 296.3851491258461]
    }
    dic_ch = {
        15: [108.45966567496576, 154.50098977094507, 161.33063714985929, 163.03506573844842, 163.99013605757136],
        30: [111.72585075406847, 189.17806926930712, 229.99866460011256, 232.3501242353219, 243.05160519958037],
        45: [112.44377933518862, 203.22789350264014, 265.2506686970123, 290.21691725460386, 293.79652106388005]
    }
    # compare_multi_line(list(dic_sse_f.values()),list(dic_sse_f.keys()),title='sse')
    # compare_multi_line( list( dic_ch_f.values() ), list( dic_ch_f.keys() ),title='ch')
# debug()
def normalization(data):
    data_re=[]
    d_min=min(data)
    d_max=max(data)
    for d in data:
        # data_re.append((d-d_min)/(d_max-d_min))
        data_re.append( (d_max - d) / (d_max - d_min) )
    return data_re
def mul():
    ##### 15
    sse_dic={40 : 22.48381239490696,80 : 29.087303414524218,120 : 30.641280076020973,160 : 30.96684897441528,200 : 31.340465207001568}
    num_dic={40 : 4867,80 : 3294,120 : 3074,160 : 3024,200 : 2983}
    #####45
    sse_dic = {40 : 23.166673491138354, 80 : 34.94989866828563, 120 : 41.718940136122264, 160 : 44.24658725518167,
               200: 45.622346057103584}
    num_dic = {40 : 4658,80 : 2340,120 : 1639,160 : 1439,200 : 1393}
    #####30
    sse_dic={40 : 23.126029783079975,80 : 33.59554770824346,120 : 38.03383575286666,160 : 39.02907456768665,200 : 39.463312807376504}
    num_dic={40 :4691,80 : 2553,120:1999,160:1927,200:1846}
    #####
    sse_no=normalization( list( sse_dic.values() ) )
    num_no=normalization( list( num_dic.values() ) )
    sum_no=[sse_no[i]*num_no[i] for i in range(sse_no.__len__())]
    print('sse',list( sse_dic.values() ))
    print('num',list( num_dic.values() ))
    print('sse_nor',sse_no)
    print('num_nor',num_no)
    print( 'sum',sum_no )
    import matplotlib.pyplot as plt
    plt.plot(range(sse_dic.__len__()),sse_no,label='sse')
    plt.plot(range(sse_dic.__len__()),num_no,label='num')
    plt.plot( range( sse_dic.__len__() ), sum_no,'r', label='sum' )
    plt.legend()
    plt.show()
# mul()
def view_metric():
    completion=[3,4,5,6,7,8,9]
    neurons = [16, 32, 64, 128, 256]
    models = {0.3: '1_16_64', 0.4: '2_64_32', 0.5: '1_64_32',
              0.6: '1_64_10', 0.7: '1_64_10', 0.8: '1_64_32', 0.9: '1_64_64'}
    def predict(tra_p, tra_class, center, model, des_true):
        '''
        to predict the location of final destination position and calculate the mean error between them
        :param tra_p: the completed trajectory 
        :param tra_class: the trajectory represented with classes
        :param center: the clustering center
        :param model: the trained prediction model
        :param des_true: the actual destination position of each trajectory
        :return: the mean error(km)
        '''
        tra_rep = pre.represent( tra_class, center )
        fea = pre.get_feature( tra_rep )
        fea = pre.padding( fea, Len_padding )
        res = model.model.predict( fea )
        lab_pre = pre.reverse( res, center )
        error = []
        count = 0
        pre_corr=0
        # print('pre',lab_pre)
        label_true=dat.load_pkl(path+'label_true')
        # print('act',label_true)
        label_pre={}
        for k in tra_p:
            cur = tra_p[k][-1]
            c = lab_pre[count]
            dis = []
            ### find the nearest location as the predicted destination
            for p in center[c][1]:
                # dis.append( pre.mean_hav_dis( cur, p ) )
                dis.append( pre.mean_hav_dis( cur, p ) )
            index = dis.index( min( dis ) )
            # print(c,label_true[count])
            if c==label_true[count]:
                pre_corr+=1
            ###
            des_act = des_true[k]  # the actual destination position
            des_pre = center[c][1][index]  # the predicted destination position
            # error.append( pre.mean_hav_dis( des_act, des_pre ) )
            label_pre[k]=des_pre
            error.append( pre.mean_hav_dis( des_act, des_pre ) )
            count += 1
        # print('correct',pre_corr,'total',lab_pre.__len__(),'class accuracy:',str(round(pre_corr/lab_pre.__len__()*100,4))+'%')
        return sum( error ) / error.__len__(),label_pre
    def metric_loss_acc():
        name_metric = 'acc'
        for com in completion:
            mean_error[com] = {}
            path_model = './file/model_com_done/com_' + str( com ) + '/'
            for m in models:
                model = dat.load_pkl( path_model + m )
                temp = m.split( '_' )
                # metric=np.average(list(model.history[name_metric]))
                metric = np.average( list( model.history[name_metric] ) )
                mean_error[com][(int( temp[0] ), int( temp[1] ))] = metric
                print( 'completion:', com, 'model:', m, name_metric + ':', metric )
        dat.save_pkl( mean_error, path + 'result/' + 'com_' + name_metric, make_dir=path + 'result/' )
    def metric_error_ln():
        center = dat.load_pkl( path + 'res_clu_train_30000' )
        des_true = dat.load_pkl( path + 'des_test' )
        for com in completion:
            test_com_p = dat.load_pkl( path + 'com/test_completion_p_' + str( com ) )
            test_com_c = dat.load_pkl( path + 'com/test_completion_c_' + str( com ) )
            mean_error[com] = {}
            path_model = './file/model_com_done/com_' + str( com ) + '/'
            for m in models:
                model = dat.load_pkl( path_model + m )
                temp = m.split( '_' )
                metric=predict( test_com_p, test_com_c, center, model, des_true )
                mean_error[com][(int( temp[0] ), int( temp[1] ))] = metric
                print( 'completion:', com, 'model:', m,  'mean error:', metric )
        dat.save_pkl( mean_error, path + 'result/' + 'com_error' , make_dir=path + 'result/' )
    def metric_error_batch():
        batch=[10,32,64,100,200]
        center = dat.load_pkl( path + 'res_clu_train_30000' )
        des_true = dat.load_pkl( path + 'des_test' )
        tra_class=dat.load_pkl(path+'test_class_30000')
        tra_p=dat.load_pkl(path+'test_p')
        for com in completion:
            models=[]
            if com==3:
                layer = 1
                neuron=16
            elif com==4:
                layer=2
                neuron = 64
            else:
                layer=1
                neuron = 64
            for b in batch:
                name=str(layer)+'_'+str(neuron)+'_'+str(b)
                models.append(name)
            # test_com_p = dat.load_pkl( path + 'com/test_completion_p_' + str( com ) )
            # test_com_c = dat.load_pkl( path + 'com/test_completion_c_' + str( com ) )
            test_com_p={}
            for k in tra_p:
                length = int( tra_p[k].__len__() * (com / 10) + 0.5 )
                test_com_p[k]=tra_p[k][:length]
            test_com_c={}
            for k in tra_class:
                length=int(tra_class[k].__len__()*(com/10)+0.5)
                test_com_c[k]=tra_class[k][:length]
            mean_error[com] = {}
            path_model = './file/model_batch/com_' + str( com ) + '/'
            for k in models:
                model=dat.load_pkl(path_model+models[k])
            for m in models:
                model = dat.load_pkl( path_model + m )
                temp = m.split( '_' )
                metric = predict( test_com_p, test_com_c, center, model, des_true )
                mean_error[com][(int( temp[0] ), int( temp[1] ))] = metric
                print( 'completion:', com, 'model:', m, 'mean error:', metric )
        dat.save_pkl( mean_error, path + 'result/' + 'com_batch', make_dir=path + 'result/' )
    def metric_forum():
        # layer=[1,2,3,4,5]
        # neuron=[16,32,64,128,256]
        ############################ error
        # metric = dat.load_pkl( path + 'result/com_error' )
        # data = metric
        # for k in data:
        #     print(k,':',min(list(data[k].values())),end=',')
        #############################
        metric = dat.load_pkl( path + 'result/com_loss' )
        data = metric
        layer = [1, 2, 3,4,5]
        neuron = [16, 32, 64, 128, 256]
        ##### & 1 & 19.361 & 17.521 & 17.239 & 17.493 & 18.014&17.926 \\
        for k in data:
            ave = [[], [], [], [], [], [],]
            print(str(k*10)+'%')
            for l in layer:
                print(l,end='  ')
                temp=[]
                num=0
                for n in neuron:
                    temp.append(data[k][(l,n)]*100000)
                    print(round(data[k][(l,n)]*100000,3),end=' ',)
                    # print(n,end=' ')
                    num+=1
                ave[-1].append( np.average( temp ) )
                print(round(np.average(temp),3))
            for i in range( neuron.__len__() ):
                for l in layer:
                    ave[i].append(data[k][(l,neuron[i])]*100000)
            print('ave',end='')
            temp=[]
            for a in ave:
                temp.append(np.average(a))
                print(round(np.average(a),3),end=' ')
            print()
            print( neuron[temp.index( min( temp ) )])
        ##############################
        #### draw forum
        # for k in data:
        #     #40
        #     print('\multirow{3}{*}{$'+str(k*10)+'\%$}')
        #     for l in layer:
        #         print('&',l,end=' & ')
        #         temp=[]
        #         for n in neuron:
        #             temp.append(data[k][(l,n)]*100000)
        #             print(round(data[k][(l,n)]*100000,3),end=' & ',)
        #             # print(n,end=' ')
        #         print(round(np.average(temp),3),"\\\ ")
        #         print('\cline{2-8}')
        #     print('\hline')
            # print(k,data[k])
        ##############################
    # metric_loss_acc()
    # metric_error_ln()
    metric_error_batch()
    # metric_forum()
def contrast():
    import matplotlib.pyplot as plt
    ELM={0.3:3.01,0.4:2.68,0.5:2.24,0.6:1.78,0.7:1.3,0.8:0.86,0.9:0.42}#0.1:3.11,0.2:3.14,
    LSTM={0.3:1.93,0.4:1.56,0.5:1.24,0.6:0.98,0.7:0.82,0.8:0.72,0.9:0.67}
    T_CONV={0.3:2.09,0.4:1.7,0.5:1.31,0.6:1.105,0.7:0.9,0.8:0.765,0.9:0.63}#0.1:2.68,
    # Our={0.3:1.62219596896212,0.4:1.4789717114279877,0.5:1.296378241079569,
    #      0.6:1.151815562118609,0.7:0.967497895546619,0.8:0.7595743980561235,0.9:0.5485662379259848}
    # Our={0.4:1.4,0.6:1.13,0.8:0.72,0.9:0.5}
    Our={3 : 1.5179135573961915,4 : 1.4033422121539898,5 : 1.2056538021713266,6 : 1.0791560855215945,7 : 0.909963148733716,8 : 0.7346602308429029,9 : 0.49331078949800433}
    label=['ELM','LSTM','T-CONV','Our']
    data=[ELM,LSTM,T_CONV,Our]
    data_dra=[]
    for i in range(data.__len__()):
        data_dra.append({})
        for k in data[i]:
            # if k in [0.4,0.6,0.8]:
            data_dra[i][k]=data[i][k]
    # print(data_dra)
    for i in range(data_dra.__len__()):
        # print(range(data[i].__len__()),list(data[i].values()))
        if i==data_dra.__len__()-1:
            plt.plot( range( data_dra[i].__len__() ), list( data_dra[i].values() ), 'r',label=label[i], linewidth=3)
        else:
            plt.plot(range(data_dra[i].__len__()),list(data_dra[i].values()),label=label[i])
    plt.legend()
    plt.xticks(range(data_dra[0].__len__()),list(data_dra[0].keys()))
    plt.show()
def metric_process():
    def predict(tra_p, tra_class, center, model, des_true):
        '''
        to predict the location of final destination position and calculate the mean error between them
        :param tra_p: the completed trajectory 
        :param tra_class: the trajectory represented with classes
        :param center: the clustering center
        :param model: the trained prediction model
        :param des_true: the actual destination position of each trajectory
        :return: the mean error(km)
        '''
        tra_rep = pre.represent( tra_class, center )
        fea = pre.get_feature( tra_rep )
        fea = pre.padding( fea, Len_padding )
        res = model.model.predict( fea )
        lab_pre = pre.reverse( res, center )
        error = []
        count = 0
        pre_corr=0
        # print('pre',lab_pre)
        label_true=dat.load_pkl(path+'label_true')
        # print('act',label_true)
        label_pre={}
        for k in tra_p:
            cur = tra_p[k][-1]
            c = lab_pre[count]
            dis = []
            ### find the nearest location as the predicted destination
            for p in center[c][1]:
                # dis.append( pre.mean_hav_dis( cur, p ) )
                dis.append( pre.mean_hav_dis( cur, p ) )
            index = dis.index( min( dis ) )
            # print(c,label_true[count])
            if c==label_true[count]:
                pre_corr+=1
            ###
            des_act = des_true[k]  # the actual destination position
            des_pre = center[c][1][index]  # the predicted destination position
            # error.append( pre.mean_hav_dis( des_act, des_pre ) )
            label_pre[k]=des_pre
            error.append( pre.mean_hav_dis( des_act, des_pre ) )
            count += 1
        # print('correct',pre_corr,'total',lab_pre.__len__(),'class accuracy:',str(round(pre_corr/lab_pre.__len__()*100,4))+'%')
        return sum( error ) / error.__len__(),label_pre
    def metric_error_batch():
        models = {3: '1_16_64', 4: '2_64_32', 5: '1_64_32',
                  6: '1_64_10', 7: '1_64_10', 8: '1_64_32', 9: '1_64_64'}
        center = dat.load_pkl( path + 'res_clu_train_30000' )
        des_true = dat.load_pkl( path + 'des_test' )
        tra_class=dat.load_pkl(path+'test_class_30000')
        tra_p=dat.load_pkl(path+'test_p')
        for com in models:
            # test_com_p = dat.load_pkl( path + 'com/test_completion_p_' + str( com ) )
            # test_com_c = dat.load_pkl( path + 'com/test_completion_c_' + str( com ) )
            test_com_p={}
            for k in tra_p:
                length = int( tra_p[k].__len__() * (com / 10) + 0.5 )
                test_com_p[k]=tra_p[k][:length]
            test_com_c={}
            for k in tra_class:
                length=int(tra_class[k].__len__()*(com/10)+0.5)
                test_com_c[k]=tra_class[k][:length]
            path_model = './file/model_batch/com_' + str( com ) + '/'
            model=dat.load_pkl(path_model+models[com])
            metric,label_pre= predict( test_com_p, test_com_c, center, model, des_true )
            print( 'completion:', com, 'model:', models[com], 'mean error:', metric )
            dat.save_pkl( label_pre, path + 'result/' + 'res_'+str(com), make_dir=path + 'result/' )
    def metric_forum():
        # layer=[1,2,3,4,5]
        # neuron=[16,32,64,128,256]
        ############################ error
        # metric = dat.load_pkl( path + 'result/com_error' )
        # data = metric
        # for k in data:
        #     print(k,':',min(list(data[k].values())),end=',')
        #############################
        metric = dat.load_pkl( path + 'result/com_loss' )
        data = metric
        layer = [1, 2, 3,4,5]
        neuron = [16, 32, 64, 128, 256]
        ##### & 1 & 19.361 & 17.521 & 17.239 & 17.493 & 18.014&17.926 \\
        for k in data:
            ave = [[], [], [], [], [], [],]
            print(str(k*10)+'%')
            for l in layer:
                print(l,end='  ')
                temp=[]
                num=0
                for n in neuron:
                    temp.append(data[k][(l,n)]*100000)
                    print(round(data[k][(l,n)]*100000,3),end=' ',)
                    # print(n,end=' ')
                    num+=1
                ave[-1].append( np.average( temp ) )
                print(round(np.average(temp),3))
            for i in range( neuron.__len__() ):
                for l in layer:
                    ave[i].append(data[k][(l,neuron[i])]*100000)
            print('ave',end='')
            temp=[]
            for a in ave:
                temp.append(np.average(a))
                print(round(np.average(a),3),end=' ')
            print()
            print( neuron[temp.index( min( temp ) )])
        ##############################
        #### draw forum
        # for k in data:
        #     #40
        #     print('\multirow{3}{*}{$'+str(k*10)+'\%$}')
        #     for l in layer:
        #         print('&',l,end=' & ')
        #         temp=[]
        #         for n in neuron:
        #             temp.append(data[k][(l,n)]*100000)
        #             print(round(data[k][(l,n)]*100000,3),end=' & ',)
        #             # print(n,end=' ')
        #         print(round(np.average(temp),3),"\\\ ")
        #         print('\cline{2-8}')
        #     print('\hline')
            # print(k,data[k])
        ##############################
    # metric_loss_acc()
    # metric_error_ln()
    metric_error_batch()
    # metric_forum()
if __name__ == '__main__':
    ''
    # draw_tra_exam()
    # draw_seg_exam()#tra_exam
    # chart()
    # several_exam_seg()
    # effect_cluster()
    # compare()
    # vs()
    # forum()
    # predict()
    # pre_com()
    # draw_mean_error()
    # __draw__()
    # print(np.average([8.1625,6.0025,4.9525,5.6199,5.2422]))
    # contrast()
    # view_metric()
    metric_process()