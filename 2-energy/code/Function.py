# -*- coding: utf-8 -*-
"""
Created on Thursday,3 October 10:47 2019

Modified on 3 October - 

@author:Jeaten
"""
import os,pickle,json
import numpy as np
File_path='./File/'
Model_path='./File/Model/' #the path of you save trained Lstm models
Tolerate = 5 #the parameter of early stop (if the loss increases after Tolerate training times, then stop training)
Mem_Sent={} #It is used to record which vehicles have been sent messages
Trajectory={}
Sta_sent={}
Query={}
rate_energy=0
class DataModule:
    '''
    this defined class is used to perform various data operations we need 
    '''
    def __init__(self):
        self.delta = 15  ###interval of departing dict's key by time###
        self.day = 1440  # every day has 1440 minutes in minutes#
        self.interval = 96  # the interval divide a day with 15 minutes
        self.longitudeUp = 51  # the up bound of longitude
        self.longitudeLower = 0  # the lower bound of longitude
        self.latitudeUp = 51  # the up bound of latitude
        self.latitudeLower = 0  # the lower bound of latitude
    def load_json(self, dict_path):
        '''
        load dict file, dict's whole path is given is ok
        :param dict_path: dict's path
        :return: the dict loaded
        '''
        dict_file = open( dict_path, 'r' )
        js = dict_file.read()
        dict = json.loads( js )
        dict_file.close()
        return dict
    def load_pkl(self, pkl_path):
        '''
        load pkl file like json above
        :param pkl_path: pkl dict's whole path
        :return: the loaded pkl dict
        '''
        with open( pkl_path, 'rb' )as file_point:
            return pickle.load( file_point )
    def save_pkl(self, dict, save_path):
        '''
        save the pickle dict to a fixed path
        :param Dict: the pickle dict we want to save
        :param SavePath: the pkl dict's saving path
        :return: whether the function is executed successfully
        '''
        with open( save_path, 'wb' )as file_point:
            pickle.dump( dict, file_point, pickle.HIGHEST_PROTOCOL )
        file_point.close()
        return True
    def load_list(self,path):
        '''
        load the saved list from file
        :param path: the path of file
        :return: the list format data
        '''
        array=np.load(path,encoding="latin1")
        return array.tolist()
    def save_array(self, array, path):
        '''
        save array as file
        :param array: the array you want to save
        :param path: the save path
        :return: True if it is executed successfully
        '''
        np.save( path, array )
        return True
    def trans_pkl_python2(self,data,path):
        import pickle
        pickle.dump(data,open(path,"wb"), protocol=2)
    def average(self,list):
        '''
        this function is used to calculate the average of a single or multiple list([] or [[],[],...,[]])
        :param list: the list we want to calculate
        :return: the average value(if list is single, return an average value; multiple, an average list)
        '''
        try:
            if len(list[0])>0:
                stack=[0]*len(list[0])
                for l in list:
                    for j in range(len(l)):
                        stack[j]+=l[j]
                for j in range(len(stack)):
                    stack[j]/=len(list)
                return stack
        except:
            ave = 0
            for i in list:
                ave += i
            return ave / len( list )
    def median(self,list):
        '''
         this function is used to calculate the median of a single or multiple list([] or [[],[],...,[]])
        :param list: the list we want to calculate
        :return: the median(if list is single, return a median value; multiple, a median list)
        '''
        try:
            if len(list[0])>0:
                return list[int(len(list)/2)]
        except:
            return list[int(len(list)/2)]
    def eucdistance(self, listA, listB):
        '''
        this function is used to calculate the Euclidean distance of two lists.
        :param listA: list A
        :param listB: list B
        :return: the Euclidean distance of the list A and list B
        '''
        dis = 0
        assert len( listA ) == len( listB )
        for i in range( len( listA ) ):
            m = abs( listA[i] - listB[i] )
            # print(m*m)
            dis += m * m
        # print(dis)
        return dis
    def standardize(self,data):
        '''
        this function is used to generate the data for train, predict or correspond
        :param data: the original data
        :return: the data we can use
        '''
        Grid_Veh = {}
        #transform original trajectory to {day:{grid:{interval:[vehicles]}}} format
        '''
        9 (12, 37) {36: [1334]}
        9 (12, 38) {36: [1334]}
        9 (12, 39) {36: [1334], 10: [1877], 40: [6059], 42: [6059], 37: [7270], 35: [8201]}
        9 (13, 40) {36: [1334], 40: [6059], 41: [6059], 42: [6059], 43: [7238]}
        9 (13, 41) {36: [1334]}
        '''
        for day in data:
            Grid_Veh[day]={}
            for veh in data[day]:
                for key in data[day][veh]:
                    for k in data[day][veh][key]:
                        if k[1][0] in range( self.longitudeLower, self.longitudeUp ) and k[1][1] in range(self.latitudeLower, self.latitudeUp ):
                            try:
                                if Grid_Veh[day][k[1]]:
                                    try:
                                        Grid_Veh[day][k[1]][k[0]].append(veh)
                                    except:
                                        Grid_Veh[day][k[1]][k[0]]=[]
                                        Grid_Veh[day][k[1]][k[0]].append( veh )
                            except:
                                Grid_Veh[day][k[1]]={}
                                try:
                                    Grid_Veh[day][k[1]][k[0]].append( veh )
                                except:
                                    Grid_Veh[day][k[1]][k[0]] = []
                                    Grid_Veh[day][k[1]][k[0]].append( veh )
        '''
        transform to following format:
        9 :{(36,(13, 41)):[1334]}
        '''
        Tu_Veh={}
        for day in Grid_Veh:
            Tu_Veh[day]={}
            for grid in Grid_Veh[day]:
                for i in Grid_Veh[day][grid]:
                    Tu_Veh[day][tuple([i,grid])]=list(set(Grid_Veh[day][grid][i]))
        # DataModule().save_pkl(Grid_Veh,'./File/Grid_Veh.pkl')
        DataModule().save_pkl( Tu_Veh, './File/Tu_Veh.pkl' )
        '''
        transform to format above to the format can be used to train
        8 (6, 39) [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,...]
        '''
        Act={}
        for day in Grid_Veh:
            Act[day]={}
            for grid in Grid_Veh[day]:
                Act[day][grid]=[]
                for interval in range(1,97):
                    try:
                        Act[day][grid].append(len(list(set(Grid_Veh[day][grid][interval]))))
                    except:
                        Act[day][grid].append(0)
        DataModule().save_pkl(Act,'./File/Original.pkl')
        return Act
    def distribution(self,data,test_day=8):
        train={}
        test={test_day:{}}
        for day in data:
            if day!=test_day:
                for grid in data[day]:
                    try:
                        train[grid].append(data[day][grid])
                    except:
                        train[grid]=[]
                        train[grid].append(data[day][grid])
            else:
                for grid in data[day]:
                    test[test_day][grid]=data[day][grid]
        # DataModule().save_pkl( train,File_path + 'train_stand.pkl' )
        DataModule().save_pkl( test, File_path + 'test_set.pkl' )
        for grid in train:
            ############ rounding-off method ######################
            train[grid]=[int(i+0.5) for i in self.average(train[grid])]
        return train,test
    def normalize(self,data):
        Norm={}
        for grid in data:
            temp=[]
            for i in data[grid]:
                try:
                    temp.append(i/max(data[grid]))
                except:
                    temp.append(0)
            Norm[grid]=temp
        return Norm
    def generate_train(self,data):
        import random,numpy
        train_x=[]
        train_y=[]
        for grid in data:
            train_x.append(data[grid])
            train_y.append(random.randint(0,9))
        return numpy.array(train_x),numpy.array(train_y)
class Draw:
    '''
    this class will be helpful if we want to draw some figure
    '''
    def draw(self,data):
        '''
        this function is used to draw the figure of a single list([1,2,3])
        :param data: the list we want to draw
        :return: return True if this function is be executed successfully
        '''
        import matplotlib.pyplot as plt
        # coordinate_x = range( 7, len( data ) + 7)#for elbow method#this is for [5:]
        coordinate_x = range( 2, len( data ) + 2 )#for elbow method#this is for [:]
        # coordinate_x=range(1,len(data)+1)#general draw
        # print(coordinate_x)
        # for i in coordinate_x:
        #     print(i)
        # plt.xlabel('K')#for elbow method
        # plt.ylabel('SSE')#for elbow method
        plt.plot(coordinate_x,data,'-'+'b'+'.')
        plt.show()
        return True
    def multi_draw(self,data,title=''):
        '''
        this function is used to draw the figure of a multiple list([[],[],...,[]])
        :param data: the multi list we want to draw the figure
        :param title: the figure's title we have draw
        :return: return True if this function is be executed successfully
        '''
        import matplotlib.pyplot as plt
        coordinate_x=range(1,len(data[0])+1)
        # label=[]
        color=['r','g','b','cyan','k','deeppink','lime','orange','slategrey','indigo','deepskyblue','gold','tan','dodgerblue','yellow']
        for i in range(len(data)):
            plt.plot(coordinate_x,data[i],color=color[i%len(color)],label=str(i+1))
        # plt.xlabel('Interval')
        # plt.ylabel('The number of normalized vehicle nodes')
        plt.title(title)
        plt.legend()
        plt.show()
        return True
    def draw_pre(self,original,prediction,title=''):
        assert len(original)==len(prediction)
        import matplotlib.pyplot as plt
        coordinate_x = range( 1, len( original[0]) + 1 )
        # label=[]
        color = ['r', 'g', 'b', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold','tan', 'dodgerblue', 'yellow']
        for i in range( len( original ) ):
            plt.plot( coordinate_x, original[i], color=color[i % len( color )], label='Original '+str( i + 1 ) )
            plt.plot( coordinate_x, prediction[i], color=color[i+4], label='Predicted ' + str( i + 1 ) )
        plt.xlabel('Interval')
        plt.ylabel('The number of vehicle nodes')
        plt.title( title )
        plt.legend()
        plt.show()
        return True
    def draw_heat(self,data):
        '''
        this function is used to draw the heatmap
        :param data: formatted 2D list
        :return: None
        '''
        from matplotlib import pyplot as plt
        from matplotlib import cm
        from matplotlib import axes
        cmap = 'gist_yarg'
        figure = plt.figure( facecolor='w' )
        ax = figure.add_subplot( 1, 1, 1 )
        vmax = 0
        vmin = 0
        for i in data:
            for j in i:
                if j > vmax:
                    vmax = j
                if j < vmin:
                    vmin = j
        map = ax.imshow( data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower' )
        plt.colorbar( mappable=map, cax=None, ax=None, shrink=0.9 )
        plt.show()
        '''
        this function can be called by this:
        data=DataModule().load_pkl('F:\Python\Project\Fitting\File/Heat_map.pkl')
        list=[]
        for i in range(0,51):
            temp=[]
            for j in range(0,51):
                temp.append(0)
            list.append(temp)
        for k in data:
            list[k[1]][k[0]]=data[k]#because the data is the other way around, we must reverse it to draw
        Draw().draw_heat(list)
        '''
    def draw_histogram_2(Data, Label=['5', '10', '15']):
        import matplotlib.pyplot as plt
        Num = 2
        Width = 0.4
        Coordinate = [[] for i in range( Num )]
        Height = Data
        Color = ['r', 'g', 'b', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue',
                 'gold',
                 'tan', 'dodgerblue', 'yellow']
        for k in range( 1, len( Data[0] ) + 1, 1 ):
            Coordinate[0].append( k - Width / 2 )
            # Coordinate[1].append( k )
            Coordinate[1].append( k + Width / 2 )
        for i in range( len( Data ) ):
            plt.bar( Coordinate[i], Height[i], color=Color[i], label=Label[i], width=Width )
        plt.legend()
        plt.xlabel( 'Cluster number' )
        plt.ylabel( 'Loss' )
        plt.show()
    def draw_histogram(self,Data,Label=['5', '10', '15']):
        import matplotlib.pyplot as plt
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        Num = 3
        Width = 0.25
        Coordinate = [[] for i in range( Num )]
        Height =Data
        Color = ['r', 'g', 'b', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
                 'tan', 'dodgerblue', 'yellow']
        for k in range( 1,len(Data[0])+1, 1 ):
            Coordinate[0].append(k-Width)
            Coordinate[1].append(k)
            Coordinate[2].append(k+Width)
        for i in range(len(Data)):
            plt.bar(Coordinate[i], Height[i], color=Color[i], label=Label[i], width=Width)
        plt.legend()
        plt.xlabel('类')#Cluster number
        plt.ylabel('误差')#Loss
        plt.show()
class Lstm:
    def __init__(self):
        self.step=1 #this parameter is used to confirm the step of we want to predict in using lstm
    def generate_data(self,data,window):
        '''
        this function is used to generate the training data of LSTM
        :param data: the original data
        :return: the data can be used to train lstm
        '''
        assert len(data[0])>window and window>0 and self.step>0 and self.step<=window
        train_x=[]
        train_y=[]
        for day in data:
            for s in range(len(day)):
                train_x.append([day[s-c] for c in range(window,0,-1)])
                train_y.append([day[(s+c)%len(day)] for c in range(self.step)])
        return train_x,train_y
    def train(self,feature,label,Classname):
        '''
        this function is used to train the lstm model for window data in every cluster
        :param feature: the feature in every cluster
        :param label: the label of every feature in a cluster
        :param Classname: the specified cluster class
        :return: True 
        '''
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation
        from keras.callbacks import EarlyStopping
        from keras.layers.recurrent import LSTM
        train_x=feature
        train_y=label
        interval=np.array(train_x)
        data=np.array(train_y)
        x_train = np.reshape( interval, (interval.shape[0],interval.shape[1], 1) )
        model = Sequential()
        model.add( LSTM(96, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True ) )
        model.add( LSTM(192, return_sequences=True ) )
        model.add( LSTM(384, return_sequences=True ) )
        model.add( LSTM(192, return_sequences=True ) )
        model.add( LSTM(96, return_sequences=False ) )
        model.add( Dense(1))
        model.add( Activation( 'linear' ) )
        model.compile( loss="mse", optimizer="rmsprop" )
        early_stopping = EarlyStopping( monitor='loss', patience=Tolerate, verbose=1 )
        BATCH_SIZE =96
        epoch = 6000
        model.fit( x_train, data, batch_size=BATCH_SIZE, verbose=1, epochs=epoch,validation_split=0,callbacks=[early_stopping])
        # res = model.predict(x_train)
        try:
            model.save(Model_path+'Fit_'+str(Classname)+'.m')
        except:
            os.makedirs(Model_path)
            model.save( Model_path+'Fit_'+ str( Classname ) + '.m' )
        # res=[int(i[0]) for i in res]
        # return res
        return True
    def load(self,Classname,type=''):
        '''
        this function is used to load the lstm model we have trained
        :param type: the model type we want to distinct it fromm others
        :param Classname: the distinctive filename
        :return: the model we have loaded
        '''
        from keras.models import load_model
        return load_model(Model_path+'Fit_'+type+str(Classname)+'.m')
    def predict(self,data,model,query,window):
        '''
        this function is used to predict every gird's node number distribution
        :param data: the original distribution
        :param model: the trained lstm model
        :param query: the dictionary each grid belongs which class
        :param window: the size of we want to use for predicting
        :return: the predicted result
        '''
        Original = {}
        Prediction = {}
        for day in data:
            Original[day] = {}
            Prediction[day] = {}
            for grid in data[day]:
                # print(day,grid,data[day][grid])
                try:
                    if grid in query:
                        # print(query[grid],data[day][grid])
                        # act, pre = Lstm().predict( data[day][grid], model[query[grid]],window)
                        feature=[]
                        label=[]
                        for s in range( len(data[day][grid]) ):
                            feature.append( [data[day][grid][s - c] for c in range( window, 0, -1 )] )
                            label.append( data[day][grid][s] )
                        x = np.array( feature )
                        x = np.reshape( x, (x.shape[0], x.shape[1], 1) )
                        pre = model[query[grid]].predict( x )
                        pre = [int( i[0] ) for i in pre]
                        Original[day][grid] = label
                        Prediction[day][grid] = pre
                except Exception as E:
                        print("Can't predict:", E )
        return Original,Prediction
    def predict_list(self,data,model,Window_pre):#for list
        window=Window_pre
        # for k in data:
        x=[]
        y=[]
        for s in range(len(data)):
            x.append([data[s - c] for c in range( window, 0, -1 )] )
            y.append(data[s])
        # print(data[k])
        # print(x)
        # print(y)
        x=np.array(x)
        x=np.reshape(x,(x.shape[0],x.shape[1], 1) )
        res=model.predict(x)
        res= [int( i[0] ) for i in res]
        return y,res
class Process:
    def class_reverse(self, data):
        '''
        see which elements are in each class
        :param data: the classified labels
        :return: the class dict
        '''
        '''
        data format:
        data:[1,4,7,7,0,...], where every number is the class of every data
        Dict:{1:[1,2,5,6],2:[0,4,8],...}, where key is the class and list is their detailed elements' index
        '''
        Dict = {}
        for i in range( len( data ) ):
            try:
                Dict[data[i]].append( i )
            except:
                Dict[data[i]] = []
                Dict[data[i]].append( i )
        return Dict
    def SSE(self, Kdict, hidden, center):
        '''
        this function is used to calculate the sum of the squared errors(SSE)
        :param Kdict: the elements of in each class
        :param hidden: the latent feature of the original distribution
        :param center: the gotten center using Kmeans
        :return: the error of SSE
        '''
        assert len( Kdict ) == len( center )
        SSE = 0
        for k in sorted( Kdict ):
            sse = 0
            for i in Kdict[k]:
                # print(self.Eucdistance(hidden[i], center[k] ))
                sse += DataModule().eucdistance( hidden[i], center[k] )
                # print(sse)
            SSE += sse
            # print(SSE)
        return SSE
    def elbow(self,path,data):
        Kdict =DataModule().load_list( path + data[0])
        Kdict =self.class_reverse( Kdict )
        hidden =DataModule().load_list( path + data[1])
        center =DataModule().load_list( path + data[2])
        SSE = self.SSE( Kdict, hidden, center )
        # print(SSE)
        return SSE
    def sum_loss(self,data,cat,model,Window_pre):
        '''
        this function is used to calculate the loss of every class
        :param data: vehicle nodes' distribution
        :param cat: the dictionary of each grid belongs which class
        :param model: the node prediction model
        :param Window_pre: the window' length of you want(this is base on our model,15 is a general value)
        :return: the loss dictionary of every class
        '''
        dis={}
        try:
            for k in cat:
                # for i in data[k]:
                act,pre=Lstm().predict(data[k],model[cat[k]],Window_pre)
                try:
                    dis[cat[k]]+=DataModule().eucdistance(act,pre)
                except:
                    dis[cat[k]]=0
                    dis[cat[k]] += DataModule().eucdistance( act, pre )
        except Exception as e:
            print(e)
        # print('count',count)
        return dis
    def prediction(self,Day,data,Query,Window):
        '''
        this function is used to predict every gird's node number distribution
        :param Day: the day we want to test
        :param data: the original distribution
        :param Query: the dictionary of each grid belongs which class
        :return: the actual distribution and predicted distribution
        '''
        model = {}
        for m in range( 0, 13 ):
            model[m] = Lstm().load(m)
        Original = {}
        Prediction = {}
        for day in Day:
            Original[day] = {}
            Prediction[day] = {}
            for grid in data[day]:
                if grid in Query:
                    try:
                        # print(Query[grid],data[day][grid])
                        act, pre = Lstm().predict( data[day][grid], model[Query[grid]],Window)
                        Original[day][grid] = act
                        Prediction[day][grid] = pre
                    except Exception as E:
                        print("Can't predict:", E )
        return Original,Prediction
    def catalogue(self,fin_id,stand):
        res =self.class_reverse( fin_id )
        # DataModule().save_pkl(res,'./File/Index_draw.pkl')
        mem = []
        for grid in stand:
            mem.append( grid )
            if len( mem ) >= len( fin_id ):
                break
        Query = {}
        for c in res:
            for e in res[c]:
                Query[mem[e]] = c
        return Query
    def generate_fit(self,Stand,Query,window):
        class_stand={}
        for grid in Query:
            for d in Stand[grid]:
                try:
                    class_stand[Query[grid]].append(d)
                except:
                    class_stand[Query[grid]]=[]
                    class_stand[Query[grid]].append( d )
        DataModule().save_pkl(class_stand,path+'class_stand.pkl')
        Fit = {}
        cat_stand = class_stand
        for k in sorted( cat_stand ):
            Fit[k] = {'feature': [],'label':[]}
            feature, label = Lstm().generate_data( cat_stand[k],window)
            assert len( feature ) == len( label )
            for i in range( len( feature ) ):
                Fit[k]['feature'].append( feature[i])
                Fit[k]['label'].append( label[i] )
        return Fit
class Correspondence:
    def static(self,key,value):
        day=8
        for v in value:
            # print(Trajectory[v])
            for Element in Trajectory[day][v]:
                if key in Trajectory[day][v][Element]:
                    # Sta_sent_clu[Query[key[1]]].append( tuple( [v, Element] ) )
                    try:
                        # if tuple([v,Element]) not in Sta_sent_clu[Query[key[1]]]:
                        #     Sta_sent_clu[Query[key[1]]].append(tuple([v,Element]))
                        Sta_sent[tuple([v,Element])].append(key)
                        # Sta_sent_clu[Query[key[1]]].append( tuple( [v, Element] ) )
                    except:
                        Sta_sent[tuple( [v, Element] )]=[]
                        Sta_sent[tuple( [v, Element] )].append( key )
                        # Sta_sent_clu[Query[key[1]]].append( tuple( [v, Element] ) )
                        # Sta_sent_clu[Query[key[1]]].append( key )
                    # print(Element)
                    # Sta_sent.append(tuple([v,Element]))
    def priority(self,tra,place):
        '''
        this function is used to determine the priority of a place in a trajectory
        :param tra: the trajectory
        :param place: the specified place of a trajectory
        :return: priority of a place in a trajectory
        '''
        return (1+(tra.index(place)+1))/(1+len(tra)-(tra.index(place)+1))
    def rank(self,day,key,value):
        '''
        this function is used to determine to send messages to which vehicles
        :param day: which day we want to test
        :param key: (interval,(gird)):the key of a trajectory
        :param value: vehicles that appear in a grid
        :return: vehicles are ordered according to their priorities
        '''
        Rank={}
        for v in value:
            # print(Trajectory[v])
            for Element in Trajectory[day][v]:
                if key in Trajectory[day][v][Element]:
                    Rank[v]=self.priority(Trajectory[day][v][Element],key)
                    # print(day,Trajectory[v][Element])
        Order=sorted(Rank,key=Rank.__getitem__,reverse=True)
        # print(value)
        # print(Order)
        return Order
    def allocation(self,Ori,Pre):
        Can_send={}
        for d in Ori:
            Can_send[d] = {}
            for g in Ori[d]:
                Can_send[d][g]=[]
                for i in range(len(Ori[d][g])):
                    if Pre[d][g][i]<=int(Ori[d][g][i]*rate_energy):
                        Can_send[d][g].append(Pre[d][g][i])
                    else:
                        Can_send[d][g].append(int(Ori[d][g][i]*rate_energy))
        return Can_send
    def rank_backup(self,day,key,value):
        '''
        this function is used to determine to send messages to which vehicles
        :param day: which day we want to test
        :param key: (interval,(gird)):the key of a trajectory
        :param value: vehicles that appear in a grid
        :return: vehicles are ordered according to their priorities
        '''
        Rank={}
        for v in value:
            # print(Trajectory[v])
            for Element in Trajectory[v]:
                if key in Trajectory[v][Element]:
                    Rank[v]=self.priority(Trajectory[v][Element],key)
                    # print(day,Trajectory[v][Element])
        Order=sorted(Rank,key=Rank.__getitem__,reverse=True)
        return Order
    def send_message(self,Key,Can_send):
        '''
        this function is used to send message to specific vehicles
        :param Key: (interval,(grid)):the rsu of gird can send messages in interval
        :param Can_send: vehicles in gird that rus can send messages to them
        :return: none
        '''
        for veh in Can_send:
            try:
                Mem_Sent[veh].append(Key)
            except:
                Mem_Sent[veh]=[]
                Mem_Sent[veh].append(Key )
    def direct(self,Original,Tra):
        index = 0
        Day = 8
        energy_sta=0
        for k in Original[Day]:
            energy = int( sum( Original[Day][k] ) * rate_energy)
            energy_sta+=energy
            energy_slot =int(energy/96)
            # print(energy,energy_slot)
            for interval in range( 1, len( Original[Day][k] ) ):
                try:
                    if Original[Day][k][interval]<=energy_slot:
                        index+=Tra[Day][(interval, k)].__len__()
                        # print((interval, k), Tra[Day][(interval, k)])
                        self.send_message( (interval, k), Tra[Day][(interval, k)] )
                        self.static( (interval, k), Tra[Day][(interval, k)] )
                    else:
                        index+=energy_slot
                        self.send_message( (interval, k), Tra[Day][(interval, k)][:energy_slot] )
                        self.static( (interval, k), Tra[Day][(interval, k)][:energy_slot])
                except Exception as E:
                    continue
                    # print("can't predict:",E)
        print( 'sent messages:', index, 'sent vehicles:', len( Mem_Sent ),'total energy:',energy_sta)
    def direct__(self,Original,Tra):
        index=0
        Day=8
        for k in Original[Day]:
            count=0
            energy=int(sum(Original[Day][k])*0.75)
            for interval in range(1,len(Original[Day][k])):
                try:
                    ######### key vehicles
                    count+=Original[Day][k][interval]
                    if count>energy:
                        break
                    else:
                        if index<598768:
                            self.send_message((interval, k),Tra[Day][(interval, k)])
                            self.static((interval, k),Tra[Day][(interval, k)])
                            # print(Original[Day][k][interval])
                            index += len(Tra[Day][(interval,k)])
                        else:
                            break
                except Exception as E:
                    continue
                    # print("can't predict:",E)
        print('sent messages:',index,'sent vehicles:',len(Mem_Sent))
    def strategy(self,Original,Prediction,Day,Tra):
        '''
        this function is used to determine the sending strategy
        :param Original: what vehicles in a grid in all day
        :param Prediction: rsu predicted the number of vehicle nodes all day
        :param Day: the specific day we want to test
        :param Tra: the trajectory of all vehicles in all grid
        :return: none
        '''
        # index = {}
        # for k in range( 0, 13 ):
        #     index[k] = 0
        index=0
        for k in Original[Day]:
            # print(k,Original[Day][k])
            # print(k,Prediction[Day][k])
            # print('=='*96)
            for interval in range(1,len(Original[Day][k])):
                if Original[Day][k][interval-1]<=Prediction[Day][k][interval-1] and Prediction[Day][k][interval-1]!=0:
                    # print((interval,k),Original[Day][k][interval-1],Prediction[Day][k][interval-1])
                    '''Send directly'''
                    try:
                        # Can_send = {(interval, k): Tra[Day][(interval, k)]}
                        # print(Can_send,Tra[Day][(interval,k)])
                        self.send_message((interval, k),Tra[Day][(interval, k)])
                        self.static( (interval, k), Tra[Day][(interval, k)] )
                        index+=len(Tra[Day][(interval, k)])
                    except Exception as E:
                        pass
                        print("Can't predict",E)
                elif Original[Day][k][interval-1]>Prediction[Day][k][interval-1] and Original[Day][k][interval-1] !=0 and Prediction[Day][k][interval-1] !=0:
                    '''Sort by priority before sending'''
                    try:
                        # print(Day,k, interval, Original[Day][k][interval - 1], Prediction[Day][k][interval - 1],Tra[Day][(interval, k)])
                        Can_send = self.rank( Day, (interval, k), Tra[Day][(interval, k)] )
                        # if Prediction[Day][k][interval - 1]>0:
                            # print( len( Can_send ), Prediction[Day][k][interval - 1] )
                            # print(Can_send[:Prediction[Day][k][interval - 1]])
                            # print(Can_send)
                            # print(Tra[Day][(interval, k)])
                            # print('==='*96)
                        self.send_message((interval, k),Can_send[:Prediction[Day][k][interval-1]])
                        self.static( (interval, k), Can_send[:Prediction[Day][k][interval-1]])
                            # index +=Prediction[Day][k][interval-1]
                        index+=len(Can_send[:Prediction[Day][k][interval - 1]])
                    except Exception as E:
                        print('Error:',E)
        print('Sent:',index)
        print(len(Mem_Sent),len(Trajectory[Day]),'Coverage rate:',len(Mem_Sent)/len(Trajectory[Day]))
# path=File_path
# Original=DataModule().load_pkl(path+'Original_res.pkl')
# Prediction=DataModule().load_pkl(path+'Prediction_res.pkl')
# stand=DataModule().load_pkl(path+'Tu_Veh.pkl')
# Correspondence().direct(Original,stand)
# d=Correspondence().allocation(Original,Prediction)
# Correspondence().strategy(Original,Prediction,8,d)

# Correspondence().strategy(Original,Prediction,1,d)
class Used:
    def train(self):
        import time
        s = time.time()
        print( 'starting' )
        Filepath='./'
        data = DataModule().load_pkl( Filepath+'Fin_noave.pkl' )
        # print(len(data))
        # train simple one or multi
        K=[12,11,10]
        for k in K:
            #     # print(k,data[k])
            Lstm().train( data[k]['feature'], data[k]['label'], k )
        '''
        # train overall
        for k in data:
            #     # print(k,data[k])
            F.Lstm().train( data[k]['feature'], data[k]['label'], k )
        '''

        print( 'ending' )
        e = time.time()
        print( 'time is'+ str((e - s) / 60)+'mins')
    def cal_loss(self):
        # Dict={}
        # for k in Query:
        #     try:
        #         Dict[Query[k]].append(Test[k])
        #     except:
        #         Dict[Query[k]]=[]
        #         Dict[Query[k]].append( Test[k] )
        #     # print(Query[k],Test[k])
        path = '/home/Jeaten/ICC/DCN/Dynamic/File/'
        # Model_path = '/home/Jeaten/ICC/DCN/Com/Fin_noave_5/Model/'
        Model_path = '/home/Jeaten/ICC/DCN/Dynamic/File/Model_15/'
        Model_path = Model_path
        Query = DataModule().load_pkl( path + 'Query_ave.pkl' )
        Test = DataModule().load_pkl( path + 'Test_set.pkl' )
        Model = {}
        for k in range( 0, 13 ):
            Model[k] = Lstm().load( '', k )
        loss = Process().sum_loss( Test, Query, Model, 15 )  # int(Model_path[-2])
        print(loss)
        '''
        order=[4, 8, 6, 12, 5, 11, 2, 1, 3, 9, 10, 7, 0]
        loss={0: 5113, 7: 2204, 11: 6331, 2: 7439, 6: 3144, 10: 320636, 8: 12498, 3: 17033, 1: 555078, 4: 2908, 9: 61938, 12: 6815, 5: 58639}
        '''
class Result:
    def elbow(self,data):
        import matplotlib.pyplot as plt
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        coordinate_x = range( 7, len( data ) + 7)#for elbow method#this is for [5:]
        # coordinate_x = range( 2, len( data ) + 2 )  # for elbow method#this is for [:]
        # coordinate_x=range(1,len(data)+1)#general draw
        # print(coordinate_x)
        # for i in coordinate_x:
        #     print(i)
        plt.xlabel('K')#for elbow method
        plt.ylabel('误差平方和')#for elbow method 'SSE'
        plt.plot( coordinate_x, data, '-' + 'b' + '.' )
        plt.show()
        return True
    def multi(self,data,title=''):
        import matplotlib.pyplot as plt
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        coordinate_x = range( 1, len( data[0] ) + 1 )
        # label=[]
        color = ['r', 'g', 'b', 'cyan', 'k', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue', 'gold',
                 'tan', 'dodgerblue', 'yellow']
        s='区域'#Grid
        for i in range( len( data ) ):
            plt.plot( coordinate_x, data[i], color=color[i % len( color )], label=s+str( i + 1 ) )
        plt.xlabel("时间段") # 'Interval'
        plt.ylabel("标准化的车辆节点数") #'The number of normalized vehicle nodes'
        plt.title( title )
        plt.legend()
        plt.show()
    def draw_elbow(self):
        # '''
        # draw the figure Elbow and Elbow_Ori
        SSE = DataModule().load_list( path + 'result_2-30.npy' )
        # Draw().draw( SSE[:] )  # Elbow
        self.elbow(SSE[5:])
        # '''
    def draw_cluster(self):
        # draw the figure of same cluster
        import matplotlib.pyplot as plt
        data, y = DataModule().load_pkl( path + 'Train.pkl' )
        data = data.tolist()
        index = DataModule().load_pkl( path + 'Index_draw.pkl' )
        D = [[0, 2, 3, 21, 22], [25, 52, 61, 62, 63, 64, 66, 67, 70, 71],
             [19, 36, 49, 53, 60, 69, 72, 75, 79, 101], [10, 11, 13, 16, 17, 27, 28, 30, 31, 32],
             [131, 132, 154, 157, 161, 174, 178, 199, 200, 232]
             ]
        for c in index:
            print( c, index[c] )
        draw_data = D[2] # 1，4
        draw = []
        for i in draw_data:
            draw.append( data[i] )
        self.multi(draw)
    def draw_map(self):
        import matplotlib.pyplot as plt
        CoordinateX=[41, 40, 40, 39, 38, 37, 36, 35, 34, 33, 32]
        CoordinateY=[13, 14, 14, 15, 15, 15, 16, 15, 16, 16, 16]
        Trajectory=DataModule().load_pkl(path+'Trajectory.pkl')
        CoordinateX=[]
        CoordinateY=[]
        for d in Trajectory:
            for veh in Trajectory[d]:
                for key in Trajectory[d][veh]:
                    for k in Trajectory[d][veh][key]:
                        # print(d,veh,key,k,k[1][0],k[1][1])
                        # print(temp)
                        if k[1][0] in range(0,51) and k[1][1] in range(0,51):
                            CoordinateX.append(k[1][0])
                            CoordinateY.append(k[1][1])
        plt.plot(CoordinateX,CoordinateY,'.')
        plt.show()
    def draw_Ori_Pre(self):
        import matplotlib.pyplot as plt
        from pylab import mpl
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        Ori=DataModule().load_pkl(path+'Original_res.pkl')
        Pre=DataModule().load_pkl(path+'Prediction_res.pkl')
        Query=DataModule().load_pkl(path+'Query.pkl')
        for k in Ori[8]:
            print(k,Query[k],Ori[8][k])
        index=[(41, 15),(34,13),(42,17),(33, 14)]#
        coordinate_x = range( 1, len( Ori[8][(41, 15)] ) + 1 )
        color = ['r', 'g', 'b', 'k','cyan', 'deeppink', 'lime', 'orange', 'slategrey', 'indigo', 'deepskyblue',
                 'gold', 'tan', 'dodgerblue', 'yellow']

        for i in range(4):
            g = index[i]
            # plt.subplot(str(411+i))
            ori='实际值'#Original
            pre='预测值'#Predicted
            plt.plot( coordinate_x, Ori[8][g], color=color[0 % len( color )], label= ori)#label='Original ' + str( i + 1 )
            plt.plot( coordinate_x, Pre[8][g], color=color[0 + 1],label= pre)#label='Predicted ' + str( i + 1 )
            plt.xlabel( '时间段' )#Interval
            # figure.ylabel=('nojijfefefefef')
            plt.ylabel( '车辆数' )#Number of Vehicles
            # plt.title()
            plt.legend()
            plt.show()
        # Draw().draw_pre(O,P,title='')
    def Center(self):
        data=DataModule().load_pkl(path+'class_stand.pkl')
        draw=[]
        for k in data:
            if k not in [5,0,3,4,12,8]:
                draw.append(DataModule().average(data[k]))
                # print(k,data[k])
                print(k,sum(DataModule().average(data[k])))
        # print(data)
        Draw().multi_draw(draw)
    def bar(self):
        path = './File/'
        # fin_id = DataModule().load_list( path + 'fin_idx.npy' )
        # stand = DataModule().load_pkl( path + 'train_stand.pkl' )
        # query = Process().catalogue(fin_id,stand)
        # Fit = Process().generate_fit( stand, query, 10 )
        Loss = {5: {0: 5113, 7: 2204, 11: 6331, 2: 7439, 6: 3144, 10: 320636,
                    8: 12498, 3: 17033, 1: 555078, 4: 2908, 9: 61938, 12: 6815, 5: 58639},
                10: {0: 5113, 7: 1768, 11: 2094, 2: 2315, 6: 2976, 10: 9138,
                     8: 1570, 3: 4342, 1: 19931, 4: 3323, 9: 9661, 12: 2197, 5: 11969},
                15: {0: 5113, 7: 1424, 11: 1558, 2: 1668, 6: 2946, 10: 8123, 8: 4837,
                     3: 3651, 1: 18311, 4: 823, 9: 12371, 12: 1411, 5: 10795}}
        order = [4, 8, 6, 12, 5, 11, 2, 1, 3, 9, 10, 7, 0]
        draw = [[], [], []]
        for k in order:
            draw[0].append( (Loss[5][k]) ** 0.5 )
            draw[1].append( (Loss[10][k]) ** 0.5 )
            draw[2].append( (Loss[15][k]) ** 0.5 )
        Draw().draw_histogram( draw )
        # Draw().multi_draw( Draw, '' )

if __name__ == '__main__':
    path=File_path
    Result().draw_elbow()
    # Result().draw_cluster()
    # Result().draw_map()
    # Result().draw_Ori_Pre()#fit
    # Result().Center()
    # Result().bar()