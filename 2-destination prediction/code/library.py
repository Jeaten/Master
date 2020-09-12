# -*- coding: utf-8 -*-
"""
@author: Jeaten
@email: ljt_IT@163.com
this python file is to implement the project of "Segmented Trajectory Clustering-Based
Destination Prediction in IoVs"
@:parameter
LONGITUDE: the entire range of longitude
LATITUDE: the entire range of latitude
PATIENCE: the times we can tolerate during training our GRU model
"""
import numpy as np
LONGITUDE=[-8.65,-8.57] # the entire range of longitude
LATITUDE=[41.10,41.20] # the entire range of latitude
PATIENCE=3 # the times we can tolerate during training our GRU model
class Data:
    '''
    the class is about data operation
    '''
    def __init__(self):
        self.num_min_tra=12 # the minimum number of points
        self.num_max_tra=600 # the maximum number of points
        self.lon_min=LONGITUDE[0] # the minimum longitude
        self.lon_max=LONGITUDE[1] # the maximum longitude
        self.lat_min=LATITUDE[0] # the minimum latitude
        self.lat_max=LATITUDE[1] # the maximum latitude
    def load_pkl(self, pkl_path):
        '''
        load pkl file
        :param pkl_path: pkl dict's whole path
        :return: the loaded pkl dict
        '''
        import pickle
        with open( pkl_path, 'rb' )as file_point:
            return pickle.load( file_point )
    def save_pkl(self,dict,save_path,make_dir=''):
        '''
        save the pickle dict to a fixed file path
        :param dict: the pickle dict we want to save
        :param save_path: the pkl dict's saving path
        :param make_dir: create if the folder does not exist
        :return: True when the function is executed successfully
        '''
        import os,pickle
        try:
            with open( save_path, 'wb' )as file_point:
                pickle.dump( dict, file_point, pickle.HIGHEST_PROTOCOL )
            file_point.close()
        except:
            os.makedirs( make_dir )
            with open( save_path, 'wb' )as file_point:
                pickle.dump( dict, file_point, pickle.HIGHEST_PROTOCOL )
            file_point.close()
        return True
    def read_csv(self, path):
        '''
        this function is used to read '.csv' file
        :param path: the csv file path
        :return: the read data
        '''
        import pandas as pd
        df = pd.read_csv( path )
        return df
    def average(self,data):
        '''
        this function is used to calculate the average of a 2d list 
        :param data: the list data you want to calculate
        :return: the average of 2d list
        '''
        temp=[[],[]]
        for d in data:
            temp[0].append(d[0])
            temp[1].append(d[1])
        return list((np.average(temp[0]),np.average(temp[1])))
    def normalize(self,data):
        '''
        to normalize a series of data
        :param data: the data you want to normalize
        :return: normalized data
        '''
        nor=[]
        nor_min=min(data)
        nor_max=max(data)
        for d in data:
            nor.append((d-nor_min)/(nor_max-nor_min))
        return nor
    def extract_data(self,path):
        '''
        extract trajectory data from original csv file
        :param path: the original csv file path
        :return: the extracted data from original csv file
        '''
        data=self.read_csv(path)
        missing_data=data['MISSING_DATA']
        tra=data['POLYLINE']
        re_tra={}
        num_tra=0
        for i in range(missing_data.__len__()):
            # print(i,missing_data[i])
            if missing_data[i]==False and eval(tra[i]).__len__()>=self.num_min_tra and eval(tra[i]).__len__()<=self.num_max_tra:
                re_tra[num_tra]=eval(tra[i])
                num_tra+=1
                # if num_tra>=30000:
                #     break
        return re_tra
    def filter_length(self,data):
        '''
        filter the trajectories by the number of points(too long or short is useless)
        :param data: the trajectories composed of 2d(longitude and latitude) points
        :return: filtered trajectories
        '''
        tra_fil={}
        num_tra=0
        for k in data:
            count = 0
            for d in data[k]:
                if d[0] >=self.lon_min and d[0] <= self.lon_max and d[1] >= self.lat_min and d[1] <= self.lat_max:
                    count+=1
            if count==data[k].__len__():
                tra_fil[num_tra]=data[k]
                num_tra+=1
        print('Number of tracks after filtering:',num_tra-1)
        return tra_fil
class Segment:
    '''
    this class is about how to segment the trajectories
    '''
    S=[] # to save the segmented sub-trajectory
    def __init__(self):
        self.threshold=0.0005 # the threshold of trajectory segment
    def cal_p2s(self,point,point1,point2):
        '''
        calculate the distance of point to a line segment
        :param point: the point
        :param point1: one of points constituting the line segment
        :param point2: another point constituting the line segment
        :return: the distance of point to a line segment
        '''
        A=point1[1]-point2[1]
        B=point2[0]-point1[0]
        C=point1[0]*point2[1]-point2[0]*point1[1]
        return abs(A*point[0]+B*point[1]+C+0.000000000001)/(A**2+B**2+0.0000000000001)**0.5 # +0.000000000001 to prevent divide 0 zero
    def segment(self,tra):
        '''
        this function is used to separate trajectory
        :param tra: the tra you want to separate # 2d [[1,2],[latitude,longitude],...,[]]
        :return: None
        '''
        # print(tra)
        dis = [self.cal_p2s( tra[i], tra[0], tra[-1] ) for i in range( 1, tra.__len__() - 1 )]
        # print(dis)
        # print( max( dis ), dis )
        dis_max = max( dis )
        if dis_max < self.threshold:
            # print(tra)
            self.S.append( tra.tolist() )
            # return tra
        else:
            # print(np.where(dis==dis_max))
            index = np.where( dis == dis_max )
            # print(index)
            if tra[:index[0][0]].__len__() >= 3:
                # print(tra[:index[0][0]])
                # print(tra[:index[0][0] + 2].__len__())
                self.segment( tra[:index[0][0] + 2] )
            else:
                # print(tra[:index[0][0]])

                self.S.append( tra[:index[0][0] + 2].tolist() )
                # return tra[:index[0][0]]
            if tra[index[0][0] + 1:].__len__() >= 3:
                # print(tra[index[0][0] + 1:].__len__())
                self.segment( tra[index[0][0] + 1:] )
            else:
                # print(tra[index[0][0]-1:])
                self.S.append( tra[index[0][0] + 1:].tolist() )
                # return tra[index[0][0]-1:]
        return
    def fill_cons(self,tra):
        '''
        fill the segmented sub trajectory(if there are some duplicate points, the filled will be also duplicate in this version)
        :param tra: the segmented sub trajectory
        :return: the filled sub trajectory
        '''
        re_tra = []
        for seq in tra:
            # print(seq)
            x = seq[-1][0] - seq[0][0]
            y = seq[-1][1] - seq[0][-1]
            temp=[]
            for t in range( 0, seq.__len__() ):
                temp.append( [seq[0][0] + t * x / (seq.__len__() - 1), seq[0][1] + t * y / (seq.__len__() - 1)] )
            # print(temp)
            re_tra.append(temp)
        # print(re_tra)
        # re_tra.append( tra[-1][-1] )
        # print(re_tra)
        return re_tra
    def segmenting(self,data):
        '''
        to segment trajectories with functions in this class
        :param data: trajectory, dict format:{k:tra(a series of points)}
        :return: segmented trajectories
        '''
        tra_sub={}
        for k in data:
            self.segment(np.array(data[k]))
            key=self.fill_cons(self.S)
            self.S=[]
            tra_sub[k]=key
        return tra_sub
class Cluster:
    '''
    this class is about clustering
    '''
    def __init__(self):
        self.Length_min=3 # not include this value itself
        self.angle_tan = 0.6  # the angle threshold # 0.6 referring to 30
        self.dis_threshold = 0.0008240389678366271 #(80m) # the distance threshold
        # self.angle_tan = 0.2679491924311227  # the angle threshold # 0.6 referring to 30
        # self.angle_tan=0.9999999999999999 # 45
        # self.angle_tan=0.5773502691896257 # 30
        # self.angle_tan=0.2679491924311227 # 15
    def cal_equation(self,point1,point2):
        '''
        calculate the A,B,C of equation point to a line segment(Ax+By+C=0)
        :param point1: one of points constituting the line segment
        :param point2: another point constituting the line segment
        :return: A,B,C
        '''
        A=point1[1]-point2[1]
        B=point2[0]-point1[0]
        C=point1[0]*point2[1]-point2[0]*point1[1]
        return A,B,C
    def Euclid_dis(self, point1, point2):
        '''
        this function is used to calculate the Euclid distance of two 2d points
        :param point1: the first 2d point
        :param point2: the another 2d point
        :return: the Euclid distance
        '''
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    def distance(self, sub_tra1, sub_tra2):
        '''
        calculate the minimum distance of two sub-trajectories
        :param sub_tra1: short trajectory
        :param sub_tra2: long trajectory
        :return: minimum distance of two sub-trajectories
        '''
        dis = []
        if sub_tra1.__len__() <= sub_tra2.__len__():
            for p1 in sub_tra1:
                for p2 in sub_tra2:
                    # print(self.Euclid_dis(p1,p2))
                    dis.append( self.Euclid_dis( p1, p2 ) )
            dis_sum = 0
            for l in range( sub_tra1.__len__() ):
                # print(dis[l*sub_tra2.__len__():(l+1)*sub_tra2.__len__()])
                dis_sum += min( dis[l * sub_tra2.__len__():(l + 1) * sub_tra2.__len__()] )
            # print(dis_sum)
            return dis_sum
        else:
            for p2 in sub_tra2:
                for p1 in sub_tra1:
                    # print( self.Euclid_dis( p1,p2 ) )
                    dis.append( self.Euclid_dis( p1, p2 ) )
            dis_sum = 0
            for l in range( sub_tra2.__len__() ):
                # print(dis[l*sub_tra1.__len__():(l+1)*sub_tra1.__len__()])
                dis_sum += min( dis[l * sub_tra1.__len__():(l + 1) * sub_tra1.__len__()] )
            return dis_sum
    def cal_angle(self, sub_tra1, sub_tra2):
        '''
        calculate the angle of two sub trajectories
        :param sub_tra1: sub trajectory1
        :param sub_tra2: sub trajectory2
        :return: the tan value of two sub trajectories
        '''
        # print('sub1',sub_tra1)
        # print('sub2',sub_tra2)
        # print(sub_tra1[-1],sub_tra1[0])
        # print(sub_tra2[-1],sub_tra2[0])
        k1 = (sub_tra1[-1][1] - sub_tra1[0][1]) / (sub_tra1[-1][0] - sub_tra1[0][0] + 0.000000000000001)
        k2 = (sub_tra2[-1][1] - sub_tra2[0][1]) / (sub_tra2[-1][0] - sub_tra2[0][0] + 0.000000000000001)
        return abs( (k2 - k1) / (1 + k1 * k2 + 0.000000000000001) )
    def cal_min_pos(self, sub_tra1, sub_tra2):
        '''
        calculate the minimum distance of two sub-trajectories
        :param sub_tra1: short trajectory
        :param sub_tra2: long trajectory
        :return: minimum distance of two sub-trajectories
        '''
        if sub_tra1.__len__() <= sub_tra2.__len__():
            dis = [[] for i in range( sub_tra1.__len__() )]
            for i in range( sub_tra1.__len__() ):
                for j in range( sub_tra2.__len__() ):
                    # print(self.Euclid_dis(p1,p2))
                    # print(i)
                    # print( 'i=', i, 'j=', j, self.Euclid_dis( sub_tra1[i], sub_tra2[j] ) )
                    dis[i].append( self.Euclid_dis( sub_tra1[i], sub_tra2[j] ) )
            dis = np.array( dis )
            dis_sum = 0
            position = []
            for i in range( dis.__len__() ):
                dis_sum += min( dis[i] )
                # print(i,min(dis[i]),dis[i],np.where(dis[i]==min(dis[i])))
                position.append( (i, np.where( dis[i] == min( dis[i] ) )[0][0]) )
                # print(i,np.where(dis[i]==min(dis[i]))[0][0])
            # print(position)
            return dis_sum, position
        else:
            dis = [[] for i in range( sub_tra2.__len__() )]
            for i in range( sub_tra2.__len__() ):
                for j in range( sub_tra1.__len__() ):
                    # print( self.Euclid_dis( p1,p2 ) )
                    # print('i=',i,'j=',j,self.Euclid_dis(sub_tra2[i],sub_tra1[j] ))
                    dis[i].append( self.Euclid_dis( sub_tra2[i], sub_tra1[j] ) )
            dis = np.array( dis )
            dis_sum = 0
            position = []
            for i in range( dis.__len__() ):
                dis_sum += min( dis[i] )
                # print(i,min(dis[i]),dis[i],np.where(dis[i]==min(dis[i])))
                position.append( (np.where( dis[i] == min( dis[i] ) )[0][0], i) )
                # print(i,np.where(dis[i]==min(dis[i]))[0][0])
                # print(position)
            # for l in range( sub_tra2.__len__() ):
            #     # print(dis[l*sub_tra1.__len__():(l+1)*sub_tra1.__len__()])
            #     dis_sum += min( dis[l * sub_tra1.__len__():(l + 1) * sub_tra1.__len__()] )
            return dis_sum, position
    def cal_center(self,data,count,length):
        '''
        calculate the cluster center
        :param data: endpoint accumulated value
        :param count: the number of trajectories
        :param length: the length of cluster center we want to add to
        :return: cluster center
        '''
        seq = [[np.average(data[0][0]/count), np.average(data[0][1]/count)], [np.average(data[-1][0]/count), np.average(data[-1][1]/count)]]
        x = seq[-1][0] - seq[0][0]
        y = seq[-1][1] - seq[0][1]
        # print(seq[0][0],seq[0][1])
        center=[]
        for t in range( 0, length ):
            center.append( [seq[0][0] + t * x / (length - 1), seq[0][1] + t * y / (length - 1)] )
        return center
    def cluster(self,tra_seg,name,prefix='w_'):
        '''
        cluster algorithm
        :param tra_seg: the trajectory dict format{k:[sub1,sub2,...]}
        :return: the clustering result
        '''
        self.dis_threshold=name/97082.78749247293
        print( 'clustering',name,'...' )
        data=[]
        for k in tra_seg:
            for sub in tra_seg[k]:
                data.append(sub)
        cluster={} # to save clustering result, dict format:{0:sum,1:center,2:counter,3:total number of points,}
        num_cluster=0
        ### start initialize cluster
        # data[i]=[[longitude,latitude],[longitude,latitude],...,[longitude,latitude]]
        cluster[num_cluster]={0:[[],[]],1:[],2:0,3:0}
        # print(cluster[num_cluster][0])
        data[0]=sorted(data[0]) # sort is solve the problem contravariant trajectories on the same road segment affect computing clustering center
        cluster[num_cluster][0][0].append(data[0][0][0])
        cluster[num_cluster][0][0].append(data[0][0][1])
        cluster[num_cluster][0][-1].append(data[0][-1][0])
        cluster[num_cluster][0][-1].append(data[0][-1][1])
        cluster[num_cluster][1]=data[0]
        cluster[num_cluster][2]+=1
        cluster[num_cluster][3]+=data[0].__len__()
        ### end initialize cluster
        for i in range(1,data.__len__()):
            data[i]=sorted(data[i])
            if data[i].__len__()>self.Length_min:
                dis_temp=[]
                for j in cluster:
                    # print(cluster[j][1])
                    dis,pos=self.cal_min_pos(data[i],cluster[j][1])
                    dis_temp.append(dis/pos.__len__())
                dis_min=min(dis_temp)
                index=dis_temp.index(dis_min)
                if dis_min<=self.dis_threshold:
                    # print(cluster[index][0])
                    cluster[index][0][0][0]+=data[i][0][0]
                    cluster[index][0][0][1]+=data[i][0][1]
                    cluster[index][0][-1][0]+=data[i][-1][0]
                    cluster[index][0][-1][1]+=data[i][-1][1]
                    cluster[index][2] += 1 # updating count must be before center
                    cluster[index][3]+=data[i].__len__()
                    # cal_center_acc( self, data, count, center_curr, center_temp )
                    cluster[index][1]=self.cal_center(cluster[index][0],cluster[index][2],int(cluster[index][3]/cluster[index][2]))
                    # print(cluster[index])
                    # print("cal center")
                else:
                    num_cluster+=1
                    cluster[num_cluster] = {0: [[], []], 1: [], 2: 0,3:0}
                    # print(cluster[num_cluster][0])
                    cluster[num_cluster][0][0].append( data[i][0][0] )
                    cluster[num_cluster][0][0].append( data[i][0][1] )
                    cluster[num_cluster][0][-1].append( data[i][-1][0] )
                    cluster[num_cluster][0][-1].append( data[i][-1][1] )
                    cluster[num_cluster][1] = data[i]
                    cluster[num_cluster][2] += 1
                    cluster[num_cluster][3]+=data[i].__len__()
        return cluster,prefix+'_'+str(name)
    def classify__(self,tra_seg,res_clu):
        '''
        to classify the unknown sub trajectories to existed clusters
        :param train_seg: the unknown sub trajectories
        :param res_clu: the clustering result
        :return: the result of each sub trajectory belongs to which cluster
        '''
        tra_class={}
        for k in tra_seg:
            tra_class[k]=[]
            for i in range(tra_seg[k].__len__()):
                sub=sorted(tra_seg[k][i])
                dis_temp=[]
                for j in res_clu:
                    dis,pos=self.cal_min_pos(sub,res_clu[j][1])
                    dis_temp.append(dis/pos.__len__())
                dis_min=min(dis_temp)
                index=dis_temp.index(dis_min)

                tra_class[k].append(index)
        return tra_class

    def classify(self, tra_seg, res_clu):
        '''
        to classify the unknown sub trajectories to existed clusters
        :param train_seg: the unknown sub trajectories
        :param res_clu: the clustering result
        :return: the result of each sub trajectory belongs to which cluster
        '''
        tra_class = {}
        temp = []
        for k in tra_seg:
            for i in range( tra_seg[k].__len__() ):
                sub = sorted( tra_seg[k][i] )
                dis_temp = []
                for j in res_clu:
                    # print(cluster[j][1])
                    dis, pos = self.cal_min_pos( sub, res_clu[j][1] )
                    dis_temp.append( dis / pos.__len__() )
                dis_min = min( dis_temp )
                index = dis_temp.index( dis_min )
                # if dis_min<=self.dis_threshold and self.cal_angle(sub,cluster[index][1])<=self.angle_tan : #
                #     train_lab.append([sub,index])
                if self.cal_angle( sub, res_clu[index][1] ) <= self.angle_tan:
                    try:
                        tra_class[k].append( index )
                    except:
                        tra_class[k] = []
                        tra_class[k].append( index )
                else:
                    temp.append( k )
                    try:
                        tra_class[k].append( index )
                    except:
                        tra_class[k] = []
                        tra_class[k].append( index )
                    print( k, "error" )
        print( 'Errors', temp.__len__() )
        return tra_class
class Predict:
    '''
    this class is about prediction
    '''
    def __init__(self):
        self.num_cluster=277 # the number of clusters
        self.len_padding=13 # the length of padding
    def angle2radian(self,angle):
        '''
        convert the angle to radians
        :param angle: the angle you want to convert
        :return: the radians of angle
        '''
        return angle*np.pi/180
    def represent(self,tra_class,center):
        '''
        to use the first point and last point to represent a trajectory for training prediction model
        :param tra_class: the trajectories represented in class
        :param center: the clustering center
        :return: the representation of tra_class
        '''
        tra_ess = {}
        for k in tra_class:
            temp = []
            for c in tra_class[k]:
                tra_c_temp = sorted( center[c][1] )
                temp.append([tra_c_temp[0][0]+8.65,tra_c_temp[0][1]-41.10,tra_c_temp[-1][0]+8.65,tra_c_temp[-1][1]-41.10])
            tra_ess[k] = temp
        return tra_ess
    def reverse(self,vec,center):
        center_rep = {}
        label= []
        for k in center:
            tra_c_temp = sorted( center[k][1] )
            center_rep[k]=[tra_c_temp[0][0]+8.65,tra_c_temp[0][1]-41.10,tra_c_temp[-1][0]+8.65,tra_c_temp[-1][1]-41.10]
        for p in vec:
            dis = []
            p1 = p[:2]
            p2 = p[2:]
            for c in center_rep:
                c1 = center_rep[c][:2]
                c2 = center_rep[c][2:]
                dis.append((self.mean_hav_dis(p1,c1)+self.mean_hav_dis(p2,c2))/2)
            index=dis.index( min( dis ) )
            label.append(index)
        return label
    def get_fea_lab(self,data):
        '''
        extract feature and label from the trajectory data composed of classes
        :param data: the trajectory data
        :return: the feature and label we extracted
        '''
        fea=[]
        lab=[]
        for k in data:
            if data[k].__len__()>1:
                fea.append(data[k][:-1])
            else:
                fea.append([data[k][-1]])
            lab.append(data[k][-1])
        return fea,lab
    def get_feature(self,data):
        '''
        extract feature from the trajectory data composed of classes
        :param data: the trajectory data
        :return: the feature we extracted
        '''
        fea=[]
        for k in data:
            fea.append(data[k])
        return fea
    def get_label(self,data):
        '''
        extract the final destination from its true destination location
        :param data: the true destination
        :return: the destination we extracted
        '''
        lab=[]
        for k in data:
            lab.append([data[k][0]+8.65,data[k][1]-41.10])
        return lab
    def get_len_padding(self,data):
        '''
        to acquire the maximum length of feature
        :param data: the feature data
        :return: the maximum length
        '''
        max_len_temp=0
        for d in data:
            if max_len_temp<d.__len__():max_len_temp = d.__len__()
        return max_len_temp
    def padding(self,data,len_padding,value=-1):
        '''
        make all data's length be equal
        :param data: the data we want to pad
        :param len_padding: the length want to make up
        :param value: the value we want to pad with
        :return: the data have been padded
        '''
        from keras_preprocessing import sequence
        return sequence.pad_sequences( data, maxlen=len_padding,dtype='float32', value=value )
    def train_model(self,data,epoch=1000,batch=64,layer=1,neuron=16,validation=0,act='tanh',loss='mse'):
        '''
        train our deep learning model
        :param data: the data we want to train(combination of feature and label)
        :param epoch: training times
        :param batch: batch size
        :param layer: the layer of model
        :param neure: the number of neurons in each layer
        :param validation: the validation
        :param act: the active function
        :param loss: loss function
        :return: the model and model name
        '''
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation, Dropout, Masking
        from keras.layers.recurrent import GRU
        from keras.callbacks import EarlyStopping
        fea,lab=data
        x_train=np.array(fea)
        y_train = np.array( lab )
        model = Sequential()
        model.add( Masking( input_shape=(x_train.shape[1], x_train.shape[2]) ) )
        for i in range(layer-1):
            model.add(GRU(neuron,return_sequences=True))
            model.add(Dropout( 0.2 ) )
        model.add(GRU(neuron,return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(x_train.shape[2]))
        model.add(Activation(act))
        model.compile(loss=loss,optimizer="rmsprop",metrics=['acc'] )
        early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=1,
                                        restore_best_weights=True)
        model.summary()
        module = model.fit(x_train,y_train,epochs=epoch,batch_size=batch,verbose=1,validation_split=validation,
                           callbacks=[early_stopping])
        return module,str(layer)+'_'+str(neuron)+'_'+str(batch)
    def mean_hav_dis(self,point1,point2,radius=6371):
        '''
        calculate the Mean Haversine Distance two points on earth(in kilometers)
        :param point1: the first point
        :param point2: the second point
        :param radius: the radius of earth
        :return: the Mean Haversine Distance of two point on earth
        '''
        longitude_1,latitude_1,longitude_2,latitude_2=point1[0],point1[1],point2[0],point2[1]
        lat=np.sin(self.angle2radian(latitude_1-latitude_2)/2)
        lon=np.sin(self.angle2radian(longitude_1-longitude_2)/2)
        alpha=lat**2+np.cos(self.angle2radian(latitude_1))*np.cos(self.angle2radian(latitude_2))*lon**2
        return  2*radius*np.arctan(np.sqrt(alpha/(1-alpha)))