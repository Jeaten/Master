# -*- coding: utf-8 -*-
"""
Created on Thursday,3 October 10:47 2019

Modified on 3 October - 

@author:Jeaten
"""
import pickle, json, time, random
import numpy as np


class DataModule:
    def __init__(self):
        self.delta = 15  ###interval of departing dict's key by time###
        self.day = 1440  # every day has 1440 minutes in minutes#
        self.interval = 96  # the interval divide a day with 15 minutes
        self.longitudeUp = 50  # the up bound of longitude
        self.longitudeLower = 0  # the lower bound of longitude
        self.latitudeUp = 50  # the up bound of latitude
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

    def save_json(self, Dict, SavePath):
        '''
        save the json dict to a fixed path
        :param Dict: the dict we want to save
        :param SavePath: the dict path we want to save
        :return: whether the function is executed successfully
        '''
        save = json.dumps( Dict )
        save_point = open( SavePath, 'w' )
        save_point.write( save )
        save_point.close()
        return True

    def save_pkl(self, Dict, SavePath):
        '''
        save the pickle dict to a fixed path
        :param Dict: the pickle dict we want to save
        :param SavePath: the pkl dict's saving path
        :return: whether the function is executed successfully
        '''
        with open( SavePath, 'wb' )as file_point:
            pickle.dump( Dict, file_point, pickle.HIGHEST_PROTOCOL )
        file_point.close()
        return True

    def load_list(self, path):
        '''
        load the saved list from file
        :param path: the path of file
        :return: the list format data
        '''
        array = np.load( path, encoding="latin1" )
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

    def transform_intuitive(self, data):
        '''
        transform the original data to intuitive format
        :param data: original dict data whose format likes this:{(521, (35, 21)):['304'],(4638, (42, 21)):['304']}
        ((min,(longitude,latitude)):[vehicles]--where min is minute,(longitude,latitude)is the grid consists of longitude and latitude
        and vehicles are which vehicles in detail)

        the data format of we except to get is(intuitive format):
        {((day,interval),(longitude,latitude)):[vehicles]} where day is the day-th we transform minutes to day whose range should be 1-9,
        interval is interval-th of a day, if the time is 15 minutes then the range of interval is 0-96, because a day has 1440 mins and 1440/15=96,
        as well,(longitude,latitude) is the grid location and vehicles are these specific vehicles
        :return:the intuitive format data we have transformed
        '''
        '''
        original data format:
        (519, (41, 15)) : ['1', '2134', '2598', '4115', '4853', '6071', '6221', '6710', '6890', '7306', '7425', '7859', '8961']
        (520, (41, 15)) : ['1', '2134', '2598', '3606', '3671', '4853', '5355', '6057', '6710', '6890', '7306', '7319', '7332', '7425', '8961']
        (521, (41, 15)) : ['1', '1072', '2134', '3561', '3606', '3671', '4511', '4853', '5004', '5355', '6057', '6221', '6710', '7306', '7319', '7332', '7425', '8597', '8788']
        (522, (41, 15)) : ['1', '1072', '1277', '3561', '3671', '4853', '6057', '6710', '7016', '7306', '7332', '7374', '8788']
        transformed data format:
        ((6, 72), (60, 20)) : ['1780', '1780']
        ((6, 72), (9, 27)) : ['4461', '4461', '7086', '7086', '4461', '1901', '1901', '7086']
        ((6, 72), (13, 28)) : ['2374', '1111', '897']
        ((6, 72), (14, 19)) : ['6927', '6927', '6927', '6927', '6927']
        ((6, 72), (14, 34)) : ['7657', '2374', '2374']
        '''
        TrainData = {}
        for key in sorted( data ):
            try:
                # day=int(key[0]/self.day)+1
                # min=int((key[0]%1440)/self.delta)+1
                # print(day,min)
                # print([(int(key[0]/self.day)+1,int((key[0]%self.day)/self.delta)+1),(41,22)])
                TrainData[
                    tuple( [(int( key[0] / self.day ) + 1, int( (key[0] % self.day) / self.delta ) + 1), key[1]] )] += \
                data[key]

            except:
                TrainData[
                    tuple( [(int( key[0] / self.day ) + 1, int( (key[0] % self.day) / self.delta ) + 1), key[1]] )] = []
                TrainData[
                    tuple( [(int( key[0] / self.day ) + 1, int( (key[0] % self.day) / self.delta ) + 1), key[1]] )] += \
                data[key]
        return TrainData

    def transform_vector(self, data):
        '''
        transform intuitive data's format to vector format
        :param data: intuitive data(dictionary format)
        :return: 
        '''
        '''
        original dict format:
        ((1, 1), (19, 25)) : ['8918']
        ((1, 1), (19, 27)) : ['7206']
        ((1, 1), (20, 7)) : ['5147', '7606', '3722', '3722']
        ((1, 1), (20, 8)) : ['2426', '2426', '2426', '5147', '1237', '7606', '7606', '3722', '3722', '3299', '6520']
        ((1, 1), (20, 9)) : ['2426']
        transformed data format:
        (9, (46, 27)) : {45: 1}
        (9, (46, 28)) : {3: 1, 12: 1, 16: 1, 31: 1, 39: 1, 45: 2, 47: 1}
        (9, (46, 29)) : {1: 1, 2: 1, 3: 2, 4: 1, 12: 1, 16: 1, 18: 1, 19: 1, 21: 1, 32: 1, 33: 1, 45: 2, 47: 2}
        (9, (46, 30)) : {2: 2, 3: 1, 15: 1, 16: 1, 19: 1, 25: 1, 26: 1, 45: 1}
        (9, (46, 31)) : {1: 3, 3: 1, 4: 1, 15: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 2, 25: 1, 26: 2, 27: 3, 28: 1, 29: 1, 32: 1, 33: 1, 35: 4, 37: 2, 39: 1, 40: 2, 41: 1, 42: 1, 43: 1, 44: 1, 45: 2, 46: 3, 48: 3, 49: 1}
        (9, (46, 32)) : {11: 1, 29: 2, 39: 1, 40: 1, 44: 1, 46: 1}
        '''
        Dict = {}
        for key in sorted( data ):
            try:
                Dict[tuple( [key[0][0], key[1]] )][key[0][1]] = len( set( data[key] ) )
                # print(tuple([key[0][0],key[1]]))
                # print(Dict)
            except:
                Dict[tuple( [key[0][0], key[1]] )] = {}
                Dict[tuple( [key[0][0], key[1]] )][key[0][1]] = len( set( data[key] ) )
                # print(Dict)
        return Dict

    def standardize(self, data):
        '''
        standardize the vector format dict(transform its length to be self.interval(96 is an example))
        :param data: vector format pickle dictionary data
        :return: standardized dict(but not normalize)
        '''
        Dict = {}
        for key in sorted( data ):
            temp = []
            for i in range( self.interval ):
                try:
                    temp.append( data[key][i] )
                except:
                    temp.append( 0 )
            Dict[key] = temp
        # for key in Dict:
        #     print(key,Dict[key])
        # print(len(Dict[(9, (46, 32))]))
        return Dict

    def normalize(self, data):
        '''
        normalize and denoise the standardized dict so that we can use normalized data to train our model
        :param data: standardize dict(whose format is{(day,(longitude,latitude)):[0---95]})
        :return: the normalized and denoised data with key
        '''
        Dict = {}
        for key in data:
            temp = []
            '''denoise(we regard the grid as noise if gird's range is not in 0-50)'''
            if key[1][0] in range( self.longitudeLower, self.longitudeUp ) and key[1][1] in range( self.latitudeLower,
                                                                                                   self.latitudeUp ):
                for i in data[key]:
                    try:
                        '''normalize'''
                        temp.append( i / max( data[key] ) )
                    except:
                        temp.append( 0 )
                Dict[key] = temp
        # for key in Dict:
        #     print(key,Dict[key])
        return Dict


class Draw:
    def __init__(self):
        self.path = ''

    def draw(self, data):
        import matplotlib.pyplot as plt
        coordinate_x = range( 1, len( data ) + 1 )
        # print(coordinate_x)
        # for i in coordinate_x:
        #     print(i)
        plt.plot( coordinate_x, data )
        plt.show()
        return True

    def multi_draw(self, data, title):
        import matplotlib.pyplot as plt
        coordinate_x = range( 1, len( data[0] ) + 1 )
        # label=[]
        color = ['r', 'g', 'b', 'cyan', 'k', 'r', 'g', 'b', 'cyan', 'k', 'r', 'g', 'b', 'cyan', 'k']
        for i in range( len( data ) ):
            plt.plot( coordinate_x, data[i], color=color[i % len( color )], label=str( i + 1 ) )
        plt.title( title )
        plt.legend()
        plt.show()
        return True


# D=Draw()
# D.multi_draw([[1,2,3],[2,3,1]])
class Process:
    def __init__(self):
        # self.path='F:\Python\Project\List/'
        self.path = './List/'
        self.data = 'fin_idx.npy'
        self.hidden = 'cluster_data.npy'
        self.center = 'fin_centers.npy'

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

    def Eucdistance(self, listA, listB):
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
                sse += self.Eucdistance( hidden[i], center[k] )
                # print(sse)
            SSE += sse
            # print(SSE)
        return SSE

    '''
    a test example of SSE
    # # p.Eucdistance([1,3],[1.5,2.5])
    # Kdict={0:[0,2],1:[1]}
    # hidden=[[2,2,2,2],[1,1,1,1],[0,0,0,0],[1,3,4,5]]
    # center=[[0,1,0,0],[1,1,1,1]]
    # print(p.SSE(Kdict,hidden,center))
    '''

    def elbow(self):
        d = DataModule()
        Kdict = d.load_list( self.path + self.data )
        Kdict = self.class_reverse( Kdict )
        # print( Kdict )
        # for k in sorted(Kdict):
        #     print(k)
        hidden = d.load_list( self.path + self.hidden )
        # print(hidden)
        # for i in hidden:
        #     print(len(i),i)
        center = d.load_list( self.path + self.center )
        SSE = self.SSE( Kdict, hidden, center )
        # print(SSE)
        return SSE

class Schedule:
    def __init__(self):
        self.path = 'F:\Python\Project\Fitting\File/'
        self.ICCpath = 'F:\Python\Project\List/'
        # self.name = 'Normalized_Train_list.pkl'
        self.name = 'ICC_train_set.pkl'
        self.ICCname = 'original_y.npy'

    def schedule_data(self):
        Exe = DataModule()
        data_y = Exe.load_list( self.ICCpath + self.ICCname )
        # DataOperate = DataModule()
        x, Data = Exe.load_pkl( self.path + self.name )
        print( len( Data ) )
        a = 0
        print( Data[a], data[a] )
        for i in Data:
            print( i )
            # for i in range(0,9984):
            #     if data_y[i]==Data[i]:
            #         print('ok')
            #         print(i)
            # Data=Exe.transform_intuitive(Data)
            # DataOperate.save_pkl(Data,'F:\Python\Project\Fitting\File/Intuitive.pkl')
            # Data=Exe.transform_vector(Data)
            # Data=Exe.standardize(Data)
            # Data=Exe.normalize(Data)
            # DataOperate.save_pkl(Data,'F:\Python\Project\Fitting\File/Normalized_Key.pkl')
            # List=[]
            # for key in Data:
            #     List.append(Data[key])
            # print(key,':',Data[key])
            # DataOperate.save_pkl(List,'F:\Python\Project\Fitting\File/Normalized_Train_list.pkl')
            # for i in Data:
            #     print(i)
            # print(Data)
            # Data=np.array(Data)
            # print(Data)
            # print(len(Data))
            # print(len(Data[0]))
            # Label=[]
            # for i in range(10188):
            #     Label.append(random.randint(0, 9 ))
            # Label=np.array(Label)
            # dataset=[Data,Label]
            # D=DataModule()
            # D.save_pkl(dataset,'F:\Python\Project\Fitting/ICC_train_set.pkl')
            # print(dataset)
            # x,y=dataset
            # print(x)
            # print(y)

    def test(self):
        y = []
        for i in range( 10188 ):
            y.append( random.randint( 0, 9 ) )
        # print(self.path)
        # print(np.array(y))
        x = [[1, 2, 3, 4], [5, 6, 7, 8]]
        y = [0, 1]
        x = np.array( x )
        y = np.array( y )
        data = [x, y]
        # x,y=np.array(data)
        print( data )
        print( x, y )

    def used_in_main(self):
        full = []
        for i in range( len( distribution ) ):
            Count = 0
            # if distribution[i]==96:
            for j in distribution[i]:
                if j != 0:
                    Count += 1
                    # print(Count,end=' ')
            if Count == 95:
                full.append( i )
                # print(i)
        # print(full)
        Dict = {}
        for k in sorted( d ):
            for j in full:
                if j in d[k]:
                    try:
                        Dict[k].append( j )
                    except:
                        Dict[k] = []
                        Dict[k].append( j )
        # print(Dict)
        for k in Dict:
            print( k, Dict[k] )

    def view_cluster(self):
        d = DataModule()
        # data=d.load_list('F:\Python\Project\List/fin_idx.npy')
        # # S.schedule_data()
        # # S.test()
        # p=Process()
        # data=p.class_reverse(data)
        distribution, label = d.load_pkl( 'F:\Python\Project\Fitting\File/ICC_train_set.pkl' )
        distribution = distribution.tolist()
        # label = label.tolist()
        # d = d.load_pkl( 'F:\Python\Project\Fitting\File/Every_cluster.pkl' )
        # loc=[2, 5,9, 13, 14,  ]#18, 19, 24, 28, 29, 45, 46, 47, 48, 49, 51, 57, 67
        loc = [3963, 3984, 4466, 5004, 5985, 6705, 6761, 7264, 7661, 7871, 7953, 7975, 9089]
        view = []
        for i in loc:
            view.append( distribution[i] )
        D = Draw()
        D.multi_draw( view, 'class 8' )
        # print(view)
        # for i in range(len(distribution)):
        #     view.append(distribution[i])
        # print(distribution.tolist())
        # print(label.tolist())
        # for k in sorted(data):
        #     print(k,data[k])
if __name__ == "__main__":
    s = time.time()
    p = Process()
    SSE = p.elbow()
    print( SSE )
    # S = Schedule()
    # S.view_cluster()
    e = time.time()
    print( 'Time is:', e - s )