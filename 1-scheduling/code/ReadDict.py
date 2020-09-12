import os,time,datetime,json, pickle

def LoadDict(Path):#将存为文件的轨迹转换为字典
    FileDict = open( Path , 'r' )
    js = FileDict.read()
    Dict= json.loads( js )
    FileDict.close()
    return Dict
def load_pkl(pkl_path):
    '''
    load pkl file like json above
    :param pkl_path: pkl dict's whole path
    :return: the loaded pkl dict
    '''
    import pickle
    with open( pkl_path, 'rb' )as file_point:
        return pickle.load( file_point )
def ViewTrace(Dict):
    for key,value in Dict.items():
        print(key,value)
def ViewDictVehicleNum(Dict_Grid):#按车辆数目查看字典轨迹
    Vehicle=[1]#第1辆车
    for i in Vehicle:
        for key,value in Dict_Grid[str(i)].items():
            print(key,value)
def ViewDictDay(Dict_Grid):#按天查看字典轨迹
    InputDay=[1]#第1天
    for NumVehicle in range(1,len(Dict_Grid)+1):
        for Day_TraNum,Trajectory in Dict_Grid[str(NumVehicle)].items():
            Temp=Day_TraNum.split('-')
            for i in InputDay:
                if Temp[0]==str(i):
                    print(Day_TraNum,Trajectory)
def ViewDictTime(Dict_Grid):#按时间段查看字典轨迹
    time=[1440.0,2880]#1440mins-2880mins为第二天
    for NumVehicle in range( 1, len( Dict_Grid ) + 1 ):
        for Day_TraNum, Trajectory in Dict_Grid[str( NumVehicle )].items():
            if time[0]<float(Trajectory[0][0]) and float(Trajectory[0][0])<time[1]:
                print(Day_TraNum,Trajectory)
if __name__ == "__main__":
    Dict=LoadDict('D:\Jeaten\Master\IOV\深圳出租车\gps_data/')
    ViewDictTime(Dict)
    # ViewTrace( Dict )

def savedata(data):
    with open("predictlocation.file","wb") as f:
       pickle.dump(data, f)
       f.close()

def readdata(path):
    with open(path,"rb") as f:
        data = pickle.load(f)
        f.close()
    return data
