import time,copy
StartTime=time.time()
import  json,pickle
def LoadDict(FilePath):
    FileDict = open(FilePath,'r' )
    js = FileDict.read()
    Dict= json.loads( js )
    FileDict.close()
    return Dict
def SaveDict(Dict,SavePath):#要存的字典和存字典的路径（路径+名称）
    Save = json.dumps(Dict)
    FileJw = open( SavePath, 'w' )
    FileJw.write( Save )
    FileJw.close()
    return True
def Depart(DeNum,Dict):
    ReDict={}
    for NumVehicle in Dict.keys():
        if Dict[NumVehicle]:
            ReDict[NumVehicle]=Dict[NumVehicle]
        if int(NumVehicle)>DeNum:
            break
    return ReDict
def Schdule():
    DictPath = 'F:\Python\Project\File/8thEarlyPeak.json'
    Num = 1000
    Dict = LoadDict( DictPath )
    DeDict = Depart( Num, Dict )
    SavePath = 'F:\Python\Project\File/Top' + str( Num ) + '.json'
    SaveDict( DeDict, SavePath )
    ###########################
    PrePath = 'F:\Python\Project\File/'
    DictPath = PrePath + 'Top1000.json'  # 8thEarlyPeak
    Dict = LoadDict( DictPath )
    Dict1 = LoadDict( DictPath )
    D = {}
    for NumVehicle in Dict.keys():
        for NumVehicle1 in Dict.keys():
            for Tra1 in Dict1[NumVehicle1].values():
                for J in Tra1:
                    for Tra in Dict[NumVehicle].values():
                        if J in Tra:
                            D.setdefault( str( J ), [] ).append( NumVehicle )
    SaveDict( D, PrePath + 'TimeNeighbors.json' )
    PrePath = 'F:\Python\Project\File/'
    DictPath = PrePath + 'TimeNeighbors.json'  # 8thEarlyPeak
    Dict = LoadDict( DictPath )
    DictTrajectoryPath = PrePath + 'Top100.json'
    DictTrajectory = LoadDict( DictTrajectoryPath )
    TrajectoryNeighbor = {}
    Count = 0
    for NumVehicle in DictTrajectory.keys():
        for k, v in DictTrajectory[NumVehicle].items():
            Temp = []
            Count += 1
            for Point in v:
                Temp.append( [Point[1], len( set( Dict[str( Point )] ) )] )
            TrajectoryNeighbor[Count] = Temp
    SaveDict( TrajectoryNeighbor, PrePath + 'RepeatNeighbor.json' )
def Average(List):
    Ave=0
    if List:
        for i in List:
            Ave+=i
        print(Ave/len(List))
        return Ave/len(List)
Average([1])
def Distinct(List):
    Temp=[]
    Ave=[]
    for i in range(len(List)-1):
        if List[i][0]==List[i+1][0]:
            Ave.append(List[i])
        else:
            Ave.append( List[i] )
            print(Ave)
            Ave=[]
    print(List)
if __name__ == "__main__":
    # Schdule()
    PrePath = '/home/Jeaten/Project/File/'
    DictPath = PrePath + 'TimeNeighbors.json'  # 8thEarlyPeak
    Dict = LoadDict( DictPath )
    DictTrajectoryPath = PrePath + '8thEarlyPeak.json'
    DictTrajectory = LoadDict( DictTrajectoryPath )
    TrajectoryNeighbor = {}
    Count = 0
    for NumVehicle in DictTrajectory.keys():
        for k, v in DictTrajectory[NumVehicle].items():
            Temp = []
            Count += 1
            for Point in v:
                Temp.append( [Point[1], len( set( Dict[str( Point )] ) )] )
            TrajectoryNeighbor[Count] = Temp
    SaveDict( TrajectoryNeighbor, PrePath + 'RepeatNeighbor.json' )
    EndTime=time.time()
    print('Time is:',EndTime-StartTime)