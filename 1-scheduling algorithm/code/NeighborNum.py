import time,copy
StartTime=time.time()
import  json
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
    Num = 100
    Dict = LoadDict( DictPath )
    DeDict = Depart( Num, Dict )
    SavePath = 'F:\Python\Project\File/Top' + str( Num ) + '.json'
    SaveDict( DeDict, SavePath )
if __name__ == "__main__":
    # Schdule()
    PrePath='/home/Jeaten/Project/File/'
    DictPath=PrePath+'8thEarlyPeak.json'#Top100
    TimeVehicle={}
    Dict=LoadDict(DictPath)
    Dict1 = LoadDict( DictPath )
    D={}
    for NumVehicle in Dict.keys():
        for NumVehicle1 in Dict.keys():
            for Tra1 in Dict1[NumVehicle1].values():
                for J in Tra1:
                    for Tra in Dict[NumVehicle].values():
                        if J in Tra:
                            D.setdefault( str( J ), [] ).append( NumVehicle )
    SaveDict( D, PrePath + 'TimeNeighbors.json' )
    EndTime=time.time()
    print('Time is:',EndTime-StartTime)