import time,json
StartTime=time.time()
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
def RandDistinct(List):
    Temp=[]
    # Ave=[]
    Temp.append(List[0])
    # Ave.append(List[0])
    for i in range(1,len(List)):
        if List[i][0]==Temp[-1][0]:
            continue
            # Ave.append(List[i])
        else:
            Temp.append(List[i])
    return Temp
if __name__ == "__main__":
    a = [1, 2, 3]
    b = [1, 2,4]
    ret1 = list( set( a ).union( set( b ) ) )
    print(ret1)
    # PrePath = '/home/Jeaten/Project/File/'
    # RepeatNeighbor=PrePath+'RepeatNeighbor.json'
    # RepeatNeighbor=LoadDict(RepeatNeighbor)
    # RandDisNeighbor={}
    # for Num,ReNei in RepeatNeighbor.items():
    #     RandDisNeighbor[Num]=RandDistinct(ReNei )
    # SaveDict( RandDisNeighbor, PrePath+'RandDisNeighbor.json' )
    EndTime=time.time()
    print('Time is:',EndTime-StartTime)