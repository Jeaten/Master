import copy
from sklearn.externals import joblib
from keras.models import load_model
def tackledata(dict):
    tackledict = {}
    for vehicle in dict:
        if dict[vehicle] == {}:
            continue
        tackledict[vehicle] = {}
        for seq in dict[vehicle]:
            tackledict[vehicle][seq] = []
            for item in dict[vehicle][seq]:
                tackledict[vehicle][seq].append((item[0], (item[1][0], item[1][1])))
    for vehicle in tackledict:
        # tempset = set()
        templist = []
        numlist = []
        removelist = []
        for seq in tackledict[vehicle]:
            templist.append(set(tackledict[vehicle][seq]))
            numlist.append(seq)
        for i in range(len(templist) - 1):
            for j in range(i + 1, len(templist)):
                if templist[i].issubset(templist[j]):
                    removelist.append(i)
                elif templist[j].issubset(templist[i]):
                    removelist.append(j)
        if len(removelist) != 0:
            for i in removelist:
                try:
                    tackledict[vehicle].__delitem__(numlist[i])
                except:
                    print( 'KeyError!' )
    tempdict = copy.deepcopy(tackledict)
    for vehicle in tackledict:
        tempset = set()
        templist = []
        for seq in tackledict[vehicle]:
            for tempseq in tackledict[vehicle]:
                if seq == tempseq:
                    continue
                else:
                    if set(tackledict[vehicle][seq]).intersection(set(tempdict[vehicle][tempseq])) != set():
                        if len(tackledict[vehicle][seq]) < len(tempdict[vehicle][tempseq]):
                            if seq not in templist:
                                templist.append(seq)
                        else:
                            if tempseq not in templist:
                                templist.append(tempseq)
            #
            # if tempset.intersection(set(tackledict[vehicle][seq])) != set():
            #     # print(vehicle)
            #     templist.append(seq)
            # else:
            #     tempset = tempset.union(set(tackledict[vehicle][seq]))
        for seq in templist:
            tackledict[vehicle].__delitem__(seq)

    for vehicle in tackledict:
        tempset = set()
        # templist = []
        for seq in tackledict[vehicle]:
            if tempset.intersection(set(tackledict[vehicle][seq])) != set():
                print(vehicle)
                # templist.append(seq)
            else:
                tempset = tempset.union(set(tackledict[vehicle][seq]))
        # for seq in templist:
        #     tackledict[vehicle].__delitem__(seq)
    return tackledict

def generate3dloct(dict):
    dataloct = {}
    for key in dict.keys():
        for value in dict[key].values():
            for item in value:
                # print item
                if not dataloct.has_key(item[1]):
                    dataloct[item[1]]= {item[0]:[key]}
                else:
                    if not dataloct[item[1]].has_key(item[0]):
                        dataloct[item[1]][item[0]] = [key]
                    else:
                        dataloct[item[1]][item[0]].append(key)
    return dataloct

# originaldata = {'v1':{"1-1":[(1,(1,2)), (2, (2,3)), (3,(3,4))],"1-2":[(5, (5,6)), (6, (7,8)), (7, (8,9))]}}

def generate3dtloc(dict):
    datatloc = {}#datatloc->{t: {loc :[v1,v2], loc: [v3,v4]}
    for key in dict.keys():
        for value in dict[key].values():
            for item in value:
                # print(item)
                if item[0] not in datatloc:
                    # print(item[1])
                    datatloc[item[0]]= {item[1]:[key]}
                else:
                    if item[1] not in datatloc[item[0]]:
                        datatloc[item[0]][item[1]] = [key]
                    else:
                        datatloc[item[0]][item[1]].append(key)
    return datatloc

def generateseqtdict(data):
    #seqtdict -> {vehicle: {t: seq, t2: seq2}}
    seqtdict = {}
    for vehicle in data.keys():
        seqtdict[vehicle] = {}
        for seq in data[vehicle].keys():
            for item in data[vehicle][seq]:
                seqtdict[vehicle][item[0]] = seq
    return seqtdict

def generatetimeindex(data):
    timeindex = {}
    for vehicle in data:
        timeindex[vehicle]={}
        for seq in data[vehicle]:
            timeindex[vehicle][seq]={}
            i = 0
            for item in data[vehicle][seq]:
                timeindex[vehicle][seq][item[0]] = i
                i += 1
    return  timeindex

def importmodel(ModelPath):
    rowmodeldict = {}
    clomodeldict = {}
    for length in range(1, 10,1):
        rowmodeldict[length] = joblib.load(ModelPath+'Row_Fea'+str(length)+'.m')
        clomodeldict[length] = joblib.load(ModelPath + 'Clo_Fea' + str(length) + '.m')
    return rowmodeldict, clomodeldict

def LoadLstmModel(LstmModelPath):
    rowmodeldict = {}
    clomodeldict = {}
    for length in range( 1, 10, 1 ):
        rowmodeldict[length] = load_model( LstmModelPath+ 'Lstm_Row_Fea_' + str( length) )
        clomodeldict[length] = load_model( LstmModelPath + 'Lstm_Clo_Fea_' + str( length ) )
    return rowmodeldict, clomodeldict
