import copy,bayes
import numpy as np

def originalbayespre(inputlist, rowmodeldict, clomodeldict):
    #input style [(1, (1, 2)), (2, (2, 3)), (3, (3, 4))], need to remove time
    # if len(inputlist)<3:
    #     print(inputlist)
    # templist = []
    # for item in inputlist:
    #     templist.append(item[1])
    ### call prediction function
    modelpath = './Bayes_Gauss/'
    # if len(templist) < 3:
    # #     print(templist)
    #     loc = inputlist[-1][1]
    # else:
    loc = bayes.Predict(inputlist, rowmodeldict, clomodeldict)
    # use the last time in inputlist
    preloc = ((inputlist[-1][0], 'pre'), loc)
    return preloc

def bayespre(inputlist, rowmodeldict, clomodeldict):
    #input style [(1, (1, 2)), (2, (2, 3)), (3, (3, 4))], need to remove time
    # if len(inputlist)<3:
    #     print(inputlist)
    templist = []
    for item in inputlist:
        templist.append(item[1])
    ### call prediction function
    modelpath = './Bayes_Gauss/'
    # if len(templist) < 3:
    # #     print(templist)
    #     loc = inputlist[-1][1]
    # else:
    loc = bayes.Predict(templist, rowmodeldict, clomodeldict)
    # use the last time in inputlist
    preloc = ((inputlist[-1][0], 'pre'), loc)
    return preloc

def lstmpre(inputlist,rowmodeldict, clomodeldict):
    # input style [(1, (1, 2)), (2, (2, 3)), (3, (3, 4))], need to remove time
    # if len(inputlist)<3:
    #     print(inputlist)
    templist = []
    for item in inputlist:
        templist.append( item[1] )
    ### call prediction function
    Row = []
    Clo = []
    for j in templist:
        Row.append( j[0] )
        Clo.append( j[1] )
    # print(Feature)
    # print(ModelPath+'Row_Fea'+str(len(Row))+'.m')
    # RowModel=joblib.load(ModelPath+'Row_Fea'+str(len(Row))+'.m')
    # CloModel=joblib.load(ModelPath +'Clo_Fea'+ str(len(Clo))+'.m')
    # print(len(Row))

    Row=np.array([Row])
    Clo=np.array([Clo])
    Row=np.reshape( Row, (Row.shape[0], Row.shape[1], 1) )
    Clo = np.reshape( Clo, (Clo.shape[0], Clo.shape[1], 1) )
    ResRow = rowmodeldict[len( Row[0] )].predict(Row).tolist()
    ResClo = clomodeldict[len( Clo[0] )].predict( Clo ).tolist()
    loc=[int(ResRow[0][0]),int(ResClo[0][0])]
    # return Res[0]
    # use the last time in inputlist
    preloc = ((inputlist[-1][0], 'pre'), loc)
    return preloc


def compress(originallist, copylist, compresslist):
    # returnlist = []
    # copylist->[ ((t, 'pre/ori'),(x,y))  , ((t, 'pre/ori'),(x,y))]
    #compresslist->[(t, (x,y)), (t, (x,y))]
    copylist.append(((originallist[0][0], 'ori'),originallist[0][1]))
    compresslist.append(originallist[0])
    for i in range(1, len(originallist), 1):
        if originallist[i][1] == originallist[i-1][1]:
            copylist.append((originallist[i][0], 'ori'))
        else:
            copylist.append(((originallist[i][0], 'ori'),originallist[i][1]))
            compresslist.append((originallist[i]))
    # if compresslist.__len__() < 3:
    #     print(compresslist)
    return

def originalpredict(originaldata, startpreloc, endpreloc, rowmodeldict, clomodeldict):
    predict = copy.deepcopy(originaldata)
    for vehicle in originaldata:
        for seq in originaldata[vehicle]:
            predictlist = []
            for index in range(len(originaldata[vehicle][seq])):
                if index <= 14:
                    predictlist.append(bayespre(originaldata[vehicle][seq][:index + 1], rowmodeldict, clomodeldict))
                else:
                    ####compress the list [(t, (x,y))]
                    compresslist = copy.deepcopy(originaldata[vehicle][seq][:index + 1])
                    removelist = []
                    for i in range(len(compresslist)-1):
                        if compresslist[i][1] == compresslist[i+1][1]:
                            removelist.append(compresslist[i+1])
                    for item in removelist:
                        compresslist.remove(item)
                        if len(compresslist) == 15:
                            break
                    predictlist.append(bayespre(compresslist[-15:], rowmodeldict, clomodeldict))
            predict[vehicle][seq] = copy.deepcopy(predictlist)
    return predict


# predict = {}
def predictloc_bayes(data,startpreloc, endpreloc, rowmodeldict, clomodeldict):
    # predict = {}
    copylist = []
    compresslist = []
    predict = copy.deepcopy(data)
    j=0
    t = 0
    #predict -> {vehicle: {seq: [((t,'pre'),(x,y)), ((t, 'pre'), (x,y))], seq:[] }}
    for vehicle in data.keys():
        for seq in data[vehicle].keys():
            t += 1
            copylist.clear()
            compresslist.clear()
            compress(data[vehicle][seq], copylist, compresslist)
            ########################################################
            ###### whether adding an if conditione
            # print(copylist)
            # print(compresslist)
            if len(compresslist) > 10:
                # print(compresslist)
                j+=1
            if startpreloc < min(endpreloc+1, len(compresslist)+1):
                for i in range(startpreloc, min(endpreloc+1, len(compresslist)+1), 1):
                    copylist[copylist.index(((compresslist[i-1][0], 'ori'), compresslist[i-1][1]))] = bayespre(compresslist[:i], rowmodeldict, clomodeldict)
                    # if len(compresslist)<3:
                    #     print(vehicle, seq, compresslist)
                # print(copylist)
            else:
                i = min(endpreloc+1, len(compresslist))
                while i > 0:
                    copylist[copylist.index(((compresslist[i - 1][0], 'ori'), compresslist[i - 1][1]))] = bayespre(
                        compresslist[:i], rowmodeldict, clomodeldict)
                    i -= 1

            for i in range(len(copylist)):
                # print(copylist[i])
                # print(len(copylist[i]))
                if copylist[i][1] is 'ori':
                    copylist[i] = ((copylist[i][0], copylist[i-1][0][1]),copylist[i-1][1])
            predict[vehicle][seq] = copy.deepcopy(copylist)
            # if vehicle == '3' and  seq == '8-7':
            #     print('test')
            #     print(predict[vehicle][seq])
    print(j,t)
    return predict
def predictloc_lstm(data,startpreloc, endpreloc, rowmodeldict, clomodeldict):
    # predict = {}
    copylist = []
    compresslist = []
    predict = copy.deepcopy(data)
    j=0
    t = 0
    #predict -> {vehicle: {seq: [((t,'pre'),(x,y)), ((t, 'pre'), (x,y))], seq:[] }}
    for vehicle in data.keys():
        for seq in data[vehicle].keys():

            t += 1
            copylist.clear()
            compresslist.clear()
            compress(data[vehicle][seq], copylist, compresslist)
            ########################################################
            ###### whether adding an if conditione
            # print(copylist)
            # print(compresslist)
            # if len(compresslist) > 10:
            #     # print(compresslist)
            #     j+=1
            if startpreloc < min(endpreloc+1, len(compresslist)+1):
                for i in range(startpreloc, min(endpreloc+1, len(compresslist)+1), 1):
                    # print(compresslist[:i])
                    copylist[copylist.index(((compresslist[i-1][0], 'ori'), compresslist[i-1][1]))] = lstmpre(compresslist[:i], rowmodeldict, clomodeldict)
                    # if len(compresslist)<3:
                    #     print(vehicle, seq, compresslist)
                # print(copylist)
            else:
                i = min(endpreloc+1, len(compresslist))
                while i > 0:
                    copylist[copylist.index(((compresslist[i - 1][0], 'ori'), compresslist[i - 1][1]))] = lstmpre(compresslist[:i], rowmodeldict, clomodeldict)
                    i -= 1

            for i in range(len(copylist)):
                # print(copylist[i])
                # print(len(copylist[i]))
                if copylist[i][1] is 'ori':
                    copylist[i] = ((copylist[i][0], copylist[i-1][0][1]),copylist[i-1][1])
            predict[vehicle][seq] = copy.deepcopy(copylist)
            # if vehicle == '3' and  seq == '8-7':
            #     print('test')
            #     print(predict[vehicle][seq])
    print(j,t)
    return predict