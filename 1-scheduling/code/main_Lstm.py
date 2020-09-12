import importdata, predict, com, newstatistic, ReadDict,statistic
data = {}
originaldata = {}
path = '/home/Jeaten/Jeaten/Profit/'
startpreloc = 1
endpreloc = 9
# deltat = 1
bandwidth =1
ManhatunDis=0
ProcessFarNear=True
# modelpath = '/home/Jeaten/Jeaten/Model_Lstm/'
modelpath='/home/Jeaten/Project/Program/BayesLstm/File/Lstm/'
data = ReadDict.LoadDict(path+'8thOnline.json')
originaldata = importdata.tackledata(data)
data3dtloc = importdata.generate3dtloc(originaldata)
seqtdict = importdata.generateseqtdict(originaldata)
# LoadLstmModel(LstmModelPath)
[rowmodeldict, clomodeldict] = importdata.LoadLstmModel(modelpath)
print('start_predictlocation')#
predictlocation = predict.predictloc_lstm(originaldata, startpreloc, endpreloc, rowmodeldict, clomodeldict)
ReadDict.savedata(predictlocation)
# ReadDict.savedata(data)
# [rowmodeldict, clomodeldict] = importdata.importmodel(modelpath)

predictlocation = ReadDict.readdata(path+"8thLstmpredictlocation.file")
print('end_predictlocation')
list = []
X=[]
Y=[]
Z=[]
for deltat in range(1, 10):
    sendresultdict = com.initsendresultdict(originaldata)
    # cansenddict = com.cansendorder(originaldata, predictlocation,deltat)
    cansenddict = com.cansendlast( originaldata, predictlocation, deltat,ManhatunDis )
    com.sendprocess(originaldata, data3dtloc, bandwidth, cansenddict, seqtdict, sendresultdict,ProcessFarNear)
    returnprofile, totalmessage, rate = statistic.statistic(originaldata, data3dtloc, cansenddict, sendresultdict)
    X.append(returnprofile)
    Y.append(totalmessage)
    Z.append(rate)
    print(deltat,returnprofile, totalmessage, rate)
print('[',X,',')
print(Y,',')
print(Z,']')
