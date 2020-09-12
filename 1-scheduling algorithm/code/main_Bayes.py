import importdata, predict, com, newstatistic, ReadDict,statistic
data = {}
originaldata = {}
path = '/home/Jeaten/Program/BayesLstm/File/'
startpreloc = 1
endpreloc = 14
# deltat = 1
bandwidth =3
ManhatunDis=1
ProcessFarNear=True
modelpath = '/home/Jeaten/Program/BayesLstm/File/BayesGauss/'
data = ReadDict.LoadDict(path)
originaldata = importdata.tackledata(data)
data3dtloc = importdata.generate3dtloc(originaldata)
seqtdict = importdata.generateseqtdict(originaldata)
[rowmodeldict, clomodeldict] = importdata.importmodel(modelpath)
predictlocation = ReadDict.readdata(path+"8thBayespredictlocation.file")
print('end_predictlocation')
list = []
X=[]
Y=[]
Z=[]
for deltat in range(1, 16):
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
