import importdata,ReadDict,predict
data = {}
originaldata = {}
path = '/home/Jeaten/Program/BayesLstm/File/'
startpreloc = 1
endpreloc = 14
# deltat = 1
bandwidth = 10
modelpath = '/home/Jeaten/Program/BayesLstm/File/BayesGauss/'
data = ReadDict.LoadDict(path)
originaldata = importdata.tackledata(data)
data3dtloc = importdata.generate3dtloc(originaldata)
seqtdict = importdata.generateseqtdict(originaldata)
# [rowmodeldict, clomodeldict] = importdata.LoadLstmModel(modelpath)
[rowmodeldict, clomodeldict] = importdata.importmodel(modelpath)
print('start_predictlocation')
# LstmPredict(Input,RowModelLstm,CloModelLstm)
predictlocation = predict.originalpredict(originaldata, startpreloc, endpreloc, rowmodeldict, clomodeldict)
ReadDict.SaveFile(predictlocation,path+'8thBayespredictlocation.file')
# ReadDict.SaveFile(predictlocation,FilePath)
# predictlocation = ReadDict.readdata(path+"predictlocation.file")#Lstm
# print(predictlocation)
print('end_predictlocation')
# for deltat in range(1, 6):
#     sendresultdict = com.initsendresultdict(originaldata)
#     # cansenddict = com.cansendorder(originaldata, predictlocation,deltat)
#     cansenddict = com.Testcansend(originaldata, predictlocation)
#     com.sendprocess(originaldata, data3dtloc, bandwidth, cansenddict, seqtdict, sendresultdict)
#     totalprofile = statistic.statistic(originaldata, data3dtloc, cansenddict, sendresultdict)
#     print(totalprofile)
