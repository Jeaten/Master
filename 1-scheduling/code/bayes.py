from sklearn.externals import joblib
# import joblib
def Predict(Feature,rowmodeldict, clomodedict):
    Row=[]
    Clo=[]
    Res = []
    for j in Feature:
        Row.append(j[0])
        Clo.append(j[1])
    # print(Feature)
    # print(ModelPath+'Row_Fea'+str(len(Row))+'.m')
    # RowModel=joblib.load(ModelPath+'Row_Fea'+str(len(Row))+'.m')
    # CloModel=joblib.load(ModelPath +'Clo_Fea'+ str(len(Clo))+'.m')
    # print(len(Row))
    ResRow=rowmodeldict[len(Row)].predict([Row]).tolist()
    ResClo=clomodedict[len(Clo)].predict([Clo]).tolist()
    for i in range(0,len(ResRow)):
        Res.append((ResRow[i],ResClo[i]))
    return Res[0]
# if __name__ == "__main__":
#     ModelPath='F:\Python\File\Bayes_Gauss/'#存储模型的路径
#     Feature=[(41,14),(40,14),(39,13)]#想要预测的特征点
#     print(Predict(Feature,ModelPath))