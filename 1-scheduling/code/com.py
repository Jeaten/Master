import copy, random
def initsendresultdict(data):
    sendresult = {}
    for vehicle in data.keys():
        sendresult[vehicle] = {}
        for seq in data[vehicle]:
            sendresult[vehicle][seq]=[]
    return  sendresult

def cansendrand(originaldata, predictloc, deltat):
    cansenddict = {}
    for vehicle in predictloc.keys():
        cansenddict[vehicle] = {}
        for seq in predictloc[vehicle].keys():
            # print(predictloc[vehicle][seq])
            cansenddict[vehicle][seq] = []
            index = random.randint(1, min(len(predictloc[vehicle][seq])-1, 15))
            cansenddict[vehicle][seq].append(((predictloc[vehicle][seq][index][0][0], 'cansend'), predictloc[vehicle][seq][index][1]))
    return  cansenddict

def cansendorder(originaldata, predictloc, order):
    # for vehicle in originaldata:
    #     for seq in originaldata[vehicle]:
    #         for i in range(len(originaldata[vehicle][seq])):
    #             if originaldata[vehicle][seq][i][0] == predictloc[vehicle][seq][i][0][0]:
    #                 print(originaldata[vehicle][seq])
    #                 print(predictloc[vehicle][seq])
    # print('endendend')
    cansenddict = {}
    for vehicle in predictloc:
        cansenddict[vehicle] = {}
        for seq in predictloc[vehicle]:
            # print(predictloc[vehicle][seq])
            cansenddict[vehicle][seq] = []
            if order > len(predictloc[vehicle][seq]) > 0:
                # if len(predictloc[vehicle][seq])>0:
                #     cansenddict[vehicle][seq].append(
                #         ((predictloc[vehicle][seq][-1][0][0], 'cansend'), (predictloc[vehicle][seq][-1][1])))
                continue
            else:
                # index = order-1
                # # while index >= 0:
                templist = []
                i = -1
                for item in originaldata[vehicle][seq]:
                    i = i+1
                    if item[1] not in templist:
                        templist.append(item[1])
                        if len(templist) == order:
                            index = i
                            cansenddict[vehicle][seq].append(((predictloc[vehicle][seq][index][0][0], 'cansend'), (predictloc[vehicle][seq][index][1])))
                            break
                if len(templist) < order:
                    # index = templist[-1]
                    pass
                    # cansenddict[vehicle][seq].append(((predictloc[vehicle][seq][order-1][0][0], 'cansend'), (predictloc[vehicle][seq][order-1][1])))

    # for vehicle in cansenddict:
    #     for seq in cansenddict[vehicle]:
    #         if vehicle == '6962' and seq == '8-4':
    #             print(vehicle, seq, cansenddict[vehicle][seq])
    return cansenddict

def cansendbest(data, predictloc, order):
    #cansenddict -> {vehicle: {seq:[  ( (t, 'cansend'),(preloc) ),   ]} }
    cansenddict = {}
    for vehicle in predictloc:
        cansenddict[vehicle] = {}
        for seq in predictloc[vehicle]:
            cansenddict[vehicle][seq] = []
            i = 0
            while i < len(predictloc[vehicle][seq]):
                if manhdistance(predictloc[vehicle][seq][i][1], data[vehicle][seq][-1][1]) <= 0:
                    cansenddict[vehicle][seq].append(
                        ((predictloc[vehicle][seq][i][0][0], 'cansend'), predictloc[vehicle][seq][i][1]))
                    # if data[vehicle][seq][i][1] != data[vehicle][seq][i-1][1]and \
                    #         manhdistance(predictloc[vehicle][seq][i][1], predictloc[vehicle][seq][i-1][1])<=0:
                    #     print('satisfy', i, len(data[vehicle][seq]))
                    # else:
                    #     print('not', i, len(data[vehicle][seq]))
                    break
                i = i + 1
            if len(cansenddict[vehicle][seq]) == 0:
                i = 0
                while i < len(predictloc[vehicle][seq]):
                    if manhdistance(predictloc[vehicle][seq][i][1], data[vehicle][seq][-1][1]) == 1:
                        cansenddict[vehicle][seq].append(
                            ((predictloc[vehicle][seq][i][0][0], 'cansend'), predictloc[vehicle][seq][i][1]))
                        break
                    i += 1
            if len(cansenddict[vehicle][seq]) == 0:
                i = 0
                while i < len(predictloc[vehicle][seq]):
                    if manhdistance(predictloc[vehicle][seq][i][1], data[vehicle][seq][-1][1]) == 2:
                        cansenddict[vehicle][seq].append(
                            ((predictloc[vehicle][seq][i][0][0], 'cansend'), predictloc[vehicle][seq][i][1]))
                        break
                    i += 1

    return cansenddict

def cansenddelay(data, predictloc, order):
    #cansenddict -> {vehicle: {seq:[  ( (t, 'cansend'),(preloc) ),   ]} }
    cansenddict = {}
    for vehicle in predictloc:
        cansenddict[vehicle] = {}
        for seq in predictloc[vehicle]:
            cansenddict[vehicle][seq] = []
            i = 1
            while i < len(predictloc[vehicle][seq]):
                # if manhdistance(data[vehicle][seq][i][1], predictloc[vehicle][seq][i][1]) <= \
                #         manhdistance(data[vehicle][seq][i-1][1], predictloc[vehicle][seq][i-1][1]):
                if manhdistance(data[vehicle][seq][i][1], predictloc[vehicle][seq][i][1]) <= 1:
                    cansenddict[vehicle][seq].append(
                        ((predictloc[vehicle][seq][i][0][0], 'cansend'), predictloc[vehicle][seq][i][1]))
                    break
                i += 1



    return cansenddict

def AveList(List):
    AveX=0
    AveY=0
    templist = []
    for item in List:
        templist.append(item[1])
    for i in templist:
        AveX+=i[0]
        AveY+=i[1]
    return (int(AveX/len(List)),int(AveY/len(List)))

def cansendlast(data, predictloc, order,ManhatunDis):
    #cansenddict -> {vehicle: {seq:[  ( (t, 'cansend'),(preloc) ),   ]} }
    cansenddict = {}
    for vehicle in predictloc:
        cansenddict[vehicle] = {}
        for seq in predictloc[vehicle]:
            # if vehicle == '6962':
            #     print(seq,predictloc[vehicle][seq])
            #     print(seq,data[vehicle][seq])
            cansenddict[vehicle][seq] = []
            # i = min(order, len(predictloc[vehicle][seq])-1)#10  9
            i = 0
            while i < len(predictloc[vehicle][seq]):
                # if manhdistance(AveList(predictloc[vehicle][seq][:i+1]), data[vehicle][seq][i][1])<=0:
                if manhdistance(predictloc[vehicle][seq][i][1], data[vehicle][seq][i][1]) <= ManhatunDis:
                    if i + int(order/2) < len(predictloc[vehicle][seq]):
                        if manhdistance(predictloc[vehicle][seq][int(i + order/2)][1], data[vehicle][seq][int(i+order/2)][1])<=manhdistance(predictloc[vehicle][seq][i][1], data[vehicle][seq][i][1]):
                        # if manhdistance(AveList([predictloc[vehicle][seq][i], predictloc[vehicle][seq][i + int(order/2)]]),
                        #                     data[vehicle][seq][int(i + order / 2)][1]) <= manhdistance(predictloc[vehicle][seq][i][1], data[vehicle][seq][i][1]):
                            cansenddict[vehicle][seq].append(
                                ((predictloc[vehicle][seq][int(i + order/2)][0][0], 'cansend'), predictloc[vehicle][seq][int(i + order/2)][1]))
                            break
                        #     if i + int(order/2)+1 < len(predictloc[vehicle][seq]) and manhdistance(predictloc[vehicle][seq][int(i + order / 2)+1][1],
                        #                     data[vehicle][seq][int(i + order / 2+1)][1]) <= \
                        #             manhdistance(predictloc[vehicle][seq][int(i + order/2)][1], data[vehicle][seq][int(i + order/2)][1]):
                        #         cansenddict[vehicle][seq].append(
                        #             ((predictloc[vehicle][seq][int(i + order/2)][0][0], 'cansend'), predictloc[vehicle][seq][int(i + order/2)][1]))
                        #         break
                    # else:
                    #     if i < i+ int(order/4) < len(predictloc[vehicle][seq]):
                    #         if manhdistance(predictloc[vehicle][seq][int(i + order / 4)][1],
                    #                         data[vehicle][seq][int(i + order / 4)][1]) <= manhdistance(
                    #                 predictloc[vehicle][seq][i][1], data[vehicle][seq][i][1]):
                    #             # if manhdistance(AveList(predictloc[vehicle][seq][:int(i + order / 2)]),
                    #             #                     data[vehicle][seq][int(i + order / 2)][1]) <= 0:
                    #             cansenddict[vehicle][seq].append(
                    #                 ((predictloc[vehicle][seq][int(i + order / 4)][0][0], 'cansend'),
                    #                  predictloc[vehicle][seq][int(i + order / 4)][1]))
                    #             break

                    # cansenddict[vehicle][seq].append(
                    #             ((predictloc[vehicle][seq][i][0][0], 'cansend'), predictloc[vehicle][seq][i][1]))
                    i += order
                    continue
                else:
                    i += order

            # i = 1
            # while i < len(predictloc[vehicle][seq]):
            #     # if predictloc[vehicle][seq][i][0][1] == 'ori':
            #     #     i = i+1
            #     #     continue
            #     tag = 0
            #     templist = []
            #
            #     for j in range(i):
            #         if data[vehicle][seq][i][1] != data[vehicle][seq][j][1] and \
            #                         manhdistance(predictloc[vehicle][seq][i][1],
            #                                      predictloc[vehicle][seq][j][1]) <= 2 and \
            #                                                 i- j == 1:
            #             for x in range(j,i+1):
            #                 if data[vehicle][seq][x][1] not in templist:
            #                     templist.append(data[vehicle][seq][x][1])
            #             if len(templist) > order:
            #                 cansenddict[vehicle][seq].append( ((predictloc[vehicle][seq][i][0][0],'cansend'), predictloc[vehicle][seq][i][1]) )
            #                 tag = 1
            #                 break
            #     if tag == 0 and len(cansenddict[vehicle][seq]) > 0:
            #         cansenddict[vehicle][seq].append( ((predictloc[vehicle][seq][i][0][0], 'cansend'), cansenddict[vehicle][seq][-1][1]) )
            #     i = i+1
    return cansenddict


def cansend(data, predictloc, order):
    #cansenddict -> {vehicle: {seq:[  ( (t, 'cansend'),(preloc) ),   ]} }
    cansenddict = {}
    for vehicle in predictloc:
        cansenddict[vehicle] = {}
        for seq in predictloc[vehicle]:
            # if vehicle == '6962':
            #     print(seq,predictloc[vehicle][seq])
            #     print(seq,data[vehicle][seq])
            cansenddict[vehicle][seq] = []
            i = 1
            while i < len(predictloc[vehicle][seq]):
                # if predictloc[vehicle][seq][i][0][1] == 'ori':
                #     i = i+1
                #     continue
                tag = 0
                templist = []

                for j in range(i):
                    if data[vehicle][seq][i][1] != data[vehicle][seq][j][1] and \
                                    manhdistance(predictloc[vehicle][seq][i][1],
                                                 predictloc[vehicle][seq][j][1]) <= 2 and \
                                                            i- j == 1:
                        for x in range(j,i+1):
                            if data[vehicle][seq][x][1] not in templist:
                                templist.append(data[vehicle][seq][x][1])
                        if len(templist) > order:
                            cansenddict[vehicle][seq].append( ((predictloc[vehicle][seq][i][0][0],'cansend'), predictloc[vehicle][seq][i][1]) )
                            tag = 1
                            break
                if tag == 0 and len(cansenddict[vehicle][seq]) > 0:
                    cansenddict[vehicle][seq].append( ((predictloc[vehicle][seq][i][0][0], 'cansend'), cansenddict[vehicle][seq][-1][1]) )
                i = i+1
    # for vehicle in cansenddict:
    #     for seq in cansenddict[vehicle]:
    #         if len(cansenddict[vehicle][seq]) == 0:
    #             cansenddict[vehicle][seq].append( ((predictloc[vehicle][seq][min(order-1, len(predictloc[vehicle][seq])-1)][0][0], 'cansend'), (predictloc[vehicle][seq][min(order-1, len(predictloc[vehicle][seq])-1)][1]))  )

                #############################################################################
            # i = 0
            # while i < len(predictloc[vehicle][seq]):
            #     if predictloc[vehicle][seq][i][0][1] == 'ori':
            #         # print(predictloc[vehicle][seq][i][0][1])
            #         i += 1
            #         continue
            #     if i+deltat <= len(predictloc[vehicle][seq])-1 and \
            #                     manhdistance(predictloc[vehicle][seq][i][1],predictloc[vehicle][seq][i+deltat][1])<=2 and \
            #                     data[vehicle][seq][i][1] != data[vehicle][seq][i+deltat][1]:
            #         # for j in range(deltat+1):
            #         cansenddict[vehicle][seq].append( ((predictloc[vehicle][seq][i+deltat][0][0],'cansend'), predictloc[vehicle][seq][i][1]) )
            #         # while
            #         i += deltat
            #         while i < len(predictloc[vehicle][seq])-1:
            #             if data[vehicle][seq][i][1] == data[vehicle][seq][i+1][1]:
            #                 cansenddict[vehicle][seq].append(((predictloc[vehicle][seq][i + 1][0][0], 'cansend'),
            #                                                   predictloc[vehicle][seq][i][1]))
            #                 i += 1
            #             else:
            #                 break
            #         continue
            #     i += 1
            #############################################################################
            # if len(cansenddict[vehicle][seq])==0:
            #     cansenddict[vehicle][seq].append(
            #         ((predictloc[vehicle][seq][-1][0][0], 'cansend'), predictloc[vehicle][seq][-1][1]))
    return cansenddict

def manhdistance(x, y):
    return abs(x[0]-y[0])+ abs(x[1]-y[1])


def sendprocess(originaldata, datatloc, bandwidth, cansenddict, seqtdict, sendresult,TF):
    # print(seqtdict['6962'])
    # statusdict/datatloc->{t: {loc :[v1,v2], loc: [v3,v4]}
    statusdict = copy.deepcopy(datatloc)#vehicles who can send messages
    # t_min = min(datatloc.keys())
    # t_max = max(datatloc.keys())
    for t in sorted(datatloc):
        for loc in datatloc[t]:
            # if t == 10627 and loc ==(33,12):
            #     print(t, loc, statusdict[t][loc])
            for vehicle in datatloc[t][loc]:
                # print(vehicle)
                # print(cansenddict[vehicle][seqtdict[vehicle][t]])
                # index = [temp for temp in cansenddict[vehicle][seqtdict[vehicle][t]]]
                tlist = []
                if len(cansenddict[vehicle][seqtdict[vehicle][t]]) > 0:
                    for item in cansenddict[vehicle][seqtdict[vehicle][t]]:
                        tlist.append(item[0][0])
                    if t not in tlist:
                        # if vehicle == '6962' and seqtdict[vehicle][t] == '8-4':
                        #     print(t, loc,statusdict[t][loc])
                        statusdict[t][loc].remove(vehicle)
                        # if vehicle == '6962' and seqtdict[vehicle][t] == '8-4':
                        #     print(t, statusdict[t][loc])

                else:
                    statusdict[t][loc].remove(vehicle)
                # if vehicle == '6962' and seqtdict[vehicle][t] == '8-4':
                #     print(tlist)
    # print('test')
    # print(statusdict[10564][(36, 24)])
    # # [((10568, 'cansend'), (36.0, 25.0))]
    for t in sorted(statusdict.keys()):
        for loc in statusdict[t]:
            if len(statusdict[t][loc])<= bandwidth:
                for vehicle in statusdict[t][loc]:
                    if len(sendresult[vehicle][seqtdict[vehicle][t]]) == 0:
                        sendresult[vehicle][seqtdict[vehicle][t]].append((t, loc))
            else:
                competedict = {}
                # competedict = collections.OrderedDict()
                for vehicle in statusdict[t][loc]:
                    # if vehicle == '6962' :
                        # print("4",t, loc,statusdict[t][loc])
                    if len(sendresult[vehicle][seqtdict[vehicle][t]]) == 0:
                        templist = cansenddict[vehicle][seqtdict[vehicle][t]]
                        index = 0
                        for item in templist:
                            if item[0][0] == t:
                                index = templist.index(item)
                                break
                        try:
                            competedict[vehicle] = [loc, cansenddict[vehicle][seqtdict[vehicle][t]][index][1]]
                            competedict[vehicle].append(manhdistance(competedict[vehicle][0], competedict[vehicle][1]))
                        except:
                            print('Com list index out of range')
                competevehicle = sorted(competedict, key=lambda k: competedict[k][2], reverse=TF)
                for vehicle in competevehicle[:bandwidth]:
                    sendresult[vehicle][seqtdict[vehicle][t]].append((t, loc))
                for vehicle in competevehicle[bandwidth:]:
                    try:
                        index = originaldata[vehicle][seqtdict[vehicle][t]].index((t, loc))
                        if index < len(originaldata[vehicle][seqtdict[vehicle][t]])-1:
                            if statusdict[originaldata[vehicle][seqtdict[vehicle][t]][index+1][0]][originaldata[vehicle][seqtdict[vehicle][t]][index+1][1]].__contains__(vehicle):
                                continue
                            else:
                                statusdict[originaldata[vehicle][seqtdict[vehicle][t]][index+1][0]][originaldata[vehicle][seqtdict[vehicle][t]][index+1][1]].append(vehicle)
                                cansenddict[vehicle][seqtdict[vehicle][t]].append(((originaldata[vehicle][seqtdict[vehicle][t]][index+1][0],'cansend'),competedict[vehicle][1]))
                    except:
                        print(t, loc, vehicle, seqtdict[vehicle][t])
                    # else:
    # print(sendresult['6962'])

    return sendresult

def ManhatunDis(List1,List2):
    return abs(List1[0]-List2[0])+abs(List1[1]-List2[1])
def Testcansend(Dict,predictloc):
    cansenddict = {}
    # Break=0
    for vehicle in predictloc.keys():
        cansenddict[vehicle] = {}
        # Break+=1
        for seq in predictloc[vehicle].keys():
            cansenddict[vehicle][seq] = []
            # if Break>20:
            #     break
            # print(Dict[vehicle][seq])
            Count=0
            for i in range(1,len(predictloc[vehicle][seq])):
                For = (predictloc[vehicle][seq][i-1][1], tuple( Dict[vehicle][seq][i-1][1] ))
                if tuple(Dict[vehicle][seq][i][1])!=For[1]:
                    Count += 1
                    # print(tuple( Dict[vehicle][seq][i][1] ) , For[1])
                    Back=(predictloc[vehicle][seq][i][1],tuple(Dict[vehicle][seq][i][1]))
                    if ManhatunDis(Back[1],Back[0])!=0 and Count>18:
                        if ManhatunDis(For[0],Back[0])/ManhatunDis(Back[1],Back[0])==0:
                            # print(For,Back,ManhatunDis(For[0],Back[0])/ManhatunDis(Back[1],Back[0]))
                            cansenddict[vehicle][seq].append(((predictloc[vehicle][seq][i][0][0], 'cansend'), predictloc[vehicle][seq][i][1]) )
                # if ManhatunDis(predictloc[vehicle][seq][i][1],predictloc[vehicle][seq][i-1][1])

                # print(predictloc[vehicle][seq][i],Dict[vehicle][seq][i],ManhatunDis(predictloc[vehicle][seq][i][1],Dict[vehicle][seq][i][1]))
            # print(seq,predictloc[vehicle][seq])
            # print(seq,Dict[vehicle][seq])
            # print(((predictloc[vehicle][seq][0][0][0],'cansend'), predictloc[vehicle][seq][0][1]))
    return cansenddict