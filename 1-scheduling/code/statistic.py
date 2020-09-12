import com,math

def profile(desloc, preloc, mintotalvehicle,maxtotalvehicle, neighbor ):
    # if totalvehicle == 0:
    #     return 0
    # print(neighbor)
    # if neighbor == 0:
    #     return  0
    # if neighbor> 500:
    #     return 0
    if com.manhdistance(desloc, preloc)>2:
        # print(com.manhdistance(desloc, preloc))
        return 0
    else:
        # print(neighbor, totalvehicle)
        # print(1.0-2.0*com.manhdistance(desloc,preloc)/6.0)
        # print(mintotalvehicle,maxtotalvehicle, neighbor)
        # return (1.0-2.0*com.manhdistance(desloc,preloc)/6.0)*(1.0*(neighbor-mintotalvehicle)/(500-mintotalvehicle))
        # return (1.0-2.0*com.manhdistance(desloc,preloc)/6.0)*(1.0*(neighbor))
        # return (1.0-2.0*com.manhdistance(desloc,preloc)/6.0)
        # return (1.0*(neighbor-mintotalvehicle))#/(maxtotalvehicle-mintotalvehicle))
        return (1.0-2.0*com.manhdistance(desloc,preloc)/6.0)*math.log(neighbor+math.e)
def statistic(originaldict, tlocdict, cansenddict, sendresultdict):
    # print('cansenddict')
    # print(cansenddict)
    # print('sendresultdict')
    # print(sendresultdict)
    returnprofile = 0
    totalprofile1 = 0
    totalprofile06 = 0
    totalprofile03 = 0
    totalprofile0 = 0
    # totalvehicles = originaldict.__len__()
    maxtotalvehicles = 0
    mintotalvehicles = 10000
    totalmessage = 0

    for vehicle in sendresultdict:
        for seq in sendresultdict[vehicle]:
            totalneighborset = set()
            # totalneighbornum = 0
            if len(sendresultdict[vehicle][seq]) == 0:
                # print('not send')
                continue
            for item in originaldict[vehicle][seq]:
                # except dumplicate
                totalneighborset = totalneighborset.union(tlocdict[item[0]][item[1]])
                #include dumplicate
                # totalneighbornum = totalneighbornum + len(tlocdict[item[0]][item[1]])
            totalneighborset.remove(vehicle)
            mintotalvehicles = min(totalneighborset.__len__(),mintotalvehicles)
            # mintotalvehicles = min(totalneighbornum, mintotalvehicles)
            # maxtotalvehicles = max(totalneighbornum, maxtotalvehicles)

    for vehicle in sendresultdict:
        for seq in sendresultdict[vehicle]:
            # totalvehicles = 0

            # currentloc = sendresultdict[vehicle][seq][0][1]
            # print(sendresultdict[vehicle][seq])
            # print(vehicle +'  '+ seq)
            # print(sendresultdict[vehicle][seq])
            if len(sendresultdict[vehicle][seq]) == 0:
                # print('not send')
                continue
            t = sendresultdict[vehicle][seq][0][0]
            # print(t)
            # for item in cansenddict[vehicle][seq]:
            #     # if item[0][0] == t:
            #     print(vehicle, seq,item)
            # print(cansenddict[vehicle][seq])
            # print(sendresultdict[vehicle][seq])
            try:
                preloc = [item[1] for item in cansenddict[vehicle][seq] if item[0][0] == t][0]
            except:
                print('List index out of range')
            desloc = originaldict[vehicle][seq][-1][1]
            trace = [temp for temp in originaldict[vehicle][seq] if temp[0] > t]
            neighborset = set()
            for item in trace:
                neighborset = neighborset.union(tlocdict[item[0]][item[1]])

            # neighborset.remove(vehicle)
            neighbornum = neighborset.__len__()
            # neighbornum = 0
            # for item in trace:
            #     neighbornum += len(tlocdict[item[0]][item[1]])
            # print(profile(desloc, preloc, totalvehicles, neighbornum))
            tempprofile = profile(desloc, preloc, mintotalvehicles, maxtotalvehicles, neighbornum)
            returnprofile += tempprofile
            # if tempprofile >0 :
            totalmessage = totalmessage + 1

            #
            # if returnprofile ==1:
            #     totalprofile1 += 1
            # elif 1 > returnprofile > 0.6:
            #     totalprofile06 += 1
            # elif 0.4 > returnprofile > 0.3:
            #     totalprofile03 += 1
            # else:
            #     totalprofile0 += 1
    rate = returnprofile/totalmessage
    # print(returnprofile,totalmessage,rate)
    return returnprofile,totalmessage,rate
    # return 1.0*totalprofile1/totalmessage, 1.0*totalprofile06/totalmessage, 1.0*totalprofile03/totalmessage, 1.0*totalprofile0/totalmessage
    # return 1.0*totalprofile1, 1.0*totalprofile06, 1.0*totalprofile03, 1.0*totalprofile0


