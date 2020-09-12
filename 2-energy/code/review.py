import os,time
import Function as F
F.rate_energy=0.5
path='./File/'
F.Model_path='./File/Model/'
file=path+'Trajectory.pkl'
window=10
test_day=8
time_start=time.time()
print('start generating training data...')
Trajectory=F.Trajectory=F.DataModule().load_pkl(file)
Original=F.DataModule().load_pkl(path+'Original_res.pkl')
Prediction=F.DataModule().load_pkl(path+'Prediction_res.pkl')
stand=F.DataModule().load_pkl(path+'Tu_Veh.pkl')
F.Query=F.DataModule().load_pkl(path+'Query.pkl')
print('total vehicles in day 8:',Trajectory[8].__len__()) ## total vehicles in day 8 :6538
##### statistic the total vehicles
# count=0
# for key in Original:
#     # print(key,Original[key])
#     for k in Original[key]:
#         # print(k,Original[key][k])
#         count+=sum(Original[key][k])*0.75
# print(count) # total vehicles: 859296
print(859296*F.rate_energy)
##### end
##### s--the number of vehicles can send
# Can_send=F.Correspondence().allocation(Original,Prediction)
# count=0
# for key in Can_send:
#     for k in Can_send[key]:
#         count+=sum(Can_send[key][k])
#         # print(Can_send[key])
# print('can send:',count) # when 0.75: 598768
##### e
##### s  direct
F.Correspondence().direct(Original,stand)
### sent messages: 636571 sent vehicles: 6378
### under condition: sent messages: 598769 sent vehicles: 6177
##### e

##### s
# Can_send=F.Correspondence().allocation(Original,Prediction)
# F.Correspondence().strategy(Original,Can_send,test_day,stand)

##### e