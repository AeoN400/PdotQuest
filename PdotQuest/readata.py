#~
import pandas as pd
import numpy as np


def types(s):
    t = []
    s=list(s)
    for i in range(len(s)):
        if s[i] not in t:
            t.append(s[i])
        else:
            continue
    #print(len(t))
    return t

# def tess_pytransit_bin():
#     rb=np.loadtxt('tess_pytransit_bin.txt', delimiter=',')
#     if  rb.shape == (3,):
#         bep=np.array([int(rb[0])])
#         bt=np.array([rb[1]])
#         ber=np.array([rb[2]])
#     else:
#         bep=np.array(rb[:,0]).astype(int)
#         bt=np.array(rb[:,1])
#         ber=np.array(rb[:,2])
#     return bep,bt,ber

class Read_Data:
    def __init__(self,n):
        self.n = n
    
    def ref_data(self,print_ref = True):
        rd = pd.read_csv('ref_data.csv', header=0)
        ndex=[]
        self.print_req = print_ref
        
        for i in range(len(rd['System'])):
            if rd['System'][i] == self.n:
                ndex.append(i)
        ndex=np.array(ndex)
        if len(ndex) > 0 and print_ref:
            print(types(rd['Reference'][ndex]))
            
        if len(ndex) > 0:
            ep = np.array(rd['Orbit number'][ndex])
            te = np.array(rd['T_mid'][ndex])
            er = np.array(rd['Uncertainty (days)'][ndex])
        else:
            ep = None
            te = None
            er = None
            print('None')
        return ep,te,er
    
#     def ref_data(self,req,print_ref = True):
#         a = pd.read_csv('lit_data.csv', header=0)
#         ndex=[]
#         self.print_req = print_ref
        
#         if req == 'ref':
#             for i in range(len(a['System'])):
#                 if a['System'][i] == self.n and a['Reference'][i] != 'This work':
#                     ndex.append(i)
#             ndex=np.array(ndex)
#             if len(ndex) > 0 and print_ref:
#                 print(types(a['Reference'][ndex]))
#         elif req == 'winn_tess':
#             for i in range(len(a['System'])):
#                 if a['System'][i] == self.n and a['Reference'][i] == 'This work':
#                     ndex.append(i)
#             ndex=np.array(ndex)

#         if len(ndex) > 0:
#             ep = np.array(a['Orbit number'][ndex])
#             te = np.array(a['T_mid'][ndex])
#             er = np.array(a['Uncertainty (days)'][ndex])
#         else:
#             ep = None
#             te = None
#             er = None
#             print('None')
#         return ep,te,er
    
    def tess_pytransit(self,ref = True):
        rt=np.loadtxt('tess_pytransit.txt', delimiter=',')
        ep=np.array(rt[:,0]).astype(int)
        te=np.array(rt[:,1])
        er=np.array(rt[:,2])
        self.ref = ref
        if ref:
            repoch,rtime,rerror = self.ref_data('ref')
            ep = np.append(repoch,ep)
            te = np.append(rtime,te)
            er = np.append(rerror,er)
        return ep,te,er
    
#     def winn(self):
#         repoch,rtime,rerror = self.ref_data('ref',print_ref = False)
#         wepoch,wtime,werror = self.ref_data('winn_tess')
#         ep = np.append(repoch,wepoch)
#         te = np.append(rtime,wtime)
#         er = np.append(rerror,werror)
#         return ep,te,er
    
    
    