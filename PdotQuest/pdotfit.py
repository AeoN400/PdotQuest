#~
import pandas as pd
import numpy as np
import os
import emcee
import corner
import matplotlib.pyplot as plt
import random
from uncertainties import ufloat
from astropy.timeseries import LombScargle
from PdotQuest.readata import tess_pytransit_bin

def set_epoch(te,p):
    e=[]
    for i in range(len(te)):
        e.append(round((te[i]-te[0])/p))
    e=np.array(e)
    return e

def plot_resi(ep,te,er):
    c, ccov = np.polyfit(ep, te, deg=1, w=1/er, cov=True)
    tlin = np.poly1d(c)(ep)
    multi = 24*60
    fig, ax = plt.subplots(figsize=(13,4), constrained_layout=True)
    ax.errorbar(ep, multi*(te-tlin), multi*er, fmt='.')
    ax.axhline(0, c='k', lw=1, ls='--')
    plt.setp(ax, xlabel='Epoch', ylabel='Timing residuals [min]');
    return fig

def xi2(a,b,e):
    x=[]
    for i in range(len(a)):
        xi = (a[i]-b[i])**2/e[i]**2
        x.append(xi)
    x=np.array(x)
    x2 = x.sum()
    return x2

class LinearFit:
    def __init__(self,times,errors,period,rej_sigma=3,rej_mode='sad'):
        self.times = times
        self.errors = errors
        self.period = period
        self.epps = set_epoch(self.times,self.period)
        self.time0 = times[0]
        self.p_est = [self.period - 0.0002, self.period + 0.0002]
        self.t0_est = [self.time0 - 0.05, self.time0 + 0.05]
        self.rej_sigma=rej_sigma
        self.rej_mode=rej_mode # 'sad'-standard absolute deviation,'mad'-median absolute deviation
        
    def functp0(self,theta):
        m=24*60*60*1000*365
        nl=self.epps[-1]//1000*1000+1000
        p,t0 = theta
        t=[]
        for i in range(0,nl):
            t_n=t0+i*p
            t.append(t_n)
        return t,p

    def log_prior0(self,theta):
        p,t0 = theta
        if (self.p_est[0] < p < self.p_est[1] and self.t0_est[0] < t0 < self.t0_est[1]): 
            return 0.0
        return -np.inf

    def log_likelihood0(self,theta,n,te,err):
        tn,p=self.functp0(theta)
        tn=np.array(tn,dtype=object)

        x=[]
        for j in range(len(te)):
            xi=(te[j]-tn[n[j]])**2/err[j]**2
            x.append(xi)
        x=np.array(x,dtype=object)
        return -0.5 * np.sum(x)
    
    def log_probability0(self,theta,n,te,err):
        lp = self.log_prior0(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood0(theta,n,te,err)

    def MCMC0(self,ep,te,er):
        N = 100
        pos1 = np.random.uniform(self.p_est[0], self.p_est[1], N)
        pos2 = np.random.uniform(self.t0_est[0], self.t0_est[1], N)
        pos = np.vstack([pos1,pos2]).T
        nwalkers, ndim = pos.shape
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability0,args=(ep,te,er)
        )
        sampler.run_mcmc(pos, 500, progress=True)
        N_burn = 200
        samples = sampler.chain[:, N_burn:, :].reshape((-1, ndim))

        ###
        self.p0,self.t0_0 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
        print('p = %f'%self.p0[0], 't0 = %f'%self.t0_0[0])
    
    def tn_fit0(self,ep):
        tn,pn=self.functp0((self.p0[0],self.t0_0[0]))
        tn0=[]
        for i in range(len(ep)):
            tn0.append(tn[ep[i]])
        tn0=np.array(tn0)
        return tn0
    
    def filt_l_0(self,tn0):
        residuals0 = self.times-tn0
        if self.rej_mode == 'sad':
            s0=self.rej_sigma*np.abs(np.std(residuals0))
        if self.rej_mode == 'mad':
            s0=self.rej_sigma*np.abs(np.median(residuals0))
        count = 0
        fl=[]
        for i in range(len(tn0)):
#             fl.append(abs(residuals0[i]) < s0 and (self.errors[i] < s0 or self.errors[i]< 0.002))
#             if abs(residuals0[i]) > s0 or (self.errors[i] > s0 and self.errors[i] >= 0.002):
            fl.append(abs(residuals0[i]) < s0)
            if abs(residuals0[i]) > s0:
                count += 1
        return s0,fl,count

    def filt_l_n(self,tn0,fl):
        residuals = self.times-tn0
        if self.rej_mode == 'sad':
            sn=self.rej_sigma*np.abs(np.std(residuals[fl]))
        if self.rej_mode == 'mad':
            sn=self.rej_sigma*np.abs(np.median(residuals[fl]))
        count = 0
        fln=[]
        for i in range(len(tn0)):
#             fln.append(abs(residuals[i]) < sn and self.errors[i] < sn)
#             if abs(residuals[i]) > sn or self.errors[i] > sn:
            fln.append(abs(residuals[i]) < sn)
            if abs(residuals[i]) > sn:
                count += 1
        return sn,fln,count

    def filt0(self,tn0,fl):
        ttime = self.times[fl]
        errrr = self.errors[fl]
        eeeep = self.epps[fl]
        tnn = tn0[fl]
        return eeeep,ttime,errrr,tnn
    
    def run_MCMC0(self):
        self.MCMC0(self.epps,self.times,self.errors)
        tnf0 = self.tn_fit0(self.epps)
        darray0 = self.filt0(tnf0,self.filt_l_0(tnf0)[1])
        print(self.filt_l_0(tnf0)[2])
        elim0=[]
        elim0.append(self.filt_l_0(tnf0)[2])
        flist0=[]
        flist0.append(self.filt_l_0(tnf0)[1])

        while self.filt_l_0(tnf0)[2] != 0:
            self.MCMC0(darray0[0],darray0[1],darray0[2])
            tnf0 = self.tn_fit0(self.epps)
            self.sn0,fln0,count0 = self.filt_l_n(tnf0,flist0[-1])
            flist0.append(fln0)
            print(count0)
            elim0.append(count0)
            self.f_epp0,self.f_time0,self.f_error0,self.f_tn0l=self.filt0(tnf0,flist0[-1])
            if elim0[-1] == elim0[-2]:
                break

        if self.filt_l_0(tnf0)[2] == 0:
            self.sn0,fln0,count0 = self.filt_l_0(tnf0)
            self.f_epp0,self.f_time0,self.f_error0,self.f_tn0l = self.epps,self.times,self.errors,tnf0

        ba0=np.array(fln0)
        al0 = ~ba0
        self.a_epp0,self.a_time0,self.a_error0,self.a_tn0l=self.filt0(tnf0,al0)
    
    def plot_residuals0(self):
        multi = 24*60*60
        plt.rcParams['figure.figsize'] = [13,4]
        plt.errorbar(self.f_epp0,(self.f_time0-self.f_tn0l)*multi, self.f_error0*multi, fmt='.')
        plt.errorbar(self.a_epp0,(self.a_time0-self.a_tn0l)*multi, self.a_error0*multi, fmt='.',color='silver',alpha=0.5)
        plt.axhline(0, c='k', lw=1, ls='--')
        plt.fill_between(self.f_epp0, (self.sn0)*multi,(-self.sn0)*multi, alpha=0.2)

        plt.ylabel("Timing residuals [s]",fontsize=14)
        plt.xlabel("Epoch",fontsize=14)
        plt.show()
        
        
    def BIC0(self):
        return xi2(self.f_time0,self.f_tn0l,self.f_error0)+2*np.log(len(self.f_tn0l))
    
    def plot_LombScargle(self):
        ls=LombScargle(self.f_epp0,self.f_time0-self.f_tn0l,self.f_error0)
        frequency, power = ls.autopower()
        print(power.max())
        #probabilities = [0.1, 0.05, 0.01]
        fap=ls.false_alarm_level(0.01)
        plt.rcParams['figure.figsize'] = [13,4]
        plt.plot(frequency, power)
        plt.axhline(fap, c='k', lw=1, ls='--',label='FAP=1%')
        plt.ylabel("Lomb-Scargle Power",fontsize=14)
        #plt.xlabel("period [days]",fontsize=14)
        plt.xlabel("Frequency [days]",fontsize=14)
        plt.legend(loc='upper right')
        plt.show()
    
class QuadraFit():
    def __init__(self,times,errors,period,a_est=[-100,0],rej_sigma=3,rej_mode='sad'):
        self.times = times
        self.errors = errors
        self.period = period
        self.epps = set_epoch(self.times,self.period)
        self.time0 = times[0]
        self.p0_est = [self.period - 0.0002, self.period + 0.0002]
        self.t0_est = [self.time0 - 0.05, self.time0 + 0.05]
        self.a_est = a_est
        self.rej_sigma=rej_sigma
        self.rej_mode=rej_mode
        
    def functp(self,theta):
        m=24*60*60*1000*365
        nl=self.epps[-1]//1000*1000+1000
        t=[np.nan]*nl
        p=[np.nan]*nl
        a,t0,p0 = theta
        t[0]=t0
        p[0]=p0
        for j in range(1,nl):
            t[j]=t[j-1]+p[j-1]
            p[j]=(1+a/(2*m))/(1-a/(2*m))*p[j-1]# bo
        return t,p

    def log_likelihood(self,theta,n,te,err):
        tn,pn=self.functp(theta)
        tn=np.array(tn,dtype=object)

        x=[]
        for j in range(len(te)):
            xi=(te[j]-tn[n[j]])**2/err[j]**2
            x.append(xi)
        x=np.array(x,dtype=object)
        return -0.5 * np.sum(x)

    def log_prior(self,theta):
        a,t0,p0 = theta
        if (self.a_est[0] < a < self.a_est[1] 
            and self.t0_est[0] < t0 < self.t0_est[1] 
            and self.p0_est[0] < p0 < self.p0_est[1]): 
            return 0.0
        return -np.inf

    def log_probability(self,theta,n,te,err):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta,n,te,err)
    
    def MCMC(self,ep,tt,er):
        N = 100
        pos1 = np.random.uniform(self.a_est[0], self.a_est[1], N)
        pos2 = np.random.uniform(self.t0_est[0], self.t0_est[1], N)
        pos3 = np.random.uniform(self.p0_est[0], self.p0_est[1], N)
        pos = np.vstack([pos1,pos2,pos3]).T
        nwalkers, ndim = pos.shape
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,args=(ep,tt,er)
        )
        sampler.run_mcmc(pos, 500, progress=True)
        N_burn = 200
        self.samples = sampler.chain[:, N_burn:, :].reshape((-1, ndim))

        ###
        self.a0,self.t0_0,self.p0_0=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(self.samples, [16, 50, 84],
                                                    axis=0)))
        print('a = %f'%self.a0[0], 't0 = %f'%self.t0_0[0], 'p0 = %f'%self.p0_0[0])

    def tn_fit(self,ep):
        tn,pn = self.functp((self.a0[0],self.t0_0[0],self.p0_0[0]))
        tn0=[]
        for i in range(len(ep)):
            tn0.append(tn[ep[i]])
        tn0=np.array(tn0)
        return tn0
    
    def filt_bl_0(self,tn0):
        if self.rej_mode == 'sad':
            s0=self.rej_sigma*np.abs(np.std(self.times-tn0))
        if self.rej_mode == 'mad':
            s0=self.rej_sigma*np.abs(np.median(self.times-tn0))
        count = 0
        bl=[]
        for i in range(len(tn0)):
#             bl.append(abs(self.times[i]-tn0[i]) < s0 and (self.errors[i] < s0 or self.errors[i]< 0.002))
#             if abs(self.times[i]-tn0[i]) > s0 or (self.errors[i] > s0 and self.errors[i]>= 0.002):
            bl.append(abs(self.times[i]-tn0[i]) < s0)
            if (abs(self.times[i]-tn0[i]) > s0):
                count += 1
        return s0,bl,count

    def filt_bl_n(self,tn0,bl):
        if self.rej_mode == 'sad':
            sn=self.rej_sigma*np.abs(np.std(self.times[bl]-tn0[bl]))
        if self.rej_mode == 'mad':
            sn=self.rej_sigma*np.abs(np.median(self.times[bl]-tn0[bl]))
        count = 0
        bln=[]
        for i in range(len(tn0)):
#             bln.append(abs(self.times[i]-tn0[i]) < sn and self.errors[i] < sn)
#             if abs(abs(self.times[i]-tn0[i])) > sn or self.errors[i] > sn:
            bln.append(abs(self.times[i]-tn0[i]) < sn)
            if abs(abs(self.times[i]-tn0[i])) > sn:
                count += 1
        return sn,bln,count

    def filt(self,tn0,bl):
        ttime = self.times[bl]
        errrr = self.errors[bl]
        eeeep = self.epps[bl]
        tnn = tn0[bl]
        #tnn2 = f_tn0l[bl]
        #return eeeep,ttime,errrr,tnn,tnn2
        return eeeep,ttime,errrr,tnn
    
    def run_MCMC(self):
        self.MCMC(self.epps,self.times,self.errors)
        tnf = self.tn_fit(self.epps)
        sbc0 = self.filt_bl_0(tnf)
        print(sbc0[2])
        darray = self.filt(tnf,sbc0[1])
        elim=[]
        elim.append(sbc0[2])
        blist=[]
        blist.append(sbc0[1])
        while sbc0[2] != 0:
            self.result = self.MCMC(darray[0],darray[1],darray[2])
            tnf = self.tn_fit(self.epps)
            self.sbc= self.filt_bl_n(tnf,blist[-1])
            blist.append(self.sbc[1])
            print(self.sbc[2])
            elim.append(self.sbc[2])
            self.f_epp,self.f_time,self.f_error,self.f_tn0=self.filt(tnf,blist[-1])
            if elim[-1] == elim[-2] or elim[-1]>15:
                break

        if sbc0[2] == 0:
            self.sbc = sbc0
            self.f_epp,self.f_time,self.f_error,self.f_tn0=self.epps,self.times,self.errors,tnf
        
        self.lin = LinearFit(self.times,self.errors,self.period)
        self.lin.MCMC0(self.epps,self.times,self.errors)
        self.f_tn0l1 = self.lin.tn_fit0(self.epps)[self.sbc[1]]
        ###
        ba=np.array(self.sbc[1])
        al = ~ba
        self.a_epp,self.a_time,self.a_error,self.a_tn0=self.filt(tnf,al)
        self.a_tn0l1 = self.lin.tn_fit0(self.epps)[al]
        ###
        
    def plot_residuals(self,emax=2000,add_resi = True,tess = False,add_bin = False,unitt = 's'):
        self.emax = self.epps[-1]//100*100+100
        self.add_resi = add_resi
        self.tess = tess
        self.add_bin = add_bin
        self.unitt = unitt
        if self.unitt == 's':
            multi = 24*60*60
        elif self.unitt == 'min':
            multi = 24*60
        elif self.unitt == 'h':
            multi = 24
        
        tn,pn=self.functp((self.a0[0],self.t0_0[0],self.p0_0[0]))
        tnl,pl=self.lin.functp0((self.lin.p0[0],self.lin.t0_0[0]))
        tn = np.array(tn)
        tnl = np.array(tnl)
        
        if add_bin == True:
            bepoch,btime,berror = tess_pytransit_bin()
            bepp=bepoch-bepoch[0]+int(np.round((btime[0]-self.t0_0[0])/self.period))
            if bepp[-1]>self.epps[-1]:
                self.emax = int(bepp[-1]//100*100+100)
            
        plt.rcParams['figure.figsize'] = [20,4]
        plt.plot(list(range(0,self.emax)),(tn[0:self.emax]-tnl[0:self.emax])*multi)
        
        if tess == True:
            rt=np.loadtxt('tess_pytransit.txt', delimiter=',')
            etime=np.array(rt[:,1])
            tf=[]
            for i in range(len(self.f_time)):
                tf.append(self.f_time[i] in etime)
            tf=np.array(tf)
        else:
            tf = np.zeros(len(self.f_time), dtype=bool)
        
        if add_resi == True:
            if tess == False:
                plt.errorbar(self.f_epp, multi*(self.f_time-self.f_tn0l1), multi*self.f_error, fmt='.')
            else:
                plt.errorbar(self.f_epp[~tf], multi*(self.f_time[~tf]-self.f_tn0l1[~tf]), multi*self.f_error[~tf], fmt='.')
                plt.errorbar(self.f_epp[tf], multi*(self.f_time[tf]-self.f_tn0l1[tf]), multi*self.f_error[tf], fmt='.')
            plt.errorbar(self.a_epp, multi*(self.a_time-self.a_tn0l1), multi*self.a_error, fmt='.',color='silver',alpha=0.5)
            self.top=tn[0:self.emax]-tnl[0:self.emax]+self.sbc[0]
            self.down=tn[0:self.emax]-tnl[0:self.emax]-self.sbc[0]
            plt.fill_between(list(range(0,self.emax)), self.top*multi,self.down*multi, alpha=0.2)
            
        if add_bin == True:
            btnl=tnl[bepp]
            plt.errorbar(bepp, multi*(btime-btnl), multi*berror, fmt='o')
                
        plt.axhline(0, c='k', lw=1, ls='--')
        plt.ylabel("Timing residuals [" + self.unitt + "]",fontsize=14)
        plt.xlabel("Epoch",fontsize=14)
        plt.show()
    
    def download_oc(self,txtname='O-C'):
        self.txtname=txtname

        rt=np.loadtxt('tess_pytransit.txt', delimiter=',')
        etime=np.array(rt[:,1])

        tn,pn=self.functp((self.a0[0],self.t0_0[0],self.p0_0[0]))
        tnl,pl=self.lin.functp0((self.lin.p0[0],self.lin.t0_0[0]))
        tn = np.array(tn)
        tnl = np.array(tnl)
        
        tf=[]
        for i in range(len(self.f_time)):
            tf.append(self.f_time[i] in etime)
        tf=np.array(tf)
        
        folder = os.path.exists(self.txtname) 
        if not folder:
            os.makedirs(self.txtname) 
            
        f = open(self.txtname +'//'+ 'simu.txt', 'w')
        for i in range(0,self.emax):
            f.write(str(list(range(0,self.emax))[i])+','+str((tn[0:self.emax]-tnl[0:self.emax])[i])+'\n')
        f.close()
        f = open(self.txtname + '//'+ 'ref.txt', 'w')
        for i in range(len(self.f_epp[~tf])):
            f.write(str(self.f_epp[~tf][i])+','+str(self.f_time[~tf][i]-self.f_tn0l1[~tf][i])+','+str(self.f_error[~tf][i])+'\n')
        f.close()
        f = open(self.txtname + '//'+ 'tess.txt', 'w')
        for i in range(len(self.f_epp[tf])):
            f.write(str(self.f_epp[tf][i])+','+str(self.f_time[tf][i]-self.f_tn0l1[tf][i])+','+str(self.f_error[tf][i])+'\n')
        f.close()
        f = open(self.txtname + '//'+ 'elim.txt', 'w')
        for i in range(len(self.a_epp)):
            f.write(str(self.a_epp[i])+','+str(self.a_time[i]-self.a_tn0l1[i])+','+str(self.a_error[i])+'\n')
        f.close()
        f = open(self.txtname + '//'+ 'td.txt', 'w')
        for i in range(len(self.top)):
            f.write(str(list(range(0,self.emax))[i])+','+str(self.top[i])+','+str(self.down[i])+'\n')
        f.close()
        
    def qorner(self):
        fig = corner.corner(self.samples, labels=["$a$", "$t0$","$p0$"],
                      truths=[self.a0[0], self.t0_0[0],self.p0_0[0]])

    def BICq(self):
        return xi2(self.f_time,self.f_tn0,self.f_error)+3*np.log(len(self.f_tn0))
    
    def print_posterior(self):
        ae = max(self.a0[1],self.a0[2])
        t = ufloat(self.t0_0[0],max(self.t0_0[1],self.t0_0[2]))
        p = ufloat(self.p0_0[0],max(self.p0_0[1],self.p0_0[2]))
        print('轨道周期变化率为 %.3f'% self.a0[0],'+/- %.3f'% ae,'ms/yr')
        print('拟合的t0为',t,'days')
        print('设定的t0为',self.times[0])
        print('拟合的p0为',p,'days')
        print('设定的p0为',self.period)

def mock_lin_data(time,er,ep,p0,a):
    m=24*60*60*1000*365
    nl=ep[-1]+1
    
    c, ccov = np.polyfit(set_epoch(time,p0), time, deg=1, w=1/er, cov=True)
    
    t=[np.nan]*nl
    p=[np.nan]*nl
    t[0]=time[0]
    p[0]=c[0]
    
    for j in range(1,nl):
        t[j]=t[j-1]+p[j-1]
        p[j]=(1+a/(2*m))/(1-a/(2*m))*p[j-1]
    t=np.array(t)
    t=t[ep]
    
    return t
    
def mockdata(time, er, p0, a, yr, sigma1, n, ex=False, en=0):
    ep = set_epoch(time, p0)
    nep = np.linspace(ep[-1], round(ep[-1] + yr * 365 / p0), n + 1)[1:]
    nep = np.round(nep).astype(int)  # new epoch
    print(nep)
    
    if ex:  # 如果ex为True
        enep = []  # expand epoch
        for num in nep:
            enep.extend(range(num, num + en))
        print(enep)
        epp = np.concatenate((ep, enep))
    else:
        epp = np.concatenate((ep, nep))
    
    tm = mock_lin_data(time, er, epp, p0, a)
    
    mu1 = 0
    mu2 = sigma1
    sigma2 = sigma1 / 10
    em = np.zeros(len(epp))
    
    for i in range(len(tm)):
        tm[i] += random.gauss(mu1, sigma1)
        em[i] += abs(random.gauss(mu2, sigma2))
    
    return tm, em
