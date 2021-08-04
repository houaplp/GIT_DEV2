
# coding: utf-8

# In[1]:


import os
import math
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
plt.rcParams["svg.fonttype"]="none"


# In[2]:


folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Stage_Roche/codes/Txt01"
#folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/00_tests"
#folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/PopPlinn/class/PopPlinnTxtSel2"
#folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/compRock"
#folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/_Entrance/txt"


filelist = []
for path, dirs, files in os.walk(folder_path):
    for filename in files:
        if 'txt' in filename :
                filelist.append(filename)
filelist.sort()
print(len(filelist))

#get classes in string
class_path=os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Stage_Roche/codes/Txt01/")
#class_path=os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/00_tests/")
#class_path=os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/PopPlinn/class/PopPlinnTxtSel2/")
#class_path=os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/compRock/")
#class_path=os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/_Entrance/txt/")



# In[3]:


def f_to_midi (f) :
    return 69+12*math.log(f/440,2)

print ("note de freq 261Hz :" , f_to_midi (261))

def midi_to_f (n) :
    return 440*2**((n-69)/12)

print ("frequence de note 60 :" , midi_to_f (60), "Hz")

def dx2x(intervalles):
    res=[0]
    for i in range (len (intervalles)):
        res.append(res[i]+intervalles[i-1])
    return res
               
def a2db (a):
    if(a<=0.000001):res = -120
    else:
        res = 20*math.log(a,10)
    return res

# version limitée à un ambitus 0 (0.001) - 60dB (1)               
def a2db60 (a):
    if(a<=0.001):res = 0
    else:res = 60+20*math.log(a,10)
    return res

print ("amp = 0.0 => amplitude en dB :" , round(a2db(0.000001), 3), "dB")
print ("amp = 0.0001 => amplitude en dB :" , round(a2db(0.0001), 3), "dB")
print ("amp = 0.5 => amplitude en dB :" , round(a2db(0.5), 3), "dB")
print ("amp = 1 => amplitude en dB :" , a2db(1), "dB")
print ("amp60 = 0 => amplitude60 en dB :" , a2db60(0), "dB")
print ("amp60 = 1 => amplitude60 en dB :" , a2db60(1), "dB")


# In[69]:



f_ls = [21.5,32.3,43.1,53.8,64.6,86.1,107.7,140.0,172.3,215.3,269.2,344.5,441.4,549.1,699.8,872.1,1109.0,1388.9,1755.0,2217.9,2788.5,3520.7,4435.8,5587.9,7041.4,8871.7,11175.7,14071.9]

L = [] # liste des tailles des lignes
for k in range(27):
    L.append(4+2*k)

f_c = [] # frequences centrales des bandes
for i in range(len(f_ls)-1):
    f_c.append(math.sqrt(f_ls[i+1]*f_ls[i]))
    
f_c_midi = []
for i in range(len(f_c)):
    f_c_midi.append(f_to_midi(f_c[i]))

f_c_moy = 0 
for i in range(len(f_c)):
    f_c_moy += f_c[i]
    f_c_moy /= 27
    
f_c_gmoy = 0
for i in range(len(f_c)):
    f_c_gmoy += math.log(f_c[i], 2)
    f_c_gmoyR =  2**(f_c_gmoy/27)

#formule equivalente
#f_c_gmoy2 = 0
#for i in range(len(f_c)):
#    f_c_gmoy2 += f_to_midi(f_c[i])
#    f_c_gmoyR2 =  midi_to_f(f_c_gmoy2/27)

    
print("len(f_ls) :", len(f_ls), "bornes")
W=[]
for i in range(len(f_ls)-1):
    W.append(round(f_ls[i+1]-f_ls[i], 2))
print("w :" , W)    
    
print ("f_c :", f_c)
print ("f_c_moy :", round(f_c_moy,1), "Hz") # (en Hz) moyenne des frequences des centres des bandes 
print ("f_c_gmoyR :", round(f_c_gmoyR,1), "Hz") # (en Hz) moyenne des centres calculée par les notes MIDI
#print ("f_c_gmoyR :", f_c_gmoyR2) # (en Hz) moyenne des centres calculée par les notes MIDI

f_cA = np.asarray(f_c)
f_cA = f_cA[:, np.newaxis]
#print (round(3.149 , 2))


# In[70]:


# read filetxt and generate array S
def read(filetext):
    with open(filetext) as f:
        mylist = f.read().splitlines()
        for x in range(8):
            mylist.pop(0)
        S=[]
        for element in reversed(mylist):
            element2=[float(i) for i in element.split()]
            S.append(element2)
        return S
    


# In[71]:


file1 = filelist[0]
S1 = read(file1)
file2 = filelist[1]
S2 = read(file2)

print (file1)
print (file2)


# In[72]:


# plus file name unite comme secondes 
def timetxt (filetext):
    str_L = filetext.rsplit(sep='_')
    test = 0
    result = 0
    for str in str_L:
        if test == 3:
            cent = int(str)
            result+=(cent/100)
            test = 0
        if test == 2:
            sec = int(str)
            test = 3
            result+=sec
        if re.search('deb', str):
            test = 1
            min = int(str[4-5])
            result+=(min*60)
            test = 2
    return result

print(timetxt("EntranceEICcreation_m.wav_sr44100_deb00_00_00_t02_50_pas10_00.txt"))
#timetxt("AlanStivell_PopPlinn_1971m.wav_sr44100_deb00_45_00_t02_50_pas02_50.txt")


# In[73]:


# moyennes des amplitudes par ligne (des fréquences basses vers hautes)
def moy_des_amps (S):
    Sk_mean=[]
    for k in range(27):
        sum_s=0
        for j in range(4+2*k):
            sum_s+=S[k][j]
        Sk_mean.append(sum_s/(4+2*k))
    return Sk_mean  
print("amps:",moy_des_amps(S1))


# In[74]:


#centroid
def centroid(S):
    Sk_mean = moy_des_amps(S)
    sum_sc=0
    sum_sfk=0
    for k in range(27):
        sum_sfk+=Sk_mean[k]*f_c[k]
        sum_sc+=Sk_mean[k]            
    if sum_sc == 0 :
        centroid=0
    else:
        centroid=sum_sfk/sum_sc
    #print('centroid' , centroid)
    return centroid

print("centroid:",round(centroid(S1)),"Hz")


# In[75]:


#variance et sd
def variance(S):
    Sk_mean=[]
    variance = 0
    for k in range(27):
        sum_s=0
        for j in range(4+2*k):
            sum_s+=S[k][j]
        Sk_mean.append(sum_s/(4+2*k))
    sum_var=0
    sum_skm=0
    for k in range(27):
        sum_var+=Sk_mean[k]*((f_c[k]-centroid(S))**2)
        sum_skm+=Sk_mean[k]
    if sum_skm>0:
        variance=sum_var/sum_skm
        #print("variance", variance)
    return variance


#print("variance:",round(variance(S1)))


# sd: 
def sd(filetext):
    sd=math.sqrt(variance(filetext))
    return sd

print("sd:",round(sd(S1)),"Hz")


# In[76]:


# sd: biased standard deviation (N)
def sdLow(S):
    Sk_mean = moy_des_amps(S)
    sdLow = 0
    sum_sd=0
    sum_skm=0
    for k in range(27):
        diff=f_c[k]-centroid(S)
        if diff <0:
            sum_sd+=Sk_mean[k]*(diff**2)
            sum_skm+=Sk_mean[k]
    if sum_skm > 0:
        sdLow=math.sqrt(sum_sd/sum_skm)
    return sdLow

# sd: biased standard deviation (N)
def sdHigh(S):
    Sk_mean = moy_des_amps(S)
    sdHigh = 0
    sum_sd=0
    sum_skm=0
    for k in range(27):
        diff=f_c[k]-centroid(S)
        if diff >0:
            sum_sd+=Sk_mean[k]*(diff**2)
            sum_skm+=Sk_mean[k]
    if sum_skm > 0:
        sdHigh=math.sqrt(sum_sd/sum_skm)
    return sdHigh
    
def c_min_sdlow(S): 
    return round(centroid(S) - sdLow(S),1)

def c_plus_sdHi(S): 
    return round(centroid(S) + sdHigh(S),1)


print("centroid:", round(centroid(S1),1),"Hz")
print("centre-ecartlow:", c_min_sdlow(S1),"Hz")
print("centre+ecartHigh:", c_plus_sdHi(S1),"Hz")


# In[77]:


#MIDI centroid
def midi_centroid(S):
    Sk_mean = moy_des_amps(S)
    sum_sc=0
    midi_centroid=0
    sum_sfk=0
    for k in range(27):
        sum_sfk+=Sk_mean[k]*f_to_midi(f_c[k])
        sum_sc+=Sk_mean[k]            
    if sum_sc > 0 :
        midi_centroid=sum_sfk/sum_sc
    #print('centroid' , centroid)
    return midi_centroid


# In[78]:


#ecartHigh
def midi_ecartHigh(S): 
    Sk_mean = moy_des_amps(S) 
    sum_var=0 
    sum_skm=0
    varianceHigh = 0
    for k in range(27): 
        df = f_to_midi(f_c[k])-midi_centroid(S) 
        if df>0:
            sum_var+=Sk_mean[k]*(df**2)
            sum_skm+=Sk_mean[k]
    if sum_skm>0:
        varianceHigh=sum_var/sum_skm
    #print(\"variance\", variance)
    return math.sqrt(varianceHigh)

#ecartLow 
def midi_ecartLow(S): 
    Sk_mean = moy_des_amps(S) 
    sum_var=0 
    sum_skm=0
    varianceLow = 0
    for k in range(27): 
        df = f_to_midi(f_c[k])-midi_centroid(S) 
        if df<0:
            sum_var+=Sk_mean[k]*(df**2)
            sum_skm+=Sk_mean[k]
    if sum_skm>0:
        varianceLow=sum_var/sum_skm
    #print(\"variance\", variance)
    return math.sqrt(varianceLow)

def mc_min_eclow(S): 
    return round(midi_centroid(S) - midi_ecartLow(S),1)

def mc_plus_ecHi(S): 
    return round(midi_centroid(S) + midi_ecartHigh(S),1)

mcd = round(midi_centroid(S1),1)
mcd_eL = round(mc_min_eclow(S1),1)
mcd_eH = round(mc_plus_ecHi(S1),1)
print("midi_centroid:", mcd,"MIDI /", round(midi_to_f(mcd)),"Hz")
print("centr-ecartLow:", mcd_eL,"MIDI /", round(midi_to_f(mcd_eL),1),"Hz") 
print("centr+ecartHigh:", mcd_eH,"MIDI /", round(midi_to_f(mcd_eH),1),"Hz")


# In[79]:


#midi spectral slope
def midi_spectral_slope_reg(S):
    f_csA_midi = np.asarray(f_c_midi)
    f_csA_midi = f_csA_midi[:, np.newaxis]
    Sk_meanA = np.asarray(moy_des_amps(S))
    Sk_meanA = Sk_meanA[:, np.newaxis]
    modeleReg = LinearRegression()
    modeleReg.fit(f_csA_midi, Sk_meanA)
    #plt.scatter(f_csA, Sk_meanA)
    return modeleReg

def midi_sp_slope (S):
    return midi_spectral_slope_reg(S).coef_[0][0]

def plot_slope(S):
    modeleReg = midi_spectral_slope_reg(S)
    f_csA_midi = np.asarray(f_c_midi)
    f_csA_midi = f_csA_midi[:, np.newaxis]
    Sk_meanA = np.asarray(moy_des_amps(S))
    Sk_meanA = Sk_meanA[:, np.newaxis]
    plt.scatter(f_csA_midi, Sk_meanA)
    
    x_min = 10
    x_max = 130

    X_NEW = np.linspace(x_min, x_max, 100)
    X_NEW = X_NEW[:,np.newaxis]

    Y_NEW = modeleReg.predict(X_NEW)

    plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(-0.2, 0.2)

    plt.title("regression linéaire test", fontsize=10)
    plt.xlabel('f_cs_midi')
    plt.xlabel('a_ls_midi')

    plt.savefig("simple_linear_regression_test_midi.png", bbox_inches='tight')
    plt.show()

plot_slope (S1)
print ("slope : " + str(midi_sp_slope(S1)))



# In[80]:


#spectral slope
def spectral_slope_reg(S):
    f_csA = np.asarray(f_c)
    f_csA = f_csA[:, np.newaxis]
    Sk_meanA = np.asarray(moy_des_amps(S))
    Sk_meanA = Sk_meanA[:, np.newaxis]
    modeleReg = LinearRegression()
    modeleReg.fit(f_csA, Sk_meanA)
    #plt.scatter(f_csA, Sk_meanA)
    return modeleReg

def sp_slope (S):
    return spectral_slope_reg(S).coef_[0][0]

def plot_slope(S):
    modeleReg = spectral_slope_reg(S)
    f_csA = np.asarray(f_c)
    f_csA = f_csA[:, np.newaxis]
    Sk_meanA = np.asarray(moy_des_amps(S))
    Sk_meanA = Sk_meanA[:, np.newaxis]
    plt.scatter(f_csA, Sk_meanA)
    
    x_min = 20
    x_max = 14000

    X_NEW = np.linspace(x_min, x_max, 100)
    X_NEW = X_NEW[:,np.newaxis]

    Y_NEW = modeleReg.predict(X_NEW)

    plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
    #plt.scale.LogScale
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.txaxis=dict(type='log', autorange=True)
    plt.ylim(-0.2, 0.2)

    plt.title("regression linéaire test", fontsize=10)
    plt.xlabel('f_cs')
    plt.xlabel('a_ls')
    plt.xscale("log")

    plt.savefig("simple_linear_regression_test.png", bbox_inches='tight')
    plt.show()

plot_slope (S1)
print ("slope : " + str(sp_slope(S1)))



# In[81]:


#spectral flatness 
def sp_flatness(S):
    Sk_mean = moy_des_amps(S)
    sum_skm=0
    mult_skm=1
    K=26
    for k in range(26):
        sum_skm+=Sk_mean[k]
        mult_skm*=max(Sk_mean[k], 0.00001)
        pow_skm=mult_skm**(1/K)
    if sum_skm == 0 :
        sf=0
    else:
        sf=pow_skm/((1/K)*sum_skm)
    return sf

print ("spectral_flatness:", sp_flatness(S1))


# In[82]:


# moyennes des amplitudes par ligne (des fréquences basses vers hautes)
def produit_des_amps (S):
    Sk_mult=1
    for k in range(27):
        mult_s=1
        for j in range(4+2*k):
            mult_s*=max(S[k][j]**(0.5/(k+2)), 0.00000001)
        Sk_mult*=mult_s
    return Sk_mult  

# moyennes des amplitudes par ligne (des fréquences basses vers hautes)
def flatness_list (S):
    Sk_flat=[]
    for k in range(27):
        mult_s=1
        for j in range(4+2*k):
            mult_s*=max(S[k][j]**(0.5/(k+2)), 0.00000001)
        Sk_flat.append(mult_s)
    return Sk_flat  

#spectral flatness moyenne
def sp_flatness_moy(S):
    Sk_flat = flatness_list(S)
    Sm=0
    for k in range(27):
         Sm+=Sk_flat[k]
    #print(Sm)
    return Sm/27

#formule equivalente
def sp_flatness_moy2(S):
    Sk_flat = flatness_list(S)
    Sk_flatA = np.asarray(Sk_flat)
    return Sk_flatA.mean()

def sp_flatness_max(S):
    Sk_flat = flatness_list(S)
    Sm=0
    for k in range(27):
        if Sm<Sk_flat[k]:
            Sm=Sk_flat[k]
            value = k
    return [value, f_c[value], Sm]
         
    #print(Sm)
    return Sm/27

def sp_flatness_min(S):
    Sk_flat = flatness_list(S)
    Sm=1
    for k in range(27):
        if Sm>Sk_flat[k]:
            Sm=Sk_flat[k]
            value = k
    return [value, f_c[value], Sm]

def sp_flatness_maxamp(S):
    return sp_flatness_max(S)[2]
def sp_flatness_maxfreq(S):
    return sp_flatness_max(S)[1]
def sp_flatness_minamp(S):
    return sp_flatness_min(S)[2]
def sp_flatness_minfreq(S):
    return sp_flatness_min(S)[1]

print("sp_flatness_maxamp:", sp_flatness_maxamp(S1))
print("sp_flatness_maxfreq:", sp_flatness_maxfreq(S1))
print("sp_flatness_minamp:", sp_flatness_minamp(S1))
print("sp_flatness_minfreq:", sp_flatness_minfreq(S1))


# In[83]:


def sp_crest(S):
    Sk_mean = moy_des_amps(S)
    sum_skm=0
    K=27
    screst = 99 # si tout à zero
    max_skm=[]
    for k in range(27):
        sum_skm+=Sk_mean[k]
        for j in range(4+2*k):
            max_skm.append(np.max(S[k][j]))
    max_sk=max(max_skm)
    if sum_skm > 0:
        screst=max_sk/((1/K)*sum_skm)
    return screst

print(sp_crest(S1))


# In[84]:



# rms with original value of frequency bin k=27
def rms(S):
    Sk_mean = moy_des_amps(S)
    sum_skr=[]
    for k in range(27):
        sum_sk=0
        for j in range(L[k]):
            sum_sk+= (1/L[k])*(S[k][j]**2)
        sum_skr.append(sum_sk)
    #print(sum_skr)
    sum_rms=0
    for k in range(27):
        sum_rms+=sum_skr[k]
    rms=math.sqrt(sum_rms)
    return rms  

print(rms(S1))


# In[85]:


# maxfreq25 (amp max de la bande 25) approx 8000Hz
def maxfreq25(S):
    maxfreq25=max(S[25])
    return maxfreq25


# In[86]:


# skewness (asymétrie du spectre – importance des fréquences graves) skw = 0 => spectre symétrique
def skewness(S):
    Sk_mean = moy_des_amps(S)
    sum_sk=0
    sum_skm=0
    skewness = 0
    for k in range(27):
        sum_sk+=Sk_mean[k]*((f_c[k]-centroid(S))**3)
        sum_skm+=Sk_mean[k]
    if sum_skm > 0:
        skewness=(sum_sk/sum_skm)/(sd(S)**3)
    return skewness

print(skewness(S1))

# kurtosis 
def kurtosis(S):
    Sk_mean = moy_des_amps(S)
    sum_kt=0
    sum_skm=0
    kurtosis = 0
    for k in range(27):
        sum_kt+=Sk_mean[k]*((f_c[k]-centroid(S))**4)
        sum_skm+=Sk_mean[k]
    if sum_skm > 0:
        kurtosis=(sum_kt/sum_skm)/(sd(S)**4)
    return kurtosis

print(kurtosis(S1))


# In[87]:


def sp_rolloff(S):
    Sk_mean = moy_des_amps(S)
    sum_skm=0
    sum_rf=0
    for k in range(27):
        sum_skm+=Sk_mean[k]
    for k in range(27):
        sum_rf+=Sk_mean[k]
        if sum_rf>=0.85*sum_skm:
            rf=f_c[k]
            break
    return rf


# In[88]:


#amps max et min
# max_amp = l'amplitude de la cellule la plus forte
def max_amp(S):
    maxs = []
    for k in range(27):
        maxs.append(max(S[k]))
    return max(maxs)

def max_amoy(S):
    return max(moy_des_amps(S))

def min_amp(S):
    mins = []
    for k in range(27):
        mins.append(min(S[k]))
    return min(mins)
                    
print('max_amp :',round(max_amp(S1),3))
print('max_amoy :',round(max_amoy(S1),3))
print('min_amp :',round(min_amp(S1),5))


# In[89]:


def max_freq(S):
    maxs = []
    maxi = 0
    indx = 0
    for k in range(27):
        maxs.append(max(S[k]))
    maxi = max(maxs)
    indx = maxs.index(maxi)
    #print ("indx :", indx, "amp :", maxs[indx])
    return f_c[indx]

def max_moy_freq(S):
    maxs = moy_des_amps(S)
    maxi = max(maxs)
    indx = maxs.index(maxi)
    #print ("indx :", indx, "amp :", maxs[indx])
    return f_c[indx]


print('max_freq :',round(max_freq(S1),5))
print('max_moy_freq :',round(max_moy_freq(S1),5))


# In[90]:


def mode_freq(S):
    Sk_mean = moy_des_amps(S)
    Sk_sum = np.sum(Sk_mean)
    Sk_cum = 0
    indx1 = 0
    indx2 = 0
    indx3 = 0
    res = 0
    # print ("Sk_sum : ", Sk_sum)
    if Sk_sum > 0 :
        for k in range(27):
            if Sk_cum < 0.75:
                indx3+=1
                Sk_cum+=(Sk_mean[k]/Sk_sum)
            if Sk_cum < 0.5:
                indx2+=1
            if Sk_cum < 0.25:
                indx1+=1
        #print (indx)
        res = [f_ls[indx1-1], f_ls[indx2-1], f_ls[indx3-1]]
    return [f_ls[indx1-1], f_ls[indx2-1], f_ls[indx3-1]]


def mode_freq1(S):    
    return mode_freq(S)[0]
    
def mode_freq2(S):    
    return mode_freq(S)[1]
    
def mode_freq3(S):    
    return mode_freq(S)[2]
        

print('mode_freq :',mode_freq(S1))
print('mode :',mode_freq2(S1))


# In[91]:
# In[92]:


def nbre_cell_sup_ampmoy(S):
    maxamp = max_amp(S)
    minamp = min_amp(S)
    moyamp = (maxamp + minamp)* 0.8
    n = 0
    for k in range(27):
        for j in range (len(S[k])):
            if (S[k][j] >= moyamp):
                n += 1/(k+2)
    return n

nbre_cell_sup_ampmoy(S1)

def nbre_cell_sup_ampmoy2(S):
    maxamp = max_amp(S)
    minamp = min_amp(S)
    moyamp = (maxamp + minamp)* 0.5
    #moyamp = 0.05
    n = 0
    for k in range(10):
        for j in range (len(S[k])):
            if (S[17+k][j] >= moyamp):
                n += 1/(k+19)
    return n

def nbre_cell_sup_ampmoy3(S, nlow, nhigh):
    n = 0
    delta = 1+nhigh - nlow
    for k in range(delta):
        k2 = nlow+k
        maxk = np.max(S[k2])
        #print(maxk)
        #print(S[k2])
        for j in range (len(S[k2])):
            if (S[k2][j] > maxk*0.25):
                n += 1/(k2+2)
    return round(n*50/delta, 4)

def nbre_cell_sup_ampmoy4(S, nlow, nhigh):
    n = 0
    delta = 1+nhigh - nlow
    maxamp = max_amp(S)
    minamp = min_amp(S)
    moyamp = (maxamp + minamp)* 0.3333
    for k in range(delta):
        k2 = nlow+k
        for j in range (len(S[k2])):
            if (S[k2][j] > moyamp):
                n += 1/(k2+2)
    return round(n*50/delta, 4)

print ("ampmoy", nbre_cell_sup_ampmoy(S1))
print ("ampmoy2", nbre_cell_sup_ampmoy2(S1))
print ("ampmoy3", nbre_cell_sup_ampmoy3(S1, 0, 0))
print ("ampmoy4", nbre_cell_sup_ampmoy4(S1, 0, 4))


# In[93]:


def arrayscale (Array, maxout):
    maxin = (np.max(Array))
    res =  np.multiply(Array, maxout/maxin)
    return np.round_(res, 3)


res=[]
for fichier in filelist:
    S = read(fichier)
    res.append(nbre_cell_sup_ampmoy4(S, 0, 26))
X = np.arange(len(res))
plt.plot(X, res, 'ro')
plt.grid()
#axeX = np.arange(11)
#plt.xticks(arrayscale(axeX,len(res)-1))
    
plt.xlim(-0.2, len(res)+0.2)
plt.ylim(-1, max(res)+1)
plt.title("densité", fontsize=10)
plt.show()


# In[94]:


res=[]

def cell_sup_ampmoy(S):
    return nbre_cell_sup_ampmoy4(S, 0, 26)

for fichier in filelist:
    S = read(fichier)
    res.append(cell_sup_ampmoy(S))
X = np.arange(len(res))

plt.bar(X, res, align='edge', color='blue')
plt.xlim(-1, len(res)+1)
plt.ylim(0, max(res)+1)
plt.title("density 0-26", fontsize=10)
#plt.show()

dest = "entrance_nbre_cell_sup_ampmoy4.pdf"
   
plt.savefig(dest)


# In[95]:


# densité sur zone de frequences variables
res=[]


def highcell_sup_ampmoy(S):
    return nbre_cell_sup_ampmoy4(S, 18, 24)

for fichier in filelist:
    S = read(fichier)
    res.append(highcell_sup_ampmoy(S))
X = np.arange(len(res))

plt.bar(X, res, align='edge', color='blue')
plt.xlim(-1, len(res)+1)
plt.ylim(0, max(res)+1)
plt.title("density 18-24", fontsize=10)
#plt.show()

dest = "entrance_nbre_cell_sup_ampmoy4_18-24.pdf"
   
plt.savefig(dest)


# In[96]:


# densité sur zone de frequences variables
res=[]


def highcell_sup_ampmoy(S):
    return nbre_cell_sup_ampmoy4(S, 18, 24)

for fichier in filelist:
    S = read(fichier)
    res.append(highcell_sup_ampmoy(S))
X = np.arange(len(res))

plt.bar(X, res, align='edge', color='blue')
plt.xlim(-1, len(res)+1)
plt.ylim(0, max(res)+1)
plt.title("density 18-24", fontsize=10)
#plt.show()

dest = "entrance_nbre_cell_sup_ampmoy4_18-24.pdf"
   
plt.savefig(dest)


# In[97]:


centroids = []
times = []
flatnesses = []

print(filelist[0])
print(sp_flatness(read(filelist[0])))


# In[98]:


def spectralEnv(S):
    res = []
    for k in range(27):
        res.append(np.sum(S[k])/(k+2))
    return res

#spectralEnv(S1)


# In[102]:


#------------liste d'intervales-------------------------#
# Renvoie une liste d'intervales pour une taille donnée
def liste_intervales(nb, taille):
    taille_intervale = int(taille / nb)
    intervals = [0] * nb
    for i in range(0, nb):
        intervals[i] = taille_intervale
    ecart = taille - nb * taille_intervale
    milieu = int(nb / 2)
    i = 0
    while (ecart > 0):
        intervals[milieu + i] = intervals[milieu + i] + 1
        if(i <= 0):
            i = -i + 1
        else:
            i = -i
        ecart -= 1
    return intervals

def doubleXvals(list):
    res=[0]
    j = len(list)
    for i in range (j-2):
        res.append(list[i+1])
        res.append(list[i+1])
    res.append(list[j-1])
    return res

def doubleYvals(list):
    res=[]
    j = len(list)
    for i in range (j):
        res.append(list[i])
        res.append(list[i])
    return res

#res1 = spectralEnv(S1)

def plothistoenv(amps):
    nb_dx = len(amps)
    intervales = liste_intervales(nb_dx, 27)
    x = doubleXvals(dx2x(intervales))
    y = doubleYvals(amps)
    #print ("x", x)
    #print ("y", y)
    plt.plot(x, y, color="blue", linewidth=0.5)
    plt.fill_between(x, 0, y, color='blue', alpha=.25)



# In[106]:


# nbre de band d'amplitude moyenne sup à max des Ampmoy des bande
def nbandSupMed(S):
    env = spectralEnv(S)
    ampMedian = np.max(env)/2
    res = 0
    for k in range (27):
        if (env[k] > ampMedian):
            res+=1
    return res

#tests
filetest = filelist[1]
print (filetest)
plothistoenv(spectralEnv(read(filetest)))

print (nbandSupMed(read(filetest)))

# graphique
res=[]

for fichier in filelist:
    S = read(fichier)
    res.append(nbandSupMed(S))
               
X = np.arange(len(res))

plt.bar(X, res, align='edge', color='blue')
plt.xlim(-1, len(res)+1)
plt.ylim(0, max(res)+1)
plt.title("nbandSupMed", fontsize=10)
#plt.show()

dest = "entrance_nbandSupMed.pdf"
   
plt.savefig(dest)


# In[107]:


# nbre de band d'amplitude moyenne sup à max des Ampmoy des bande
def nbandInfMed(S):
    env = spectralEnv(S)
    ampMedian = np.max(env)/2
    res = 0
    for k in range (27):
        if (env[k] < ampMedian):
            res+=1
    return res

#tests
filetest = filelist[1]
print (filetest)
plothistoenv(spectralEnv(read(filetest)))

print (nbandInfMed(read(filetest)))

# graphique
res=[]

for fichier in filelist:
    S = read(fichier)
    res.append(nbandInfMed(S))
               
X = np.arange(len(res))

plt.bar(X, res, align='edge', color='blue')
plt.xlim(-1, len(res)+1)
plt.ylim(0, max(res)+1)
plt.title("nbanInfMed", fontsize=10)
#plt.show()

dest = "entrance_nbandSupMed.pdf"
   
plt.savefig(dest)


# In[126]:


# variabilité temporelle sur une bande
def variatemps1 (bande):
    var = 0
    n = len(bande)
    for j in range(n-1):
        var += abs(bande[j]-bande[j+1])
    return var/n

#print (variatemp1([0,1, 2, -3]))
                


# In[129]:


# variabilité temporelle
def variatemps (S):
    res = 0
    for k in range(27):
        res += variatemps1(S[k])
    return 100*res/27


#Variatemps(S1)
Simpuls = read("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/RechercheAnalyse/Analyses13bWeb/00_tests/impuls2hz_wav/impuls2hz.wav_sr44100_deb00_00_00_t02_00.txt")
Snoise = read("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/RechercheAnalyse/Analyses13bWeb/00_tests/noise_wav/noise.wav_sr48000_deb00_00_00_t02_00_pas02_00.txt")
Ssinus5000 = read("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/RechercheAnalyse/Analyses13bWeb/00_tests/sinus5000_wav/sinus5000.wav_sr44100_deb00_00_10_t01_90_pas04_00.txt")
Ssinus17000 = read("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/RechercheAnalyse/Analyses13bWeb/00_tests/sinus17000_wav/sinus17000.wav_sr44100_deb00_00_10_t01_90_pas04_00.txt")

print (variatemps(Simpuls), variatemps(Snoise), variatemps(Ssinus17000))
#0.045113558959092664 0.682973430047027 0.0017755551618605636


# In[122]:


class Analyse:
    def __init__(self, filename, time, centroid, sd, c_min_sdlow, c_plus_sdHi, midi_centroid, 
                 mc_min_eclow, mc_plus_ecHi, midi_sp_slope, sp_slope, skewness, kurtosis, 
                 sp_rolloff, sp_flatness, sp_flatness_moy, sp_flatness_minamp, 
                 sp_flatness_minfreq, sp_flatness_maxamp, sp_flatness_maxfreq, 
                 sp_crest, rms, maxfreq25, max_amp, max_amoy, min_amp, max_freq, 
                 max_moy_freq, mode_freq1, mode_freq2, mode_freq3, cell_sup_ampmoy, highcell_sup_ampmoy, nbandSupMed, variatemps):
        self.filename = filename
        self.time=time
        self.centroid=centroid
        self.sd=sd
        self.c_min_sdlow=c_min_sdlow
        self.c_plus_sdHi=c_plus_sdHi
        self.midi_centroid=midi_centroid
        self.mc_min_eclow=mc_min_eclow
        self.mc_plus_ecHi=mc_plus_ecHi
        self.midi_sp_slope=midi_sp_slope
        self.sp_slope=sp_slope
        self.skewness=skewness
        self.kurtosis=kurtosis
        self.sp_rolloff=sp_rolloff
        self.sp_flatness=sp_flatness
        self.sp_flatness_moy=sp_flatness_moy
        self.sp_flatness_minamp=sp_flatness_minamp
        self.sp_flatness_minfreq=sp_flatness_minfreq
        self.sp_flatness_maxamp=sp_flatness_maxamp
        self.sp_flatness_maxfreq=sp_flatness_maxfreq
        self.sp_crest=sp_crest
        self.rms=rms
        self.maxfreq25=maxfreq25
        self.max_amp=max_amp
        self.max_amoy=max_amoy
        self.min_amp=min_amp
        self.max_freq=max_freq
        self.max_moy_freq=max_moy_freq
        self.mode_freq1=mode_freq1
        self.mode_freq2=mode_freq2
        self.mode_freq3=mode_freq3
        self.cell_sup_ampmoy=cell_sup_ampmoy
        self.highcell_sup_ampmoy=highcell_sup_ampmoy
        self.nbandSupMed=nbandSupMed
        self.variatemps=variatemps


# In[124]:


with open('../Xparams_popplin5.txt', 'w') as f:
    for fichier in filelist:
        #print (fichier, file=f)
        #print (centroid(fichier))
        print(fichier + "\t" 
              +str(timetxt(fichier))+"\t"
              +str(centroid(read(fichier)))+"\t"
              +str(sd(read(fichier)))+"\t"
              +str(c_min_sdlow(read(fichier)))+"\t"
              +str(c_plus_sdHi(read(fichier)))+"\t"
              +str(midi_centroid(read(fichier)))+"\t"
              +str(mc_min_eclow(read(fichier)))+"\t"
              +str(mc_plus_ecHi(read(fichier)))+"\t"
              +str(midi_sp_slope(read(fichier)))+"\t"
              +str(sp_slope(read(fichier)))+"\t"
              +str(skewness(read(fichier)))+"\t"
              +str(kurtosis(read(fichier)))+"\t"
              +str(sp_rolloff(read(fichier)))+"\t"
              +str(sp_flatness(read(fichier)))+"\t"
              +str(sp_flatness_moy(read(fichier)))+"\t"
              +str(sp_flatness_minamp(read(fichier)))+"\t"
              +str(sp_flatness_minfreq(read(fichier)))+"\t"
              +str(sp_flatness_maxamp(read(fichier)))+"\t"
              +str(sp_flatness_maxfreq(read(fichier)))+"\t"
              +str(sp_crest(read(fichier)))+"\t"
              +str(rms(read(fichier)))+"\t"
              +str(maxfreq25(read(fichier)))+"\t"
              +str(max_amp(read(fichier)))+"\t"
              +str(max_amoy(read(fichier)))+"\t"
              +str(min_amp(read(fichier)))+"\t"
              +str(max_freq(read(fichier)))+"\t"
              +str(max_moy_freq(read(fichier)))+"\t"
              +str(mode_freq1(read(fichier)))+"\t"
              +str(mode_freq2(read(fichier)))+"\t"
              +str(mode_freq3(read(fichier)))+"\t"
              +str(nbre_cell_sup_ampmoy(read(fichier)))+"\t"
              +str(highcell_sup_ampmoy(read(fichier)))+"\t"
              +str(nbandSupMed(read(fichier)))+"\t"
              +str(variatemps(read(fichier)))
              , file=f)


# In[105]:


centroids = []
times = []
flatnesses = []

def moyenne (list):
    return sum(list)/len(list)

#for metal_file in metal_files
fileR = filelist[0]
print (fileR)
for path, dirs, files in os.walk(folder_path):
    for filename in files:
        if fileR in filename :
            if 'txt' in filename :
                #print(filename)
                centroids.append(centroid(read(filename)))
print("moyenne centroid de l'œuvre complete : " + str(moyenne(centroids)) + " Hz")


# In[60]:


# densité sur zone de frequences variables
res=[]


def highcell_sup_ampmoy(S):
    return nbre_cell_sup_ampmoy4(S, 18, 24)

for fichier in filelist:
    S = read(fichier)
    res.append(highcell_sup_ampmoy(S))
X = np.arange(len(res))

plt.bar(X, res, align='edge', color='blue')
plt.xlim(-1, len(res)+1)
plt.ylim(0, max(res)+1)
plt.title("density 18-24", fontsize=10)
#plt.show()

dest = "entrance_nbre_cell_sup_ampmoy4_18-24.pdf"
   
plt.savefig(dest)

