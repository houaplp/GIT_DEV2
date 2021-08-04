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

folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Python-fev21/SelectedData_2a_BlueMoonofKentucky"
#folder_path="/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Stage_Mahdi/LPAnalyseNEW/Kawa/analyses/txts"
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
print("nombre de fichiers lus : " , len(filelist))
#print(filelist)

#get classes in string
class_path=os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Python-fev21/SelectedData_2a_BlueMoonofKentucky/")


# In[3]:

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

file1 = filelist[2]
S1 = read(file1)
#print ("S1 est : " , S1)
file2 = filelist[3]
S2 = read(file2)

print ("S1 est : " , file1)
print ("S2 est : " , file2)

# In[4]:

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

#print(timetxt("EntranceEICcreation_m.wav_sr44100_deb00_00_00_t02_50_pas10_00.txt"))
#timetxt("AlanStivell_PopPlinn_1971m.wav_sr44100_deb00_45_00_t02_50_pas02_50.txt")


# In[5]:

def f_to_midi (f) :
    return 69+12*math.log(f/440,2)

#print ("note de freq 261Hz :" , f_to_midi (261))

def midi_to_f (n) :
    return 440*2**((n-69)/12)

#print ("frequence de note 60 :" , midi_to_f (60), "Hz")

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

#print ("amp = 0.0 => amplitude en dB :" , round(a2db(0.000001), 3), "dB")
#print ("amp = 0.0001 => amplitude en dB :" , round(a2db(0.0001), 3), "dB")
#print ("amp = 0.5 => amplitude en dB :" , round(a2db(0.5), 3), "dB")
#print ("amp = 1 => amplitude en dB :" , a2db(1), "dB")
#print ("amp60 = 0 => amplitude60 en dB :" , a2db60(0), "dB")
#print ("amp60 = 1 => amplitude60 en dB :" , a2db60(1), "dB")

# In[6]:

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

#print("len(f_ls) :", len(f_ls), "bornes")

W=[]

for i in range(len(f_ls)-1):
    W.append(round(f_ls[i+1]-f_ls[i], 2))
    
#print("w :" , W)       
#print ("f_c :", f_c)
#print ("f_c_moy :", round(f_c_moy,1), "Hz") # (en Hz) moyenne des frequences des centres des bandes 
#print ("f_c_gmoyR :", round(f_c_gmoyR,1), "Hz") # (en Hz) moyenne des centres calculée par les notes MIDI
#print ("f_c_gmoyR :", f_c_gmoyR2) # (en Hz) moyenne des centres calculée par les notes MIDI

f_cA = np.asarray(f_c)
f_cA = f_cA[:, np.newaxis]
#print (round(3.149 , 2))


# In[7]:

# moyennes des amplitudes par ligne (des fréquences basses vers hautes)
def moy_des_amps (S):
    Sk_mean=[]
    for k in range(27):
        sum_s=0
        for j in range(4+2*k):
            sum_s+=S[k][j]
        Sk_mean.append(sum_s/(4+2*k))
    return Sk_mean

#print("amps:",moy_des_amps(S1))

# moyennes geometriques des amplitudes par ligne (des fréquences basses vers hautes)
def gmoy_des_amps (S):
    Sk_gmean=[]
    for k in range(27):
        prod_s=1
        for j in range(4+2*k):
            prod_s*=S[k][j]
        Sk_gmean.append(prod_s**(1/(4+2*k)))
    return Sk_gmean

#print("gamps:",gmoy_des_amps(S1))

def produit_des_amps (S):
    Sk_prod=[]
    K = 27
    for k in range(27):
        prod_s=1
        for j in range(4+2*k):
            prod_s*=S[k][j]
        Sk_prod.append(prod_s**(1/(K*(K+3))))
    return Sk_prod

#print("produit amps S2:",produit_des_amps(S2))

# In[8]:

#centroid
def centroid(S):
    Sk_mean = moy_des_amps(S)
    sum_sc=0
    sum_sfk=0
    for k in range(27):
        sum_sfk += Sk_mean[k]*f_c[k]
        sum_sc += Sk_mean[k]            
    if sum_sc == 0 :
        centroid=0
    else:
        centroid=sum_sfk/sum_sc
    #print('centroid' , centroid)
    return centroid

#print("centroid S1 :",round(centroid(S1)),"Hz")

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
    return variance

# sd: 
def sd(filetext):
    sd=math.sqrt(variance(filetext))
    return sd

# In[9]:

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

print("centre-ecartlow S1:", c_min_sdlow(S1),"Hz")
print("centroid S1:", round(centroid(S1),1),"Hz")
print("centre+ecartHigh S1:", c_plus_sdHi(S1),"Hz")

# In[10]:

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


# In[11]:

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
print("centr-ecartLow S1:", mcd_eL,"MIDI /", round(midi_to_f(mcd_eL),1),"Hz") 
print("midi_centroid S1:", mcd,"MIDI /", round(midi_to_f(mcd)),"Hz")
print("centr+ecartHigh S1:", mcd_eH,"MIDI /", round(midi_to_f(mcd_eH),1),"Hz")

# In[12]:

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

    #print("Sk_meanA : " , Sk_meanA)
    
    x_min = 10
    x_max = 130

    X_NEW = np.linspace(x_min, x_max, 100)
    X_NEW = X_NEW[:,np.newaxis]

    Y_NEW = modeleReg.predict(X_NEW)

    plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(0, 0.05)

    plt.title("regression linéaire > midislope", fontsize=10)
    plt.xlabel('f_cs_midi')
    plt.xlabel('a_ls_midi')

    #plt.savefig("simple_linear_regression_test_midi.png", bbox_inches='tight')
    #plt.show()

#plot_slope (S1)
#print ("slope S1: " + str(midi_sp_slope(S1)))


# In[13]:

#spectral flatness 
def sp_flatness_old(S):
    Sk_mean = moy_des_amps(S)
    sum_skm=0
    mult_skm=1
    K=26
    for k in range(26):
        sum_skm+=Sk_mean[k]
        mult_skm*=max(Sk_mean[k], 0.00000001)
        pow_skm=mult_skm**(1/K)
    if sum_skm == 0 :
        sf=0
    else:
        sf=pow_skm/((1/K)*sum_skm)
    return sf

#************ spectral flatness corrigée LP ************
def sp_flatness(S):
    Sk_mean = moy_des_amps(S)
    Sk_prod = produit_des_amps(S)
    sum_skm=0
    mult_skm=1
    K=27
    for k in range(27):
        sum_skm+=Sk_mean[k]
        mult_skm*=max(Sk_prod[k], 0.000000000001)
    if sum_skm == 0 :
        sf=0
    else:
        sf=(K*mult_skm)/sum_skm
    return sf

print ("spectral_flatness S1:", round(100*sp_flatness(S1),0), "%")
print ("spectral_flatness S2:", round(100*sp_flatness(S2),0), "%")

# In[14]:

# moyennes des amplitudes par ligne (des fréquences basses vers hautes)
def flatness_list (S):
    Sk_flat=[]
    Gmoy = gmoy_des_amps (S)
    Sk_mean = moy_des_amps(S)
    for k in range(27):
        Sk_flat.append(Gmoy[k]/Sk_mean[k])
    return Sk_flat

print ("Sk_flat:" , flatness_list(S1))


#********* graphic flatness S1 ***********
def plot_flatness(S):
    #plt.scatter(f_csA_midi, Sk_meanA)

    X = np.arange(0, 27, 1)
    Y = flatness_list(S)

    plt.plot(X, Y, color='coral', linewidth=3)
    #plt.grid()
    #plt.xlim(x_min, x_max)
    plt.ylim(0, 1.0)

    plt.title("flatness par bande", fontsize=10)
    plt.ylabel('flatness')
    plt.xlabel('bandes')

    plt.show()

plot_flatness (S1)
plot_flatness (S2)

#spectral flatness moyenne
def sp_flatness_moy(S):
    Sk_flat = flatness_list(S)
    Sm=0
    for k in range(27):
         Sm+=Sk_flat[k]
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

print("sp_flatness val max par bande S1:", round(100*sp_flatness_maxamp(S1),1), "% & freq de la bande:", round(sp_flatness_maxfreq(S1),1), "Hz")
print("sp_flatness val min par bande S1:", round(100*sp_flatness_minamp(S1),1), "% & freq de la bande:", round(sp_flatness_minfreq(S1),1), "Hz")

# In[15]:

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

print("sp_crest S1:" , sp_crest(S1))
print("sp_crest S2:" , sp_crest(S2))

# In[16]:

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

print("rms S1:", rms(S1))

# In[17]:

# maxfreq25 (amp max de la bande 25) approx     
def maxfreq25(S):
    maxfreq25=max(S[25])
    return maxfreq25

# In[18]:

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
        if sd(S) > 0:
            skewness=(sum_sk/sum_skm)/(sd(S)**3)
    return skewness

print("skewness", skewness(S1))

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
        if sd(S) > 0:
            kurtosis=(sum_kt/sum_skm)/(sd(S)**4)
    return kurtosis

print("kurtosis", kurtosis(S1))

# In[19]:

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

# In[20]:

#*********** TEMPS ************
#******************************

# variabilité temporelle sur une bande
def variatemps1 (bande):
    var = 0
    n = len(bande)
    for j in range(n-1):
        var += abs(bande[j]-bande[j+1])
    return var/n

#print (variatemp1([0,1, 2, -3]))

#……

# In[122]:
#********************** Analyses *********************
#********************** Pour Excel *******************
#*****************************************************

class Analyse:
    def __init__(self, filename, time, midi_centroid, 
                 mc_min_eclow, mc_plus_ecHi, midi_sp_slope, skewness, kurtosis, 
                 sp_rolloff, sp_flatness, sp_crest, rms, maxfreq25):
        self.filename = filename
        #self.time=time
        self.mc_min_eclow=mc_min_eclow
        self.midi_centroid=midi_centroid
        self.mc_plus_ecHi=mc_plus_ecHi
        self.midi_sp_slope=midi_sp_slope
        #self.skewness=skewness
        self.kurtosis=kurtosis
        #self.sp_rolloff=sp_rolloff
        self.sp_flatness=sp_flatness
        self.sp_crest=sp_crest
        #self.rms=rms
        self.maxfreq25=maxfreq25

# In[124]:


with open('../Xparams_bluemoona.txt', 'w') as f:
    for fichier in filelist:
        #print (fichier, file=f)
        #print (centroid(fichier))
        print(fichier + "\t" 
              #+str(timetxt(fichier))+"\t"
              +str(mc_min_eclow(read(fichier)))+"\t"
              +str(midi_centroid(read(fichier)))+"\t"
              +str(mc_plus_ecHi(read(fichier)))+"\t"
              +str(midi_sp_slope(read(fichier)))+"\t"
              #+str(skewness(read(fichier)))+"\t"
              +str(kurtosis(read(fichier)))+"\t"
              #+str(sp_rolloff(read(fichier)))+"\t"
              +str(sp_flatness(read(fichier)))+"\t"
              +str(sp_crest(read(fichier)))+"\t"
              #+str(rms(read(fichier)))+"\t"
              +str(maxfreq25(read(fichier)))
              , file=f)



