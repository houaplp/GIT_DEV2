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

# d'apres http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_ACP_Python.pdf

# In[2]:

print ("lancement Analyse en composantes principales")

# indiquer ici le nom du son (ou du dossier), inclut dans le nom du fichier Excel
name_file = "LedZEPnew_o"
name_file2 = "yes"
# indiquer ici le chemin absolu pour trouver le fichier excel
os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Python-fev21/")

#
pre_file = "Xparams_"
ext_file = ".xlsx"
X = pd.read_excel(pre_file + name_file + ext_file ,sheet_name=0,header=0,index_col=0)
#• Le fichier est un classeur Excel nommé « Xparams_popplinnSel4S.xlsx » ;
#• Les données actives sont situées dans la première feuille (sheet_name = 0) ;
#• La première ligne correspond aux noms des variables (header = 0)
#• La première colonne aux identifiants des observations (index_col = 0).

#• Fichiers destinations
# les positions des points
dest0 = name_file2+ "_pts.txt" 
# le graphe des positions des données
dest1 = name_file2+ "_ACP1.pdf" 
# le graphe des positions  des paramètres
dest2 = name_file2 + "_ACP1pars.pdf"

#taille du graphe (30 pour Sonstsest, 5 pour LedZep, 6 pour autres
scaleXY = 6

# ## Importation des données actives

# In[3]:

#dimension
print(X.shape) # (18, 6)
# nombre d'observations (ATTENTION : doit être supérieur au nombre de variables)
# il faut n > p)
n = X.shape[0]
#nombre de variables
p = X.shape[1]
print("affichage des données")
print( X)
print(n ,  " observations & ", p, " variables")

# ## Préparation des données

# In[4]:
#vérification de la version
print(sklearn.__version__) # 0.19.1

# In[5]:
#classe pour standardisation
from sklearn.preprocessing import StandardScaler
#instanciation
sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(X)
print(Z)

# In[6]:
#vérification - librairie numpy
#moyenne
print("moyenne", np.mean(Z,axis=0))
#écart-type
print("ecart", np.std(Z,axis=0,ddof=0))

# ## Analyse en composantes principales avec PCA de ‘’scikit-learn’’

# In[7]:

# Il faut instancier l’objet PCA dans un premier temps, nous affichons ses propriétés.

#classe pour l'ACP
from sklearn.decomposition import PCA
#instanciation
acp = PCA(svd_solver='full')

#affichage des paramètres
print("acp :" , acp)


# In[8]:


#calculs
coord = acp.fit_transform(Z)
#nombre de composantes calculées
print("ACP_N_Com : ", acp.n_components_) # 6

# In[9]:

# Valeurs propres et scree plot
#variance expliquée
print("ACP explained : ", acp.explained_variance_)

#valeur corrigée
eigval = (n-1)/n*acp.explained_variance_
print("eigval : ", eigval)

#ou bien en passant par les valeurs singulières
print("ACP_Singulr : ", acp.singular_values_**2/n)

# In[10]:

#proportion de variance expliquée
print("ACP Variance ratio : ", acp.explained_variance_ratio_)

# In[11]:

#scree plot
plt.plot(np.arange(1,p+1),eigval) # si p < n il est nécessaire d'avoir plus de lignes que de colonnes

plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()

# In[12]:

#cumul de variance expliquée
plt.plot(np.arange(1,p+1),np.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()

# In[13]:

#seuils pour test des bâtons brisés
bs = 1/np.arange(p,0,-1)
#bs = 1/np.arange(n,0,-1)
bs = np.cumsum(bs)
bs = bs[::-1]

# In[14]:

#test des bâtons brisés
print(pd.DataFrame({'Val.Propre':eigval,'Seuils':bs}))

# In[15]:

#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(scaleXY * -1,scaleXY) #même limites en abscisse
axes.set_ylim(scaleXY * -1,scaleXY) #et en ordonnée

#placement des étiquettes des observations
for i in range(n):
 plt.annotate(X.index[i],(coord[i,0],coord[i,1]), fontsize=6)
 
#ajouter les axes
plt.plot([-10, 10],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[scaleXY * -1,scaleXY],color='silver',linestyle='-',linewidth=1)

#affichage
#plt.show()
#plt.savefig(dest1)

#enregistrement des coordonnées des points
print("Enregistrement du fichier : " + dest0)
f = open(dest0, "w")
#for i in range(n):
#    print (coord[i,0])
#    print (coord[i,1])

for i in range(n):
    f.write (str(X.index[i]))
    f.write (" ")
    f.write (str(coord[i,0]))
    f.write (" ")
    f.write (str(coord[i,1]))
    f.write("\n")
f.close()
   

# In[16]:

#contribution des individus dans l'inertie totale
di = np.sum(Z**2,axis=1)
print(pd.DataFrame({'ID':X.index,'d_i':di}))

# In[17]:

#qualité de représentation des individus - COS2
cos2 = coord**2
for j in range(p):
 cos2[:,j] = cos2[:,j]/di
print(pd.DataFrame({'id':X.index,'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]}))

# In[18]:

#vérifions la théorie - somme en ligne des cos2 = 1
print(np.sum(cos2,axis=1))

# In[19]:

#contributions aux axes
ctr = coord**2
for j in range(p):
 ctr[:,j] = ctr[:,j]/(n*eigval[j])

print(pd.DataFrame({'id':X.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]}))

# In[20]:

#vérifions la théorie
print(np.sum(ctr,axis=0))

# In[21]:

#le champ components_ de l'objet ACP
print(acp.components_)

# In[22]:

#racine carrée des valeurs propres
sqrt_eigval = np.sqrt(eigval)

#corrélation des variables avec les axes
corvar = np.zeros((p,p))
for k in range(p):
 corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

#afficher la matrice des corrélations variables x facteurs
print("corvar :")
print(corvar)



# In[23]:

#cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(p):
 plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

# In[24]:

#cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(p):
 plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#affichage

plt.savefig(dest2)

# In[25]:

#cosinus carré des variables
cos2var = corvar**2
print(pd.DataFrame({'id':X.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))

# In[26]:

#contributions
ctrvar = cos2var
for k in range(p):
 ctrvar[:,k] = ctrvar[:,k]/eigval[k]
#on n'affiche que pour les deux premiers axes
print(pd.DataFrame({'id':X.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))


#chargement des individus supplémentaires
indSupp = pd.read_excel(pre_file + name_file2 + ext_file,sheet_name=0,header=0,index_col=0)
print(indSupp)

#centrage-réduction avec les paramètres des individus actifs
ZIndSupp = sc.transform(indSupp)
print(ZIndSupp)

#projection dans l'espace factoriel
coordSupp = acp.transform(ZIndSupp)
print(coordSupp)

#positionnement des individus supplémentaires dans le premier plan
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(scaleXY * -1,scaleXY)
axes.set_ylim(scaleXY * -1,scaleXY)

#étiquette des points actifs
for i in range(n):
    plt.annotate(X.index[i],(coord[i,0],coord[i,1]), fontsize=6)
    
#étiquette des points supplémentaires (illustratifs) en bleu ‘b’
for i in range(coordSupp.shape[0]):
    plt.annotate(indSupp.index[i],(coordSupp[i,0],coordSupp[i,1]),color='b', fontsize=6)

#ajouter les axes
plt.plot([scaleXY * -1,scaleXY],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[scaleXY * -1,scaleXY],color='silver',linestyle='-',linewidth=1)

#affichage
#plt.show()
plt.savefig(dest1)


