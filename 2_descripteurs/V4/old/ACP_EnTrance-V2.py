
# coding: utf-8

# In[15]:


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


# In[16]:


os.chdir("/Users/laurentpottier/Documents/LP/Recherches/Projet_Fondation/Langages&Maths/Anaconda/LPanalyse/_Entrance/")
#librairie pandas
#version
print(pd.__version__) # 0.23.0
#chargement de la première feuille de données
X = pd.read_excel("Xparams_entranceSelect.xlsx",sheet_name=0,header=0,index_col=0)
#Nous remarquons que :
#• Le fichier est un classeur Excel nommé « autos_acp_pour_python.xlsx » ;
#• Les données actives sont situées dans la première feuille (sheet_name = 0) ;
#• La première ligne correspond aux noms des variables (header = 0)
#• La première colonne aux identifiants des observations (index_col = 0).


# ## Importation des données actives

# In[17]:


#dimension
print(X.shape) # (18, 6)
#nombre d'observations
n = X.shape[0]
#nombre de variables
p = X.shape[1]
#affichage des données
print( X)


# ## Préparation des données

# In[18]:


#vérification de la version
print(sklearn.__version__) # 0.19.1


# In[19]:


#classe pour standardisation
from sklearn.preprocessing import StandardScaler
#instanciation
sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(X)
print(Z)


# In[20]:


#vérification - librairie numpy
import numpy
#moyenne
print("moyenne", numpy.mean(Z,axis=0))

#écart-type
print("ecart", numpy.std(Z,axis=0,ddof=0))


# ## Analyse en composantes principales avec PCA de ‘’scikit-learn’’

# In[21]:


# Il faut instancier l’objet PCA dans un premier temps, nous affichons ses propriétés.


#classe pour l'ACP
from sklearn.decomposition import PCA
#instanciation
acp = PCA(svd_solver='full')

#affichage des paramètres
print(acp)


# In[22]:


#calculs
coord = acp.fit_transform(Z)
#nombre de composantes calculées
print(acp.n_components_) # 6


# In[23]:


# Valeurs propres et scree plot
#variance expliquée
print(acp.explained_variance_)

#valeur corrigée
eigval = (n-1)/n*acp.explained_variance_
print(eigval)

#ou bien en passant par les valeurs singulières
print(acp.singular_values_**2/n)


# In[24]:


#proportion de variance expliquée
print(acp.explained_variance_ratio_)


# In[25]:


#scree plot
plt.plot(np.arange(1,p+1),eigval)
#plt.plot(np.arange(1,n+1),eigval) ???????? si n < p

plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()


# In[26]:


#cumul de variance expliquée
plt.plot(np.arange(1,p+1),np.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance vs. # of factors")
plt.ylabel("Cumsum explained variance ratio")
plt.xlabel("Factor number")
plt.show()


# In[27]:


#seuils pour test des bâtons brisés
bs = 1/np.arange(p,0,-1)
#bs = 1/np.arange(n,0,-1)
bs = np.cumsum(bs)
bs = bs[::-1]


# In[28]:


#test des bâtons brisés
print(pd.DataFrame({'Val.Propre':eigval,'Seuils':bs}))


# In[34]:


#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-8,10) #même limites en abscisse
axes.set_ylim(-5,5) #et en ordonnée
#placement des étiquettes des observations
for i in range(n):
 plt.annotate(X.index[i],(coord[i,0],coord[i,1]))
#ajouter les axes
plt.plot([-10, 10],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-12, 12],color='silver',linestyle='-',linewidth=1)
#affichage
#plt.show()


dest = "entrance_ACP1.pdf"
   
plt.savefig(dest)


# In[35]:


#contribution des individus dans l'inertie totale
di = np.sum(Z**2,axis=1)
print(pd.DataFrame({'ID':X.index,'d_i':di}))


# In[36]:


#qualité de représentation des individus - COS2
cos2 = coord**2
for j in range(p):
 cos2[:,j] = cos2[:,j]/di
print(pd.DataFrame({'id':X.index,'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]}))


# In[37]:


#vérifions la théorie - somme en ligne des cos2 = 1
print(numpy.sum(cos2,axis=1))


# In[38]:


#contributions aux axes
ctr = coord**2
for j in range(p):
 ctr[:,j] = ctr[:,j]/(n*eigval[j])

print(pd.DataFrame({'id':X.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]}))


# In[39]:


#vérifions la théorie
print(np.sum(ctr,axis=0))


# In[40]:


#le champ components_ de l'objet ACP
print(acp.components_)


# In[41]:


#racine carrée des valeurs propres
sqrt_eigval = np.sqrt(eigval)

#corrélation des variables avec les axes
corvar = np.zeros((p,p))
for k in range(p):
 corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

#afficher la matrice des corrélations variables x facteurs
print(corvar)


# In[42]:


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



# In[43]:


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

dest = "entrance_ACP1pars.pdf"
   
plt.savefig(dest)


# In[44]:


#cosinus carré des variables
cos2var = corvar**2
print(pd.DataFrame({'id':X.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]}))


# In[45]:


#contributions
ctrvar = cos2var
for k in range(p):
 ctrvar[:,k] = ctrvar[:,k]/eigval[k]
#on n'affiche que pour les deux premiers axes
print(pd.DataFrame({'id':X.columns,'CTR_1':ctrvar[:,0],'CTR_2':ctrvar[:,1]}))

