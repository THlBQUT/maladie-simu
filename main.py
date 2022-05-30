import numpy as np
import random as rd
import matplotlib.pyplot as plt

def addToListStrat1(n):
    listePatients[n] += 1
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1

#Nombre de traitements
K = 10
probaK = [0.168, 0.393, 0.012, 0.524, 0.876, 0.690, 0.420, 0.0314, 0.666, 0.0142]

listePatients = [0]*K
listeSurvecus = [0]*K

#Nombre de patients
N = 10000
nbRep = 10

#Strategie 1
for i in range(nbRep):
    for j in range(N):
        chosenK = rd.randrange(K)
        addToListStrat1(chosenK)

for i in range(K):
    listePatients[i] /= nbRep
    listeSurvecus[i] /= nbRep

print(listePatients)
print(listeSurvecus)

names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

plt.figure(figsize=(9, 6))

plt.subplot(121)
plt.bar(names, listeSurvecus)
plt.xlabel("Remèdes")
plt.ylabel("Survivants")
plt.subplot(122)
plt.bar(names, listePatients, color='#bb3333')
plt.xlabel("Remèdes")
plt.ylabel("Nombre de patients")
plt.suptitle("Stratégie n°1")
plt.show()

#strategie 2
pkn = [0]*K
maxpkn = 0
listePatients = [0]*K
listeSurvecus = [0]*K
courbe = []

def strat2init(n):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(K, probaK[n])
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1

def strat2choix(n):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(N, probaK[n])/listePatients[n]
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    addPts(courbe)

def addPts(courbe):
        courbe.append(listeSurvecus[4])

#initialisation
for j in range(K):
    strat2init(j)

for j in range(K):
    if pkn[maxpkn] < pkn[j]:
        maxpkn = j

#initialisation fin

for j in range(K,N):
    strat2choix(maxpkn)

    for i in range(K):
        if pkn[maxpkn] < pkn[i]:
            maxpkn = i

print(pkn)
print(listePatients)
print(listeSurvecus)
print(courbe)
plt.plot(courbe)
plt.show()