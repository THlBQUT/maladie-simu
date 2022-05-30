import numpy as np
import random as rd
import matplotlib.pyplot as plt

def addToList(n):
    listePatients[n] += 1
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1

#Nombre de traitements
K = 10
probaK = [0.168, 0.393, 0.012, 0.524, 0.876, 0.690, 0.420, 0.0314, 0.666, 0.0142]

listePatients = [0]*K
listeSurvecus = [0]*K

#Nombre de patients
N = 1000
nbRep = 10

for i in range(nbRep):
    for j in range(N):
        chosenK = rd.randrange(K)
        addToList(chosenK)

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


