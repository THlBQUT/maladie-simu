import numpy as np
import random as rd
import matplotlib.pyplot as plt

def addToListStrat1(n):
    listePatients[n] += 1
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1

#Nombre de traitements
K = 10
probaK = [np.sqrt(83)/100, 0.393, 0.012, 0.524, 0.876, 0.690, 0.420, 0.0314, 0.666, 0.0142]

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
courbe0 = np.zeros(N); courbe1 = np.zeros(N); courbe2 = np.zeros(N); courbe3 = np.zeros(N);
courbe4 = np.zeros(N); courbe5 = np.zeros(N); courbe6 = np.zeros(N); courbe7 = np.zeros(N);
courbe8 = np.zeros(N); courbe9 = np.zeros(N)

def strat2init(n):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(K, probaK[n])
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    courbe0[0] = 0
    courbe1[0] = 0
    courbe2[0] = 0
    courbe3[0] = 0
    courbe4[0] = 0
    courbe5[0] = 0
    courbe6[0] = 0
    courbe7[0] = 0
    courbe8[0] = 0
    courbe9[0] = 0

def strat2choix(n,N):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(N, probaK[n])/listePatients[n]
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    courbe0[N] += listeSurvecus[0]
    courbe1[N] += listeSurvecus[1]
    courbe2[N] += listeSurvecus[2]
    courbe3[N] += listeSurvecus[3]
    courbe4[N] += listeSurvecus[4]
    courbe5[N] += listeSurvecus[5]
    courbe6[N] += listeSurvecus[6]
    courbe7[N] += listeSurvecus[7]
    courbe8[N] += listeSurvecus[8]
    courbe9[N] += listeSurvecus[9]

for rep in range(10):
    listePatients = [0] * K
    listeSurvecus = [0] * K
    #initialisation
    for j in range(K):
        strat2init(j)

    for j in range(K):
        if pkn[maxpkn] < pkn[j]:
            maxpkn = j

    #initialisation fin

    for j in range(K,N):
        strat2choix(maxpkn,j)

        for i in range(K):
            if pkn[maxpkn] < pkn[i]:
                maxpkn = i

plt.plot(courbe0/10,label='Remède 1')
plt.plot(courbe1/10,label='Remède 2')
plt.plot(courbe2/10,label='Remède 3')
plt.plot(courbe3/10,label='Remède 4')
plt.plot(courbe4/10,label='Remède 5')
plt.plot(courbe5/10,label='Remède 6')
plt.plot(courbe6/10,label='Remède 7')
plt.plot(courbe7/10,label='Remède 8')
plt.plot(courbe8/10,label='Remède 9')
plt.plot(courbe9/10,label='Remède 10')
plt.title('Evolution du nombre de patients guéris en \n fonction du nombre de patients testés')
plt.xlabel('Somme des patients testés')
plt.ylabel('Nombre de patients guéris')
plt.grid()
plt.legend()
plt.show()

print(listePatients)
print(listeSurvecus)

#strategie 3
pkn = [0]*K
bornSupPkn = [0]*K
maxpkn = 0
listePatients = [0]*K
listeSurvecus = [0]*K
courbe0 = np.zeros(N); courbe1 = np.zeros(N); courbe2 = np.zeros(N); courbe3 = np.zeros(N);
courbe4 = np.zeros(N); courbe5 = np.zeros(N); courbe6 = np.zeros(N); courbe7 = np.zeros(N);
courbe8 = np.zeros(N); courbe9 = np.zeros(N)

def strat3init(n):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(K, probaK[n])
    bornSupPkn[n] = pkn[n] + np.sqrt(2*np.log(N)/listePatients[n])
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    courbe0[0] = 0
    courbe1[0] = 0
    courbe2[0] = 0
    courbe3[0] = 0
    courbe4[0] = 0
    courbe5[0] = 0
    courbe6[0] = 0
    courbe7[0] = 0
    courbe8[0] = 0
    courbe9[0] = 0

def strat3choix(n,N):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(N, probaK[n])/listePatients[n]
    bornSupPkn[n] = pkn[n] + np.sqrt(2*np.log(N)/listePatients[n])

    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    courbe0[N] += listeSurvecus[0]
    courbe1[N] += listeSurvecus[1]
    courbe2[N] += listeSurvecus[2]
    courbe3[N] += listeSurvecus[3]
    courbe4[N] += listeSurvecus[4]
    courbe5[N] += listeSurvecus[5]
    courbe6[N] += listeSurvecus[6]
    courbe7[N] += listeSurvecus[7]
    courbe8[N] += listeSurvecus[8]
    courbe9[N] += listeSurvecus[9]

for rep in range(10):
    listePatients = [0] * K
    listeSurvecus = [0] * K
    #initialisation
    for j in range(K):
        strat3init(j)

    for j in range(K):
        if bornSupPkn[maxpkn] < bornSupPkn[j]:
            maxpkn = j

    #initialisation fin

    for j in range(K,N):
        strat3choix(maxpkn,j)

        for i in range(K):
            if bornSupPkn[maxpkn] < bornSupPkn[i]:
                maxpkn = i

#print(pkn)
#print(bornSupPkn)
#print(listePatients)
#print(listeSurvecus)

plt.plot(courbe0/10,label='Remède 1')
plt.plot(courbe1/10,label='Remède 2')
plt.plot(courbe2/10,label='Remède 3')
plt.plot(courbe3/10,label='Remède 4')
plt.plot(courbe4/10,label='Remède 5')
plt.plot(courbe5/10,label='Remède 6')
plt.plot(courbe6/10,label='Remède 7')
plt.plot(courbe7/10,label='Remède 8')
plt.plot(courbe8/10,label='Remède 9')
plt.plot(courbe9/10,label='Remède 10')
plt.title('Evolution du nombre de patients guéris en \n fonction du nombre de patients testés')
plt.xlabel('Somme des patients testés')
plt.ylabel('Nombre de patients guéris')
plt.grid()
plt.legend()
plt.show()


