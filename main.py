import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.stats import beta

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

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.bar(names, listeSurvecus)
plt.xlabel("Numéro du remèdes")
plt.ylabel("Personnes guéris")
plt.title("Nombre de patients guéris pour \nchaques remèdes")
plt.subplot(122)
plt.bar(names, listePatients, color='#bb3333')
plt.xlabel("Numéro du remèdes")
plt.ylabel("Nombre de patients")
plt.title("Nombre de personnes totales soignés\npour chaque remèdes")
plt.suptitle("Stratégie n°1")
plt.show()

#strategie 2
pkn = [0]*K
maxpkn = 0
# listes pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de personnes guéris
courbe0 = np.zeros(N); courbe1 = np.zeros(N); courbe2 = np.zeros(N); courbe3 = np.zeros(N);
courbe4 = np.zeros(N); courbe5 = np.zeros(N); courbe6 = np.zeros(N); courbe7 = np.zeros(N);
courbe8 = np.zeros(N); courbe9 = np.zeros(N)

# listes pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de pers ayant eu le remede X
NbPatient0 = np.zeros(N); NbPatient1 = np.zeros(N); NbPatient2 = np.zeros(N); NbPatient3 = np.zeros(N);
NbPatient4 = np.zeros(N); NbPatient5 = np.zeros(N); NbPatient6 = np.zeros(N); NbPatient7 = np.zeros(N);
NbPatient8 = np.zeros(N); NbPatient9 = np.zeros(N)

def strat2init(n):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(K, probaK[n])
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    courbe0[0] , courbe1[0], courbe2[0], courbe3[0], courbe4[0] = 0, 0, 0, 0, 0
    courbe5[0] , courbe6[0], courbe7[0], courbe8[0], courbe9[0] = 0, 0, 0, 0, 0

def strat2choix(n,N):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(N, probaK[n])/listePatients[n]
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    #Incrémentation la liste des personnes guéris dans la liste de chaque remèdes N étant le numéro du patient
    courbe0[N] += listeSurvecus[0]; courbe1[N] += listeSurvecus[1]; courbe2[N] += listeSurvecus[2]
    courbe3[N] += listeSurvecus[3]; courbe4[N] += listeSurvecus[4]; courbe5[N] += listeSurvecus[5]
    courbe6[N] += listeSurvecus[6]; courbe7[N] += listeSurvecus[7]; courbe8[N] += listeSurvecus[8]
    courbe9[N] += listeSurvecus[9]
    NbPatient0[N] += listeSurvecus[0] / listePatients[0]; NbPatient1[N] += listeSurvecus[1] / listePatients[1]
    NbPatient2[N] += listeSurvecus[2] / listePatients[2]; NbPatient3[N] += listeSurvecus[3] / listePatients[3]
    NbPatient4[N] += listeSurvecus[4] / listePatients[4]; NbPatient5[N] += listeSurvecus[5] / listePatients[5]
    NbPatient6[N] += listeSurvecus[6] / listePatients[6]; NbPatient7[N] += listeSurvecus[7] / listePatients[7]
    NbPatient8[N] += listeSurvecus[8] / listePatients[8]; NbPatient9[N] += listeSurvecus[9] / listePatients[9]


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

ax = plt.subplot(111)
plt.plot(NbPatient0,label='Remède 1')
plt.plot(NbPatient1,label='Remède 2')
plt.plot(NbPatient2,label='Remède 3')
plt.plot(NbPatient3,label='Remède 4')
plt.plot(NbPatient4,label='Remède 5')
plt.plot(NbPatient5,label='Remède 6')
plt.plot(NbPatient6,label='Remède 7')
plt.plot(NbPatient7,label='Remède 8')
plt.plot(NbPatient8,label='Remède 9')
plt.plot(NbPatient9,label='Remède 10')
plt.title('Pourcentage de patients guéris en \n fonction du nombre de patients testés')
plt.xlabel('Somme des patients testés')
plt.ylabel('Pourcentage de patients guéris')
plt.grid()
plt.legend()
ax.legend(loc='center right')
plt.show()

print(courbe4/10)
print(listeSurvecus)

#strategie 3
pkn = [0]*K
bornSupPkn = [0]*K
maxpkn = 0
# Creation d'une liste pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de personnes guéris
courbe0 = np.zeros(N); courbe1 = np.zeros(N); courbe2 = np.zeros(N); courbe3 = np.zeros(N);
courbe4 = np.zeros(N); courbe5 = np.zeros(N); courbe6 = np.zeros(N); courbe7 = np.zeros(N);
courbe8 = np.zeros(N); courbe9 = np.zeros(N)

# listes pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de pers ayant eu le remede X
NbPatient0 = np.zeros(N); NbPatient1 = np.zeros(N); NbPatient2 = np.zeros(N); NbPatient3 = np.zeros(N);
NbPatient4 = np.zeros(N); NbPatient5 = np.zeros(N); NbPatient6 = np.zeros(N); NbPatient7 = np.zeros(N);
NbPatient8 = np.zeros(N); NbPatient9 = np.zeros(N)


def strat3init(n):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(K, probaK[n])
    bornSupPkn[n] = pkn[n] + np.sqrt(2*np.log(N)/listePatients[n])
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    courbe0[0] , courbe1[0], courbe2[0], courbe3[0], courbe4[0] = 0, 0, 0, 0, 0
    courbe5[0] , courbe6[0], courbe7[0], courbe8[0], courbe9[0] = 0, 0, 0, 0, 0

def strat3choix(n,N):
    listePatients[n] += 1
    pkn[n] = np.random.binomial(N, probaK[n])/listePatients[n]
    bornSupPkn[n] = pkn[n] + np.sqrt(2*np.log(N)/listePatients[n])

    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    #Incrémentation la liste des personnes guéris dans la liste de chaque remèdes N étant le numéro du patient
    courbe0[N] += listeSurvecus[0]; courbe1[N] += listeSurvecus[1]; courbe2[N] += listeSurvecus[2]
    courbe3[N] += listeSurvecus[3]; courbe4[N] += listeSurvecus[4]; courbe5[N] += listeSurvecus[5]
    courbe6[N] += listeSurvecus[6]; courbe7[N] += listeSurvecus[7]; courbe8[N] += listeSurvecus[8]
    courbe9[N] += listeSurvecus[9]
    NbPatient0[N] += listeSurvecus[0] / listePatients[0]; NbPatient1[N] += listeSurvecus[1] / listePatients[1]
    NbPatient2[N] += listeSurvecus[2] / listePatients[2]; NbPatient3[N] += listeSurvecus[3] / listePatients[3]
    NbPatient4[N] += listeSurvecus[4] / listePatients[4]; NbPatient5[N] += listeSurvecus[5] / listePatients[5]
    NbPatient6[N] += listeSurvecus[6] / listePatients[6]; NbPatient7[N] += listeSurvecus[7] / listePatients[7]
    NbPatient8[N] += listeSurvecus[8] / listePatients[8]; NbPatient9[N] += listeSurvecus[9] / listePatients[9]

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

ax = plt.subplot(111)
plt.plot(NbPatient0,label='Remède 1')
plt.plot(NbPatient1,label='Remède 2')
plt.plot(NbPatient2,label='Remède 3')
plt.plot(NbPatient3,label='Remède 4')
plt.plot(NbPatient4,label='Remède 5')
plt.plot(NbPatient5,label='Remède 6')
plt.plot(NbPatient6,label='Remède 7')
plt.plot(NbPatient7,label='Remède 8')
plt.plot(NbPatient8,label='Remède 9')
plt.plot(NbPatient9,label='Remède 10')
plt.title('Pourcentage de patients guéris en \n fonction du nombre de patients testés')
plt.xlabel('Somme des patients testés')
plt.ylabel('Pourcentage de patients guéris')
plt.grid()
plt.legend()
ax.legend(loc='center right')
plt.show()


#strategie 4
betaK = [0]*K

# listes pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de personnes guéris
courbe0 = np.zeros(N); courbe1 = np.zeros(N); courbe2 = np.zeros(N); courbe3 = np.zeros(N);
courbe4 = np.zeros(N); courbe5 = np.zeros(N); courbe6 = np.zeros(N); courbe7 = np.zeros(N);
courbe8 = np.zeros(N); courbe9 = np.zeros(N)

# listes pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de pers ayant eu le remede X
NbPatient0 = np.zeros(N); NbPatient1 = np.zeros(N); NbPatient2 = np.zeros(N); NbPatient3 = np.zeros(N);
NbPatient4 = np.zeros(N); NbPatient5 = np.zeros(N); NbPatient6 = np.zeros(N); NbPatient7 = np.zeros(N);
NbPatient8 = np.zeros(N); NbPatient9 = np.zeros(N)

def strat4(n,N):
    listePatients[n] += 1
    if(np.random.binomial(1, probaK[n]) == 1):
        listeSurvecus[n] += 1
    #Incrémentation la liste des personnes guéris dans la liste de chaque remèdes N étant le numéro du patient
    courbe0[N] += listeSurvecus[0]; courbe1[N] += listeSurvecus[1]; courbe2[N] += listeSurvecus[2]
    courbe3[N] += listeSurvecus[3]; courbe4[N] += listeSurvecus[4]; courbe5[N] += listeSurvecus[5]
    courbe6[N] += listeSurvecus[6]; courbe7[N] += listeSurvecus[7]; courbe8[N] += listeSurvecus[8]
    courbe9[N] += listeSurvecus[9]
    if(listePatients[0] != 0):
        NbPatient0[N] += listeSurvecus[0] / listePatients[0]
    if(listePatients[1] != 0):
        NbPatient1[N] += listeSurvecus[1] / listePatients[1]
    if(listePatients[2] != 0):
        NbPatient2[N] += listeSurvecus[2] / listePatients[2]
    if(listePatients[3] != 0):
        NbPatient3[N] += listeSurvecus[3] / listePatients[3]
    if(listePatients[4] != 0):
        NbPatient4[N] += listeSurvecus[4] / listePatients[4]
    if(listePatients[5] != 0):
        NbPatient5[N] += listeSurvecus[5] / listePatients[5]
    if(listePatients[6] != 0):
        NbPatient6[N] += listeSurvecus[6] / listePatients[6]
    if(listePatients[7] != 0):
        NbPatient7[N] += listeSurvecus[7] / listePatients[7]
    if(listePatients[8] != 0):
        NbPatient8[N] += listeSurvecus[8] / listePatients[8]
    if(listePatients[9] != 0):
        NbPatient9[N] += listeSurvecus[9] / listePatients[9]



for rep in range(10):
    listePatients = [0] * K
    listeSurvecus = [0] * K

    for i in range(K,N):
        for j in range(K):
            betaK[j] = beta(1 + listeSurvecus[j], 1 + listePatients[j] - listeSurvecus[j]).rvs()
        max_index = betaK.index(max(betaK))
        strat4(max_index,i)


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

ax = plt.subplot(111)
plt.plot(NbPatient0,label='Remède 1')
plt.plot(NbPatient1,label='Remède 2')
plt.plot(NbPatient2,label='Remède 3')
plt.plot(NbPatient3,label='Remède 4')
plt.plot(NbPatient4,label='Remède 5')
plt.plot(NbPatient5,label='Remède 6')
plt.plot(NbPatient6,label='Remède 7')
plt.plot(NbPatient7,label='Remède 8')
plt.plot(NbPatient8,label='Remède 9')
plt.plot(NbPatient9,label='Remède 10')
plt.title('Pourcentage de patients guéris en \n fonction du nombre de patients testés')
plt.xlabel('Somme des patients testés')
plt.ylabel('Pourcentage de patients guéris')
plt.grid()
plt.legend()
ax.legend(loc='center right')
plt.show()

#strategie 5

# listes pour chaque remèdes avec N colonnes pour constater l'évolution du nbr de pers ayant eu le remede X
NbPatient0 = np.zeros(N-10*K);

def strat5init(K):
    for i in range(K):
        for j in range(10):
            listePatients[i] += 1
            if(np.random.binomial(1, probaK[i]) == 1):
                listeSurvecus[i] += 1
    return listeSurvecus.index(max(listeSurvecus))

max_index = strat5init(K)

for i in range(N-10*K):
    listePatients[max_index] += 1
    if (np.random.binomial(1, probaK[max_index]) == 1):
        listeSurvecus[max_index] += 1
    NbPatient0[i] += listeSurvecus[max_index] / listePatients[max_index]

ax = plt.subplot(111)
plt.plot(NbPatient0,label='Remède '+str(max_index+1))
plt.title('Evolution du pourcentage de patients traités par\n le remède choisis')
plt.xlabel('Somme des patients testés')
plt.ylabel('Pourcentage de réussite du traitement')
plt.grid()
plt.legend()
ax.legend(loc='center right')
plt.show()