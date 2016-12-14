#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import copy
import math 
import time

#lat = 5.431e-9 #silicon lattice constant

Pi         = np.pi
srPi       = np.sqrt(Pi)
kB         = 1.381e-23    # k-Boltzmann (J/K)
eps0       = 8.854187e-12 # permittivity constant (F/m = C^2/J m)
ec         = 1.60217646e-19 # elementary charge (C)
ao         = 0.053e-09 # Bohr radius
massSi     = 46.637063e-27

epsil      = 0.043
# Stillinger-Weber Constants
sigmaSi    = 2.0951
isigmaSi   = 1/sigmaSi

A          = 7.049556277
B          = 0.6022245584
psi        = 4.0
qsi        = 0.0
al         = 1.8
lambdaSi   = 21.0
gamma      = 1.2

def norm2_3d(x): #almost 3x as fast as norm2_3d
  return math.sqrt(x[0]**2+x[1]**2+x[2]**2) 

#function for determing potential energy change from moving atm1 to new position Xi
def SWPotOne(nlist2,nlist2p,nlist3,nlist3p,X,Lb,atm1,Xi,Rij,Cij): 
    t0 = time.clock()
    Rij_new = np.copy(Rij)
    Cij_new = np.copy(Cij)
    X_new = np.copy(X)
    X_new[atm1,:] = Xi
    SWPotOne.iteration = 0

    U2_old = 0
    U2_new = 0
    U3_old = 0
    U3_new = 0

    #First calculate difference in energy from pair bonds
    #2 body potential
    for i in range(nlist2p[atm1],nlist2p[atm1+1]):
        atmj = nlist2[i]

        disij2 = X_new[atmj,:]-X_new[atm1,:] # vectors from i to j, i to k
        for l in range(3): 
            if disij2[l] > Lb[l]/2:
                disij2[l] = disij2[l] - Lb[l]
            elif -disij2[l] > Lb[l]/2:
                disij2[l] = disij2[l] + Lb[l]
        #old distance
        rij = Rij[min(atm1,atmj),max(atm1,atmj)]
        #new distance
        rij2 = norm2_3d(disij2)*isigmaSi
        Rij_new[min(atm1,atmj), max(atm1,atmj)] = rij2
        #>>rij2 = norm2_3d(disij2)        #seems just a bit redundant
        #>>Rij_new[min(atm1,atmj),max(atm1,atmj)] = rij2*isigmaSi
        #>>rij2 = Rij_new[min(atm1,atmj),max(atm1,atmj)]

        #extract cij for old position out of input Cij
        cij = Cij[min(atm1,atmj),max(atm1,atmj)]

        #checking for distances outside of cutoff raidus: outside will result in a potential of ~0            
        if rij2 > al:
            cij2 = -10e20
        else: 
            cij2 = 1/(rij2-al)
        Cij_new[min(atm1,atmj),max(atm1,atmj)] = cij2

        Utemp = A*(B*rij**(-psi)-1)*math.exp(cij)
        Utemp2 = A*(B*rij2**(-psi)-1)*math.exp(cij2)
        U2_old += Utemp
        U2_new += Utemp2
    # end 2 body loop

    t1 = time.clock()
    print("Time required for one atom 2 body potential:\t" + str(t1-t0))

    #triplet potetial contribution
    for i in range(nlist3p[atm1,0]):
        i=i+1  #NAV: Is this intended? i gets reset by the for loop every time
        #extract the triplet array using the pointer list's index
        atmi, atmj, atmk = nlist3[nlist3p[atm1,i]]
                
        #old displacements
        disij = X[atmj,:]-X[atmi,:] # vectors from i to j, i to k
        disik = X[atmk,:]-X[atmi,:]

        #new displacements
        disij2 = X_new[atmj,:]-X_new[atmi,:] # vectors from i to j, i to k
        disik2 = X_new[atmk,:]-X_new[atmi,:]
        # loop through x,y,z distance components and find nearest images
        for l in range(3): 
            if disij[l] > Lb[l]/2:
                disij[l] = disij[l] - Lb[l]
            elif -disij[l] > Lb[l]/2:
                disij[l] = disij[l] + Lb[l]
            # end if

            if disik[l] > Lb[l]/2:
                disik[l] = disik[l] - Lb[l]
            elif -disik[l] > Lb[l]/2:
                disik[l] = disik[l] + Lb[l]
            # end if

            if disij2[l] > Lb[l]/2:
                disij2[l] = disij2[l] - Lb[l]
            elif -disij2[l] > Lb[l]/2:
                disij2[l] = disij2[l] + Lb[l]
            # end if

            if disik2[l] > Lb[l]/2:
                disik2[l] = disik2[l] - Lb[l]
            elif -disik2[l] > Lb[l]/2:
                disik2[l] = disik2[l] + Lb[l]
            # end if

        #old distance
        rij = Rij[min(atmi,atmj),max(atmi,atmj)]*sigmaSi 
        rik = Rij[min(atmi,atmk),max(atmi,atmk)]*sigmaSi

        #new distance
        rij2 = Rij_new[min(atmi,atmj),max(atmi,atmj)]*sigmaSi 
        rik2 = Rij_new[min(atmi,atmk),max(atmi,atmk)]*sigmaSi 

        #old cos of angle
        dot = np.dot(disij,disik)
        cosjik = dot/(rij*rik)

        #new cos of angle
        dot2 = np.dot(disij2,disik2)
        cosjik2 = dot2/(rij2*rik2)

        #normalize distances
        rij = rij*isigmaSi
        rik = rik*isigmaSi
        rij2 = rij2*isigmaSi
        rik2 = rik2*isigmaSi

        #extract cij for old position out of input Cij
        cij = Cij[min(atmi,atmj),max(atmi,atmj)]
        cik = Cij[min(atmi,atmk),max(atmi,atmk)]
        cij2 = Cij_new[min(atmi,atmj),max(atmi,atmj)]
        cik2 = Cij_new[min(atmi,atmk),max(atmi,atmk)]

        hjik = lambdaSi*math.exp(gamma*(cij+cik))*(cosjik+1./3.)*(cosjik+1./3.)
        hjik2 = lambdaSi*math.exp(gamma*(cij2+cik2))*(cosjik2+1./3.)*(cosjik2+1./3.)
        U3_old += hjik
        U3_new += hjik2

    t2 = time.clock()
    print("Time required for system 3 body potential:\t" + str(t2-t1))

    U_old = U2_old+U3_old
    U_new = U2_new+U3_new
    dPot = U_new-U_old
    return dPot,Rij_new,Cij_new

SWPotOne.iteration = 0

def SWPotAll(nlist2,nlist2p,nlist3,X,Lb):
    t0 = time.clock()
    Natm = np.shape(X)[0]
    U2 = 0 #initial system 2 body potential energy scalar
    U3 = 0 #initial system 3 body potential energy scalar
    #stored distances
    Rij = np.zeros((Natm,Natm))

    #stored components of exponential terms
    Cij = np.zeros((Natm,Natm))

    #2 body potential
    for i in range(Natm):
        atmi = i
        for j in range(nlist2p[i],nlist2p[i+1]):
            atmj = nlist2[j]
            #only calculate each pair's energy once
            if atmi > atmj: continue

            disij = X[atmj,:]-X[atmi,:] # vectors from i to j, i to k

            # loop through x,y,z distance components and find nearest images
            for l in range(3): 
                if disij[l] > Lb[l]/2:
                    disij[l] = disij[l] - Lb[l]
                elif -disij[l] > Lb[l]/2:
                    disij[l] = disij[l] + Lb[l]
                # end if

            rij = norm2_3d(disij)*isigmaSi

            Rij[atmi,atmj] = rij

            #checking for distances outside of cutoff radius: outside will result in a potential of ~0            
            if rij > al:
                cij = -1e20
            else: 
                cij = 1/(rij-al)

            #store normalized interaction distances for future use
            Cij[atmi,atmj] = cij

            Utemp = A*(B*rij**(-psi)-1)*math.exp(cij)
            U2 += Utemp
        # end for
    # end for 2 body loop
    t1 = time.clock()
    print("Time required for system 2 body potential:\t" + str(t1-t0))

    #3 body potential
    for i in range(np.shape(nlist3)[0]):
        #atom IDs for each triplet        
        atmi= nlist3[i,0]
        atmj = nlist3[i,1]
        atmk = nlist3[i,2]

        disij = X[atmj,:]-X[atmi,:] # vectors from i to j, i to k
        disik = X[atmk,:]-X[atmi,:]

        # loop through x,y,z distance components and find nearest images
        for l in range(3): 
            if disij[l] > Lb[l]/2:
                disij[l] = disij[l] - Lb[l]
            elif -disij[l] > Lb[l]/2:
                disij[l] = disij[l] + Lb[l]
            # end if

            if disik[l] > Lb[l]/2:
                disik[l] = disik[l] - Lb[l]
            elif -disik[l] > Lb[l]/2:
                disik[l] = disik[l] + Lb[l]
            # end if

        rij = Rij[min(atmi,atmj),max(atmi,atmj)]*sigmaSi
        rik = Rij[min(atmi,atmk),max(atmi,atmk)]*sigmaSi

        dot = np.dot(disij,disik)
        cosjik = dot/(rij*rik)

        rij = rij*isigmaSi
        rik = rik*isigmaSi

        cij = Cij[min(atmi,atmj),max(atmi,atmj)]
        cik = Cij[min(atmi,atmk),max(atmi,atmk)]

        hjik = lambdaSi*math.exp(gamma*(cij+cik))*(cosjik+1./3.)*(cosjik+1./3.)
        U3 += hjik
    #end for 3 body loops
    t2 = time.clock()
    print("Time required for system 3 body potential:\t" + str(t2-t1))

    #total 2 and 3 body potentials
    U = (U2+U3)

    U_average = U/Natm
    U2_average = U2/Natm
    U3_average = U3/Natm

    print(U_average)
    print("Average potential per atom: %1.10f" % U_average)
    print("Average 2 body potential: %1.10f" % U2_average)
    print("Average 3 body potential: %1.10f" % U3_average)
    return U, Rij, Cij

