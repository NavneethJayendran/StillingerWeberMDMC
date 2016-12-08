#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import copy
import math 

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

def SWPotAll(nlist2,nlist2p,nlist3,X,Lb):
    Natm = np.shape(X)[0]
    U2 = 0 #initial system 2 body potential energy scalar
    U3 = 0 #initial system 3 body potential energy scalar
    #stored distances
    Rij = np.zeros((Natm,Natm))

    #stored components of exponential terms
    Cij = np.zeros((Natm,Natm))

    #3 body potential
    for i in range(np.shape(nlist3)[0]):
        #atom IDs for each triplet        
        atmi= nlist3[i,0]
        atmj = nlist3[i,1]
        atmk = nlist3[i,2]

        #print(X[atmi,:],X[atmj,:],X[atmk,:])

        disij = X[atmj,:]-X[atmi,:] # vectors from i to j, i to k, and j to k
        disik = X[atmk,:]-X[atmi,:]
        disjk = X[atmk,:]-X[atmj,:]

        rij = np.linalg.norm(disij)
        rik = np.linalg.norm(disik)
        rjk = np.linalg.norm(disjk)

        #print(rij,rik,rjk)


        # loop through x,y,z distance components and move find nearest images
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

            if disjk[l] > Lb[l]/2:
                disjk[l] = disjk[l] - Lb[l]
            elif -disjk[l] > Lb[l]/2:
                disjk[l] = disjk[l] + Lb[l]
            # end if

        rij = np.linalg.norm(disij)
        rik = np.linalg.norm(disik)
        rjk = np.linalg.norm(disjk)

        #print(rij,rik,rjk)

        dot1 = np.dot(disij,disik)
        dot2 = np.dot(disjk,disik)
        dot3 = np.dot(-disij,disjk)

        cosjik = dot1/(rij*rik)
        cosikj = dot2/(rjk*rik)
        cosijk = dot3/(rij*rjk)

        rij = rij*isigmaSi
        rik = rik*isigmaSi
        rjk = rjk*isigmaSi

        #store normalized interaction distances for future use
        Rij[min(atmi,atmj),max(atmi,atmj)] = rij
        Rij[min(atmi,atmk),max(atmi,atmk)] = rik
        Rij[min(atmj,atmk),max(atmj,atmk)] = rjk


        #checking for distances outside of cutoff raidus: outside will result in a potential of ~0            
        if rij > al:
            cij = -10e20
        else: 
            cij = 1/(rij-al)
#            print('something')
        # end if
        if rik > al:
            cik = -10e20
        else:
            cik = 1/(rik-al) 
#            print('something')
        # end if
        if rjk > al:
            cjk = -10e20
        else: 
            cjk = 1/(rjk-al)
#            print('something')
        # end if

        #print(cij,cik,cjk)
        #store normalized interaction distances for future use
        Cij[min(atmi,atmj),max(atmi,atmj)] = cij
        Cij[min(atmi,atmk),max(atmi,atmk)] = cik
        Cij[min(atmj,atmk),max(atmj,atmk)] = cjk

        hjik = lambdaSi*math.exp(gamma*(cij+cik))*(cosjik+1./3.)**2
        hijk = lambdaSi*math.exp(gamma*(cjk+cij))*(cosijk+1./3.)**2
        hikj = lambdaSi*math.exp(gamma*(cik+cjk))*(cosikj+1./3.)**2
        #print(hijk, hikj, hjik)
        #print(U3)
        U3 += hjik + hijk + hikj
    #end for 3 body loops

    for i in range(Natm):
        atmi = i
        for j in range(nlist2p[i],nlist2p[i+1]):
            atmj = nlist2[j]
            #reuse distance calculated from 3 body potential
            r = Rij[min(atmi,atmj),max(atmi,atmj)]
           # print(r)
            c = Cij[min(atmi,atmj),max(atmi,atmj)]
            Utemp = A*(B*r**(-psi)-r**(-qsi))*math.exp(c)
            # print(Utemp)
            U2 += Utemp
        # end for
    # end for 2 body loop

    #total 2 and 3 body potentials
    U = (U2+U3)

    U_average = U/Natm
    U2_average = U2/Natm
    U3_average = U3/Natm

    print(U_average)
    print("Average potential per atom: %1.10f" % U_average)
    print("Average 2 body potential: %1.10f" % U2_average)
    print("Average 3 body potential: %1.10f" % U3_average)
    return U, Rij

