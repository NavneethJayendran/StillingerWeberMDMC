import numpy as np
import matplotlib.pyplot as plt
import time
import nlist as n
from initSi import PureSi
from SWPo import *

np.random.seed(0)

#lattice constant
#for testing the highest density point on the research paper's curve
#lat = 4.721714604031792
#equilibrium lattice constant for silicon
lat = 5.431
sigmasi = 2.0951 #a stillinger weber parameter 

x = PureSi(3,3,3,lat)

#print(x)
nx = np.array([-1.5,1.5])
ny = np.array([-1.5,1.5])
nz = np.array([-1.5,1.5])
Lb = np.array((lat*(nx[1]-nx[0]),lat*(ny[1]-ny[0]),lat*(nz[1]-nz[0]))) # box size, Stillinger-Weber compressed domain

#simulation bounds for binning
bx = nx*lat
by = ny*lat
bz = nz*lat

rc = sigmasi*1.8+0.5 #cutoff radius = minimum cell size (for the neighborlist's cutoff radius) 
#rc = 1.8*2.0951  #this is the potential's cutoff radius

nl2,np2 = n.nlist2(bx,by,bz,rc,x)
np.savetxt('FullNeighborlist.txt',nl2,fmt='%1.0i')
np.savetxt('FullNeighborlistPointers.txt',np2,fmt='%1.0i')
#np.savetxt('CompactNeighborlist.txt',nl2,fmt='%1.0i')
#np.savetxt('CompactNeighborlistPointers.txt',np2,fmt='%1.0i')

nl3,np3 = n.nlist3(nl2,np2)
np.savetxt('Full3BodyList.txt',nl3,fmt='%1.0i')
np.savetxt('Full3BodyListPointers.txt',np3,fmt='%1.0i')


rc = rc-0.5
#####
#Checking potential functions
U,R1,C1 = SWPotAll(nl2,np2,nl3,x,Lb)
np.savetxt('RArray.txt',R1,fmt='%1.1f')

#particle ID to displace (arbitrarily chosen
atm1 = 187
#displace by small random amount
x0 = x[atm1,:]+np.random.rand(3)*0.1
#place it back in the box, if it was displaced past periodic boundary
#x0 = (x0+Lb/2)%Lb-Lb/2
x_new = np.copy(x)
x_new[atm1,:] = x0

np.savetxt('Premove.txt',x)
np.savetxt('Postmove.txt',x_new)

Unew,Rijnew,Cijnew = SWPotAll(nl2,np2,nl3,x_new,Lb)

dPotOne,Rij2,Cij2 = SWPotOne(nl2,np2,nl3,np3,x,Lb,atm1,x0,R1,C1)

assert(np.isclose(dPotOne,Unew-U))
