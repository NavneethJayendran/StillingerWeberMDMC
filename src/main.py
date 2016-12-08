import numpy as np
import matplotlib.pyplot as plt
import time
import nlist as n
from initSi import PureSi
from SWPo import *

lat = 4.6 #silicon lattice constant
sigmasi = 2.0951

x = PureSi(3,3,3,lat)

#print(x)
nx = np.array([-1.5,1.5])
ny = np.array([-1.5,1.5])
nz = np.array([-1.5,1.5])
Lb = [lat*(nx[1]-nx[0]),lat*(ny[1]-ny[0]),lat*(nz[1]-nz[0])] # box size, Stillinger-Weber compressed domain

#simulation bounds for binning
bx = nx*lat
by = ny*lat
bz = nz*lat

rc = sigmasi*1.8 #cutoff radius = minimum cell size (for the neighborlist's cutoff radius) 
#rc = 1.8*2.0951  #this is the potential's cutoff radius

nl2,np2,nlf2,npf2 = n.nlist2(bx,by,bz,rc,x)
np.savetxt('FullNeighborlist.txt',nlf2,fmt='%1.0i')
np.savetxt('FullNeighborlistPointers.txt',npf2,fmt='%1.0i')
np.savetxt('CompactNeighborlist.txt',nl2,fmt='%1.0i')
np.savetxt('CompactNeighborlistPointers.txt',np2,fmt='%1.0i')

nl3,nlp3 = n.nlist3(nlf2,npf2)
np.savetxt('Full3BodyList.txt',nl3,fmt='%1.0i')
np.savetxt('Full3BodyListPointers.txt',nlp3,fmt='%1.0i')

SWPotAll(nl2,np2,nl3,x,Lb)
