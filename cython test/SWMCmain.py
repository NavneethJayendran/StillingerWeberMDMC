import SWMC
import numpy as np
import time
np.random.seed(0)

lat = 5.431
sigmasi = 2.0951 #a stillinger weber parameter 

x = SWMC.PureSi(3,3,3,lat)
Natm = 3*3*3*8

nx = np.array([-1.5,1.5])
ny = np.array([-1.5,1.5])
nz = np.array([-1.5,1.5])
Lb = np.array((lat*(nx[1]-nx[0]),lat*(ny[1]-ny[0]),lat*(nz[1]-nz[0]))) # box size, Stillinger-Weber compressed domain

#simulation bounds for binning
bx = nx*lat
by = ny*lat
bz = nz*lat

rc = sigmasi*1.8 #cutoff radius = minimum cell size (for the neighborlist's cutoff radius) 
rs = 0.6    #shell thickness past potential rc for neighborlist generation

t0 = time.clock()
nl2,np2 = SWMC.nlist2(bx,by,bz,rc+rs,x)
#np.savetxt('FullNeighborlist.txt',nl2,fmt='%1.0i')
#np.savetxt('FullNeighborlistPointers.txt',np2,fmt='%1.0i')
#np.savetxt('CompactNeighborlist.txt',nl2,fmt='%1.0i')
#np.savetxt('CompactNeighborlistPointers.txt',np2,fmt='%1.0i')
t1 = time.clock()
nl3,np3 = SWMC.nlist3(nl2,np2)
#np.savetxt('Full3BodyList.txt',nl3,fmt='%1.0i')
#np.savetxt('Full3BodyListPointers.txt',np3,fmt='%1.0i')

rc = rc-0.5
#rs = (rc-al)*sigmaSi #unnormalized shell thickness
#rs_sq = rs**2 #squared unnormalized shell thickness

#####
#Checking potential functions
t2 = time.clock()
U,R1,C1 = SWMC.SWPotAll(nl2,np2,nl3,x,Lb)
#np.savetxt('RArray.txt',R1,fmt='%1.1f')
t3 = time.clock()
#particle ID to displace (arbitrarily chosen
atm1 = 187
#displace by small random amount
x0 = x[atm1,:]+np.random.rand(3)*0.5
#place it back in the box, if it was displaced past periodic boundary
#x0 = (x0+Lb/2)%Lb-Lb/2
dPotOne,Rij2,Cij2 = SWMC.SWPotOne(nl2,np2,nl3,np3,x,Lb,atm1,x0,R1,C1)
t4 = time.clock()

x_new = np.copy(x)
x_new[atm1,:] = x0
Unew,Rijnew,Cijnew = SWMC.SWPotAll(nl2,np2,nl3,x_new,Lb)

print('Time elapsed for 2body lists: ' +str(t1-t0))
print('Time elapsed for 3body lists: ' +str(t2-t1))
print('Time elapsed for System potential: ' +str(t3-t2))
print('Time elapsed for One potential: ' +str(t4-t3))

print('Full Neighbors found: '+str(len(nl2)))
#print('Maximum neighbor count: ' +str(int(max(ncntp))))
print('Average neighbor count: ' +str(len(nl2)/Natm))
print('Triplets found:' +str(np.shape(nl3)[0]))

print('System energy pre-move (Total, /atom):' + str(U)+ '\t' +str(U/Natm))
print('System energy post-move (Total, /atom):' + str(Unew)+ '\t' +str(Unew/Natm))

assert(np.isclose(dPotOne,Unew-U))

