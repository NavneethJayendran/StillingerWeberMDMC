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

atom_pos = PureSi(3,3,3,lat)

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
rs = (rc-al)*sigmaSi #unnormalized shell thickness
rs_sq = rs**2 #squared unnormalized shell thickness

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

class DataFrame:
  def __init__(self,  positions, U=None, S_k = None, g_r = None):
    
    if U is not None:
      self.U = U
    if S_k is not None:
      print("Implement calculation of structure factor.")
    if g_r is not None:
      print("Implement calculation of pair correlation function.")


def MC_loop(nsweeps = 1000,nc = 10, sigma=1, temp = 298, nxyz = (3,3,3),
            bxyz = None):
  (nx,ny,nz) = nxyz
  Lb = lat*np.array([nx,ny,nz])    #init box dimensions
  if bxyz is None:
    bxyz = ((0,Lb[0]), (0,Lb[1]), (0,Lb[2]))
  bx,by,bz = bxyz
  npart = nx*ny*nz*8
  atom_pos = PureSi(nx,ny,nz, lat) #init atomic positions
  neigh2, neigh2p = nlist2(bx, by, bz, rc, atom_pos)
  neigh3, neigh3p = nlist3(neigh2, neigh2p)
  U, Rij, Cij = SWPotAll(neigh2, neigh2p, nlist3, atom_pos,a*nc)
  beta = 1.0/(kB*temp)
  disp_list = np.zeros((npart, 3))
  disp_max1 = np.zeros(3); sqdist_max1 = 0; #2 max displacements since nlist
  disp_max2 = np.zeros(3); sqdist_max2 = 0; #computation & their norms squared
  for i in range(nsweeps):
    for j in range(npart):
      #don't overwrite old states in case of rejection
      recomputed = False #did we remake the neighborlists?
      disp_max1_new = disp_max1; dist_max1_old = sqdist_max1;
      disp_max2_new = disp_max2; dist_max2_old = sqdist_max2;
      dispj_new = disp_list[j]
      #end stores

      dv = np.random.rand(3)   #uniform random vector
      dv /= np.linalg.norm(v)  #uniform random unit vector
      dv *= max(rs/2,np.random.normal(sigma)) #scale by (trunc'd) Gaussian
      dispj_new += dv       #add dv to displacement j
      curr_dist = np.linalg.norm(dispj_new) #consider distance

      if curr_dist > dist_max1_new: #if longer than current max, replace
        dist_max1_new = curr_dist
        disp_max1_new = dispj_new
      elif curr_dist > dist_max2_new: #else try replacing second largest max
        dist_max2_new = curr_dist
        disp_max2_new = dispj_new

      if dist_max1_new + dist_max2_new > rs: #time to recompute neighborlists
        recomputed = True
        atom_pos[j] += dv #temporarily do this to compute the new nlists
        disp_list = np.zeros((npart,3))  #displacements are now all 0
        disp_max1 = np.zeros(3); dist_max1_new = 0;
        disp_max2 = np.zeros(3); dist_max2_new = 0; 
        neigh2, neigh2p = nlist2(bx, by, bz, rc, atom_pos) #recompute n2
        neigh3, neigh3p = nlist3(neigh2, neigh2p)          #recompute n3
        atom_pos[j] -= dv #undo that last temporary change
            
      #calculate energy differential   
      dU, Rij_new, Cij_new = SWPotOne(neigh2, neigh2p, neigh3, neigh3p, 
                                      atom_pos, Lb, j, atom_pos[j]+dv,
                                      Rij, Cij)
      if math.exp(-dU*beta) >= random.random(): #accepted move!
        U += dU           #update energy
        atom_pos[j] += dv #add dv to this atomic position
        Rij = Rij_new; Cij = Cij_new;  #update matrices
        if not recomputed: #if recomputed, we already zeroed out everything
          #but if not, update everything
          disp_list[j = disj_new 
          disp_max1 = disp_max1_new; dist_max1 = dist_max1_new;
          disp_max2 = disp_max2_new; dist_max2 = dist_max2_new;
      #rejected move!
      elif recomputed: #we "moved back" dv after computing lists, so note this
          disp_max1 = -dv; dist_max1 = np.linalg.norm(dv);

if __name__ == "__main__":
  print("Not yet implemented.")
