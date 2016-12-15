import SWMC
import numpy as np
import time
import math
import random
import heapq
np.random.seed(0)

#function to write out particle trajectories for visualization
def printpos(natm,x):
    filename = 'AtomPos.xyz'
    with open(filename,"a") as myfile:
        myfile.write( str(natm) + '\n \n')
        for i in range(natm):
            myfile.write('%4.4f %4.4f %4.4f \n'%(x[i,0],x[i,1],x[i,2]))
            
def printU(U,U2,U3):
    filename = 'PotentialTrace.dat'
    with open(filename,"a") as myfile:
        myfile.write( '%1.5f %1.5f %1.5f \n'%(U,U2,U3))

#function to ensure atoms stay within box after moving
def rebox(atom_pos,Lb):
    atom_pos = atom_pos + Lb*0.5 #shift domain to match Lb
    atom_pos = atom_pos % Lb #find correct position in shifted domain
    atom_pos = atom_pos-Lb*0.5 #shift back
    return atom_pos

class DataFrame:
  def __init__(self,  positions, U=None, S_k = None, g_r = None):
    
    if U is not None:
      self.U = U
    if S_k is not None:
      print("Implement calculation of structure factor.")
    if g_r is not None:
      print("Implement calculation of pair correlation function.")

def MC_loop(lat,rc,rs,nsweeps = 1000,nc = 10, sigma=0.0,var=0.3, temp = 1800, nxyz = (3,3,3),
            bxyz = None, atom_pos = None):
  (nx,ny,nz) = nxyz
  Lb = lat*np.array([nx,ny,nz])    #init box dimensions
  if bxyz is None:
    bx = 0.5*np.array((-Lb[0],Lb[0]))
    by = 0.5*np.array((-Lb[1],Lb[1]))
    bz = 0.5*np.array((-Lb[2],Lb[2]))
  npart = nx*ny*nz*8
  beta = 1.0/(kB*temp)

  if atom_pos is None:
      atom_pos = SWMC.PureSi(nx,ny,nz,lat) #init atomic positions

  nl2, np2 = SWMC.nlist2(bx, by, bz, rc, rs, atom_pos)
  nl3, np3 = SWMC.nlist3(nl2, np2)
  U,U2,U3 = SWMC.SWPotAll(nl2, np2, nl3, atom_pos,Lb)
  print(U/(.043*50))

  disp_list = np.zeros((npart, 3))
  dist_max1 = 0; #2 max displacements since nlist
  dist_max2 = 0; #computation & their norms squared
  i_acc = 0 #acceptance counter

  for i in range(nsweeps):
#    if i%2 ==0:
#        Utot,U2tot,U3tot = SWMC.SWPotAll(nl2, np2, nl3, atom_pos,Lb)
#        print('Utotal from full function: \t %1.5f'%Utot)
    print('U from dU: \t %1.4f'%(U))

    for j in range(npart):
      #j = int(np.random.random()*npart)
      #don't overwrite old states in case of rejection
      recomputed = False #did we remake the neighborlists?
      dist_max1_new = np.copy(dist_max1)
      dist_max2_new = np.copy(dist_max2)

      dispj_new = disp_list[j]
#      print('dispj_new'+str(dispj_new))
      #end stores

      dv = np.random.rand(3)-0.5   #uniform random vector
      dv /= np.linalg.norm(dv)  #uniform random unit vector
      dv *= min(rs/2,abs(np.random.normal(sigma,var))) #scale by (trunc'd) Gaussian
#      print('dv' + str(dv))
      dispj_new += dv       #add dv to displacement j
      curr_dist = np.linalg.norm(dispj_new) #consider distance

#      print((curr_dist,dist_max1_new,dist_max2_new))
      if sum(heapq.nlargest(2,(curr_dist,dist_max1_new,dist_max2_new))) > rs:
        #if move would cause a particle to move past threshold value, flag for new neighborlist after successfull
        recomputed = True
        nl2,np2 = SWMC.nlist2(bx,by,bz,rc,rs,atom_pos)
        dist_max2 = 0
#        disp_max1 = np.zeros(3)
#        disp_max1 = np.zeros(3)
        disp_list = np.zeros((npart,3))

      dU,dU2,dU3 = SWMC.SWPotOne(nl2, np2, nl3, np3, atom_pos, Lb, j, atom_pos[j]+dv,i)

      if math.exp(-dU*beta) >= np.random.rand(): #accepted move!
        i_acc += 1
#        print('dU for atom %3.0i is %1.4f'%(j,dU))
        if recomputed == True:
          print('New Neighborlist!')
          dist_max1 = curr_dist
          dist_max2 = 0
#          disp_max1 = np.zeros(3)
#          disp_max2 = np.zeros(3)
          disp_list = np.zeros((npart,3))
          disp_list[j,:] += dv
          recomputed = False
        else:
            dist_max1,dist_max2 = heapq.nlargest(2,(dist_max1,dist_max2,curr_dist))
#            print(sum((dist_max1,dist_max2)))
        U += dU           #update energy
        U2+= dU2
        U3+= dU3
        atom_pos[j] += dv #add dv to this atomic position
        atom_pos[j] = rebox(atom_pos[j],Lb)
#        atom_pos[0:j] -= dv/(npart-1)
#        atom_pos[j+1::] -= dv/(npart-1)
      #rejected move
      #don't updated maximum displacement distances, neighborlist, energy, or atom position
      else:
        if recomputed == True:
            dist_max1 = 0
        recomputed = False        
    print('Cumulative Acceptance Rate on sweep: \t' + str(i_acc/((i+1)*npart))+' '+str(i+1))
    printpos(npart,atom_pos)
    printU(U,U2,U3)
if __name__ == "__main__":
    lat = 5.431
    nx,ny,nz = (3,3,3)
    kB = 8.6173303e-5 #Boltzmann in eV/K
    rc = 2.0951*1.8
    rs = 0.6
#    xstart = np.loadtxt('verymelt.dat')

    MC_loop(lat,rc,rs,var=0.1,temp = 2400,nsweeps=1)

    testrun = True
    if testrun == True:
        lat = 5.431
#        lat = 4.7192

        sigmasi = 2.0951 #a stillinger weber parameter 

        x = SWMC.PureSi(3,3,3,lat)
        Natm = 3*3*3*8
#        x = xstart
        nx = np.array([-1.5,1.5])
        ny = np.array([-1.5,1.5])
        nz = np.array([-1.5,1.5])
        Lb = np.array((lat*(nx[1]-nx[0]),lat*(ny[1]-ny[0]),lat*(nz[1]-nz[0]))) # box size, Stillinger-Weber compressed domain
        print(Lb)
        #simulation bounds for binning
        bx = nx*lat
        by = ny*lat
        bz = nz*lat

        rc = sigmasi*1.8 #cutoff radius = minimum cell size (for the neighborlist's cutoff radius) 
        rs = 0.6    #shell thickness past potential rc for neighborlist generation

        t0 = time.clock()
        nl2,np2= SWMC.nlist2(bx,by,bz,rc,rs,x)
        t1 = time.clock()
        nl3,np3 = SWMC.nlist3(nl2,np2)

        #####
        #Checking potential functions
        t2 = time.clock()
        U = SWMC.SWPotAll(nl2,np2,nl3,x,Lb)
        t3 = time.clock()
        #particle ID to displace (arbitrarily chosen
        atm1 = 187
        #displace by small random amount, place back in box
        print(x[atm1])
        x0 = x[atm1,:]+np.random.rand(3)*0.5
        print(x0)
        x0 = x0+Lb*0.5
        print(x0)
        x0 = x0 % Lb
        print(x0)
        x0 = x0-Lb*0.5
        print(x0)
        x_new = np.copy(x)
        x_new[atm1] = x0
        print(x_new[atm1])
        dPotOne = SWMC.SWPotOne(nl2,np2,nl3,np3,x,Lb,atm1,x0,1)
        t4 = time.clock()

        nl22,np22 = SWMC.nlist2(bx,by,bz,rc,rs,x_new)
        nl32,np32 = SWMC.nlist3(nl22,np22)

        x_new = np.copy(x)
        x_new[atm1,:] = x0
        Unew = SWMC.SWPotAll(nl22,np22,nl32,x_new,Lb)

        print('Time elapsed for 2body lists: ' +str(t1-t0))
        print('Time elapsed for 3body lists: ' +str(t2-t1))
        print('Time elapsed for System potential: ' +str(t3-t2))
        print('Time elapsed for One potential: ' +str(t4-t3))

        print('Full Neighbors found: '+str(len(nl2)))
        #print('Maximum neighbor count: ' +str(int(max(ncntp))))
        print('Average neighbor count: ' +str(len(nl2)/Natm))
        print('Triplets found:' +str(np.shape(nl3)[0]))

        print('System energy pre-move (Total, /atom):' + str(U[0]/(.043*50))+ '\t' +str(U[0]/Natm/(.043*50)))
        print('System energy pre-move (U2, /atom):' + str(U[1]/(.043*50))+ '\t' +str(U[1]/Natm/(.043*50)))
        print('System energy pre-move (U3, /atom):' + str(U[2]/(.043*50))+ '\t' +str(U[2]/Natm/(.043*50)))

        print('System energy post-move (Total, /atom):' + str(Unew[0]/(.043*50))+ '\t' +str(Unew[0]/Natm/(.043*50)))
        print('SWPotOne: \t' +str(dPotOne[0]/(.043*50)))
        assert(np.isclose(dPotOne[0],Unew[0]-U[0]))





