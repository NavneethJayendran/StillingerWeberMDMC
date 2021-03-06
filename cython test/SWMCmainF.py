import SWMC
import SF
import numpy as np
import time
import math
import random
import heapq
np.random.seed(0)
from sortedcontainers import SortedSet
import SWMC_brute as br

#function to write out particle trajectories for visualization
def printpos(natm,x,temp,fname=None):
#    filename = str(temp) + 'K_LiquidAtomPos.xyz'
    filename = fname + '_AtomPos.xyz'
    with open(filename,"a") as myfile:
        myfile.write( str(natm) + '\n \n')
        for i in range(natm):
            myfile.write('%4.4f %4.4f %4.4f \n'%(x[i,0],x[i,1],x[i,2]))

def printU(U,Uh,temp,fname=None):
#    filename = str(temp)+'K_LiquidPotentialTrace.dat'
    filename = fname + '_PotentialTrace.dat'
    with open(filename,"a") as myfile:
        myfile.write( '%1.5f %1.5f \n'%(U,Uh))

def printnl(nl2,np2,l,v,atmi):
    filename = 'nl2s.dat'
    n = SortedSet(nl2[np2[atmi]:np2[atmi+1]])
    d = n.difference(l)

    with open(filename,"a") as myfile:
        myfile.write(str(list(d)))
        myfile.write('\n \n')

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

def MC_loop(lat,rc,rs,nsweeps = 1000,nc = 10, sigma=0.0,var=0.3, temp = 2200, nxyz = (3,3,3),
            bxyz = None, atom_pos = None, fname = 'string', sprng = None):
  print(fname)
  (nx,ny,nz) = nxyz
  Lb = lat*np.array([nx,ny,nz])    #init box dimensions
  if bxyz is None:
    bx = 0.5*np.array((-Lb[0],Lb[0]))
    by = 0.5*np.array((-Lb[1],Lb[1]))
    bz = 0.5*np.array((-Lb[2],Lb[2]))
  npart = nx*ny*nz*8
  beta = 1.0/(kB*temp)
  USWave = 0
  UHave = 0
  avecnt = 0
  if atom_pos is None:
      atom_pos = SWMC.PureSi(nx,ny,nz,lat) #init atomic positions
  Xref = np.copy(atom_pos)
  nl2, np2 = SWMC.nlist2(bx, by, bz, rc, rs, atom_pos, 1, Lb)
  nl3, np3 = SWMC.nlist3(nl2, np2)

  dr = 0.2
  jcnt = 0
  kvecs = SF.legal_kvecs(5,Lb)

  #SW potential
  U,U2,U3 = SWMC.SWPotAll(nl2, np2, nl3, atom_pos,Lb)
  #coupled periodic harmonic potential
  Uh = SWMC.HPall(Xref,atom_pos,sprng,Lb)

  disp_list = np.zeros((npart, 3))
  dist_max1 = 0; #2 max displacements since nlist
  dist_max2 = 0; #computation & their norms squared
  i_acc = 0 #acceptance counter

  for i in range(nsweeps):
    if i > 400:
        UHave += Uh
        USWave+= U
        avecnt+=1
    print('U from dU: \t %1.4f'%(U))

    for j in range(npart):        
      #don't overwrite old states in case of rejection
      recomputed = False #did we remake the neighborlists?
      dist_max1_new = np.copy(dist_max1)
      dist_max2_new = np.copy(dist_max2)

      dispj_new = disp_list[j]

      dv = np.random.rand(3)-0.5   #uniform random vector
      dv /= np.linalg.norm(dv)  #uniform random unit vector
      dv *= min(rs/2,abs(np.random.normal(sigma,var))) #scale by (trunc'd) Gaussian
      dispj_new += dv       #add dv to displacement j
      curr_dist = np.linalg.norm(dispj_new) #consider distance

      if sum(heapq.nlargest(2,(curr_dist,dist_max1_new,dist_max2_new))) > rs:
        #if move would cause a particle to move past threshold value, flag for new neighborlist after successfull
        recomputed = True
        nl2,np2 = SWMC.nlist2(bx,by,bz,rc,rs,atom_pos,1,Lb)
        nl3,np3 = SWMC.nlist3(nl2,np2)

        dist_max2 = 0
        disp_list = np.zeros((npart,3))

      dU,dU2,dU3 = SWMC.SWPotOne(nl2, np2, nl3, np3, atom_pos, Lb, j, atom_pos[j]+dv,i)
      dUh = SWMC.HPone(Xref[j],atom_pos[j],atom_pos[j]+dv,sprng,Lb)
      if math.exp(-(dU+dUh)*beta) >= np.random.rand(): #accepted move!
        i_acc += 1
        if recomputed == True:
          print('New Neighborlist!')
          dist_max1 = curr_dist
          dist_max2 = 0
          disp_list = np.zeros((npart,3))
          disp_list[j,:] += dv
          recomputed = False
        else:
            dist_max1,dist_max2 = heapq.nlargest(2,(dist_max1,dist_max2,curr_dist))
        U += dU           #update energy
        U2+= dU2
        U3+= dU3
        Uh+= dUh
        atom_pos[j] += dv #add dv to this atomic position
        atom_pos[j] = rebox(atom_pos[j],Lb)
      #rejected move
      #don't updated maximum displacement distances, neighborlist, energy, or atom position
      else:
        if recomputed == True:
            dist_max1 = 0
        recomputed = False        
    print('Cumulative Acceptance Rate on sweep: \t' + str(i_acc/((i+1)*npart))+' '+str(i+1))
    printpos(npart,atom_pos,temp,fname=fname)
    printU(U,Uh,temp,fname=fname)
  return USWave/avecnt, UHave/avecnt, atom_pos

if __name__ == "__main__":
    lat = 5.431
    nx,ny,nz = (3,3,3)
    kB = 8.6173303e-5 #Boltzmann in eV/K
    rc = 2.0951*1.8
    rs = 0.6
    k = [35,25,15,5,1]
    vari = [0.05, 0.06, 0.07,0.08,0.15] 
    #change values here with run temperature to get an integrable free energy series
    filenames = ['Solid2200K_k=35','Solid2200K_k=25','Solid2200K_k=15','Solid2200K_k=5','Solid2200K_k=1']
    xstart = np.loadtxt('2000Kstart.dat')
    harmU = np.zeros(len(k))
    SWU = np.zeros(len(k))

    filename = 'Solid2200K_Coupled_EnergyDat.dat'
    with open(filename,"w") as myfile:        
        myfile.write('SW Potential Contribution, Harmonic Potential, spring constant \n')

    for i in range(len(harmU)):
        harmU[i],SWU[i], xstart = MC_loop(lat,rc,rs,var=vari[i],temp = 2200,nsweeps=1000, fname = filenames[i], sprng=k[i],atom_pos = xstart)
        with open(filename,"a") as myfile:
            myfile.write('%1.6f %1.6f %1.3i \n'%(harmU[i],SWU[i],k[i]))


    #troubleshooting for potential functions, not normally run
    testrun = False
    if testrun == True:
        lat = 5.431

        sigmasi = 2.0951 #a stillinger weber parameter 

        x = SWMC.PureSi(3,3,3,lat)
        Natm = 3*3*3*8
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
        nl2,np2= SWMC.nlist2(bx,by,bz,rc,rs,x,0,Lb)
        t1 = time.clock()
        nl3,np3 = SWMC.nlist3(nl2,np2)
        for atm in range(Natm):
            l,v = br.twobody_sanity(atm, x, rs, rc)
            printnl(nl2,np2,l,v,atm)
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

        nl22,np22 = SWMC.nlist2(bx,by,bz,rc,rs,x_new,1,Lb)
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





