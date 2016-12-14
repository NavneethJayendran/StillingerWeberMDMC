import numpy as np
import time
#function for creating 2 body neighborlists


def distance_init(nparts, dist_mat, sigma):
  distance_sigma_inv = 1.0/sigma
  distance_matrix = dist_mat
  distance_matrix_updates = np.zeros((nparts, nparts), dtype=int)

def memoized_distance(i, j, update, positions):
  if update != distance_matrix_updates[i,j]:
    distance_matrix_updates[i,j] = update
    distance_matrix[i,j] = \
      np.linalg.norm(positions[i,:,:]-positions[j,:,:])*distance_sigma_inv
  return distance_matrix_updates[i,j]

def nlist2(bx,by,bz,rc,data):
    t0 = time.clock()
    #number of bins in each direction, dictated by box dimension and cutoff radius
    nx = int(np.floor((bx[1]-bx[0])/rc))
    ny = int(np.floor((by[1]-by[0])/rc))
    nz = int(np.floor((bz[1]-bz[0])/rc))

    rc = rc**2  #compare distances squared to avoid a sqrt calculation on each distance

    #sub-bins, 2 per rc division
    d = 2

    #(trying to reduce number per bin for looping)
    nx = nx*d
    ny = ny*d
    nz = nz*d

    idnx = nx/(bx[1]-bx[0])
    idny = ny/(by[1]-by[0])
    idnz = nz/(bz[1]-bz[0])

    print('bin size: ' +str(1/idnx) +', '+str(1/idny) +', '+ str(1/idnz) )
    blx = bx[1]-bx[0]
    bly = by[1]-by[0]
    blz = bz[1]-bz[0]


    print(nx,ny,nz)
    #counter for particles placed in each bin
    bincnt = np.zeros((nx,ny,nz),dtype=int)

    #storage for pointers to particle IDs, estimate ~1-2 per bin but give extra space
    bins = np.zeros((nx,ny,nz,5),dtype=int)

    #load xyz file; particle coordinates in first three columns, header info first 9 rows
    natm = np.shape(data)[0]

    #also create a map for particle # to bin location
    atmbin = np.zeros((natm,3),dtype=int)

    #neighborlist & nlist pointer initializaton
    nlistf = np.zeros((natm*20),dtype=int)

    nlistpf = np.zeros(natm+1,dtype=int)
    #counter for total number of neighbors found
    ncntf = 0

    #storage for per-atom neighbor count, mostly for debugging/optimization
    ncntp = np.zeros(natm,dtype=int)

    #pointer index count
    pcntf = 1

    print('Number of atoms in list:\t' + str(natm))

    for i in range(natm):
        binx = int(np.floor((data[i,0]-bx[0])*idnx))
        biny = int(np.floor((data[i,1]-by[0])*idny))
        binz = int(np.floor((data[i,2]-bz[0])*idnz))

        atmbin[i] = [binx,biny,binz]
        bins[binx,biny,binz,bincnt[binx,biny,binz]] = i
        bincnt[binx,biny,binz] += 1

    #loop integer descriptions
    #i: 'center' bin index, x
    #j: 'center' bin index, y
    #k: 'center' bin index, z
    #m : index of center atom within center bin
    #m2: index of second atom within its bin
    #atm1: 'center' atom id
    #atm2: 'second' atom id
    #i2,j2,k3: index offset for secondary bin
    #i3: neighboring bin index, x direction
    #j3: neighboring bin index, y direction
    #k3: neighboring bin index, z direction

    iflagm = False
    iflagp = False
    jflagm = False
    jflagp = False
    kflagm = False
    kflagp = False

    #loop over particle ID first, for constructing in sequential order
    for m in range(natm):
        atm1 = m
        #bin indices of center atom
        binx = int(atmbin[atm1,0])
        biny = int(atmbin[atm1,1])
        binz = int(atmbin[atm1,2])
        #position of "center" atom
        x1 = data[atm1]
        #loop over particles in neighboring bins potentially within rc (indices extending by 'd' in each direction)
        for i in range(-d,d+1):
            binx2 = binx+i
            #check indices of x direction bin for periodicity
            if (binx2 < 0):
                iflagm = True
                bixn2 = binx2+nx
            elif (binx2 >= nx):
                iflagp = True
                binx2 = binx2-nx

            for j in range(-d,d+1):
                biny2 = biny+j
                #check indices of y direction bin for periodicity
                if (biny2 < 0):
                    jflagm = True
                    biny2 = biny2+ny
                elif (biny2 >= ny):
                    jflagp = True
                    biny2 = biny2-ny

                for k in range(-d,d+1):
                    binz2 = binz+k     
                    #periodicity in z direction
                    if (binz2 < 0):
                        kflagm = True
                        binz2 = binz2+nz
                    elif (binz2 >= nz):
                        kflagp = True
                        binz2 = binz2-nz
                    nlen = bincnt[binx2,biny2,binz2]
                    #loop over atoms contained within neighboring bin
                    for m2 in range(nlen):
                        #second atom's index & position
                        atm2 = bins[binx2,biny2,binz2,m2]
                        if (atm1 == atm2): continue
                        #make sure to copy value, so we don't accidently overwrite it's position when calculating periodicity
                        x2 = np.copy(data[atm2])

                        if (iflagm == True): x2[0] -=blx
                        if (iflagp == True): x2[0] +=blx
                        if (jflagm == True): x2[1] -=bly
                        if (jflagp == True): x2[1] +=bly
                        if (kflagm == True): x2[2] -=blz
                        if (kflagp == True): x2[2] +=blz

                        #squared distance for squared rc
                        dx = np.dot(x1-x2,x1-x2)
                        if(dx<rc):
                            #place second atom's index on the full neighbor list #includes duplicate pairs
                            ncntp[atm1] += 1
                            nlistf[ncntf] = atm2
                            ncntf += 1
                    #end m2 (second atom index)
                    kflagm = False
                    kflagp = False
                #end k (second atom's z bin)
                jflagm = False
                jflagp = False
            #end j (second atom's y bin)
            iflagm = False
            iflagp = False
        #end i (second atoms x bin)
        #                  nlistp[pcnt] = ncnt
        #                  pcnt += 1
        nlistpf[pcntf] = ncntf
        pcntf += 1
        #                  print(nlistpf[pcnt-1]-nlistpf[pcnt-2])
    #end m (first atom's index)

    nlistf = np.delete(nlistf,np.s_[ncntf::])
    print('Full Neighbors found: '+str(ncntf))
    print('Time elapsed for 2body lists: ' +str(time.clock()-t0))
    print('Maximum neighbor count: ' +str(int(max(ncntp))))
    print('Average neighbor count: ' +str(np.average(ncntp)))
    print(int(sum(ncntp)))
    return nlistf, nlistpf

#function for creating a 3 body neighborlist out of the full 2 body list (includes 3 of each triplet, once each for each central atom numbering)
def nlist3(nlistf,nlistpf):
    t0 = time.clock()

    natm = np.shape(nlistpf)[0]-1
    nlist = np.zeros((natm*200,3),dtype = int)

    #3body pointer stores the triplet indices of each triplet involved with atm1
    #first index out of 120*4 is how many triplet indices follow
    nlistp = -np.ones((natm,120*4),dtype=int)
    nlistp[:,0] = 0
    #counter for triplets found
    cnt3 = 0
    #counter for where each central atom's triplet listings begin in the nlist

    for i in range(natm):
        atm1 = i
        for j in range(nlistpf[i],nlistpf[i+1]):
            atm2 = nlistf[j]
            for k in range(j+1,nlistpf[i+1]):
                atm3 = nlistf[k]

                nlistp[atm1,0] += 1
                nlistp[atm2,0] += 1
                nlistp[atm3,0] += 1

                nlist[cnt3,:] = atm1,atm2,atm3
                nlistp[atm1,nlistp[atm1,0]] = cnt3
                nlistp[atm2,nlistp[atm2,0]] = cnt3
                nlistp[atm3,nlistp[atm3,0]] = cnt3

                cnt3 += 1

    nlist = np.delete(nlist,np.s_[cnt3::],axis=0)

    print('Time elapsed for 3body lists: ' +str(time.clock()-t0))
    print('Triplets found:' +str(cnt3))
    return nlist,nlistp

#end nlist3

