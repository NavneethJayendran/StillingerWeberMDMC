import numpy as np
import time
import math
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt

#One file containing all of the necessary functions for our MC loop, to be used with Cython for speed

DTYPE = np.int
#DTYPEf = np.float64

#converts a raw C array to a numpy array

cdef pointer_to_numpy_array_float64(void * ptr, np.npy_intp size):
  cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
  cdef np.ndarray[double, ndim=1] arr = \
      np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT64, ptr)
  PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
  return arr

cdef extern from "cutils.h":
  double distance(double *v1, double *v2)
  double disp_in_box(double *v1, double *v2, double *box, double *out)
  void normalize2_3d(double *vec)
  double norm2_3d(double *vec)
  double dot3d(double *vec1, double *vec2)
  double vec_cos3d(double *vec1, double *vec2)

@cython.boundscheck(False)
@cython.wraparound(False)
def c_distance(np.ndarray[double, ndim=1, mode="c"] v1 not None,
             np.ndarray[double, ndim=1, mode="c"] v2 not None):
  return distance(&v1[0], &v2[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def c_disp_in_box(np.ndarray[double, ndim=1, mode="c"] v1 not None,
                  np.ndarray[double, ndim=1, mode="c"] v2 not None,
                  np.ndarray[double, ndim=1, mode="c"] box not None,
                  np.ndarray[double, ndim=1, mode="c"] out not None):

  return disp_in_box(&v1[0], &v2[0], &box[0], &out[0])

def p_disp_in_box(np.ndarray[double, ndim=1, mode="c"] v1 not None,
                  np.ndarray[double, ndim=1, mode="c"] v2 not None,
                  np.ndarray[double, ndim=1, mode="c"] box not None,
                  np.ndarray[double, ndim=1, mode="c"] out not None):
  double b0h = box[0]/2, b1h = box[1]/2, b2h = box[2]/2;
  out[0] = v2[0]-v1[0];
  out[1] = v2[1]-v1[1];
  out[2] = v2[2]-v1[2];
  if (out[0] > b0h)
    out[0] -= box[0];
  elif (out[0] < -b0h)
    out[0] += box[0];
  if (out[1] > b1h)
    out[1] -= box[1];
  elif (out[1] < -b1h)
    out[1] += box[1];
  if (out[2] > b2h)
    out[2] -= box[2];
  elif (out[2] < -b2h)
    out[2] += box[2];
  return sqrt(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]);


@cython.boundscheck(False)
@cython.wraparound(False)
def c_norm2_3d(np.ndarray[double, ndim=1, mode="c"] vec not None):
  return norm2_3d(&vec[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def c_normalize2_3d(np.ndarray[double, ndim=1, mode="c"] vec not None):
  normalize2_3d(&vec[0]);
  return None

@cython.boundscheck(False)
@cython.wraparound(False)
def c_dot3d(np.ndarray[double, ndim=1, mode="c"] v1 not None,
            np.ndarray[double, ndim=1, mode="c"] v2 not None):
  return dot3d(&v1[0], &v2[0])


ctypedef np.int_t DTYPE_t
#ctypedef np.float32 DTYPEf_t

cdef double Pi         = np.pi
cdef double srPi       = np.sqrt(Pi)
cdef double kB         = 1.381e-23    # k-Boltzmann (J/K)
cdef double eps0       = 8.854187e-12 # permittivity constant (F/m = C^2/J m)
cdef double ec         = 1.60217646e-19 # elementary charge (C)
cdef double ao         = 0.053e-09 # Bohr radius
cdef double massSi     = 46.637063e-27

cdef double epsil      = 0.043*50
#cdef double epsil      = 1
# Stillinger-Weber Constants
cdef double sigmaSi    = 2.0951
cdef double isigmaSi   = 1/sigmaSi

cdef double A          = 7.049556277
cdef double B          = 0.6022245584
cdef double psi        = 4.0
cdef double qsi        = 0.0
cdef double al         = 1.8
cdef double lambdaSi   = 21.0
cdef double gamma      = 1.2


#function file for initializing silicon crystal lattice
#nx,ny,nz are number of unit cells in each direction; a is lattice constant (width of cubic cells)
def PureSi(nx,ny,nz,a):
      uc = np.zeros((8,3))
      uc[0,0] = 0
      uc[0,1] = 0
      uc[0,2] = 0

      uc[1,0] = 0
      uc[1,1] = 0.5*a
      uc[1,2] = 0.5*a

      uc[2,0] = 0.25*a
      uc[2,1] = 0.25*a
      uc[2,2] = 0.75*a

      uc[3,0] = 0.25*a
      uc[3,1] = 0.75*a
      uc[3,2] = 0.25*a

      uc[4,0] = 0.5*a
      uc[4,1] = 0
      uc[4,2] = 0.5*a

      uc[5,0] = 0.5*a
      uc[5,1] = 0.5*a
      uc[5,2] = 0

      uc[6,0] = 0.75*a
      uc[6,1] = 0.25*a
      uc[6,2] = 0.25*a

      uc[7,0] = 0.75*a
      uc[7,1] = 0.75*a
      uc[7,2] = 0.75*a

      X = np.zeros((nx*ny*nz*8,3))
      n = 0
      for k in range(nz):
            for j in range(ny):
                  for i in range(nx):
                        for m in range(8):
                              X[n,0] = uc[m,0]+a*i
                              X[n,1] = uc[m,1]+a*j
                              X[n,2] = uc[m,2]+a*k
                              n += 1

      #offset for origin at center of cell
      X[:,0] = X[:,0] - a*nx/2
      X[:,1] = X[:,1] - a*ny/2
      X[:,2] = X[:,2] - a*nz/2

      return X


@cython.boundscheck(False)
@cython.wraparound(False)
def SWPotOne(
    np.ndarray[DTYPE_t] nlist2,
    np.ndarray[DTYPE_t] nlist2p,
    np.ndarray[DTYPE_t,ndim=2] nlist3,
    np.ndarray[DTYPE_t,ndim=2] nlist3p,
    np.ndarray[double,ndim=2] X,
    np.ndarray[double] Lb,
    int atm1,
    np.ndarray[double] Xi,
    int flag):

    #declaring data types for Cython

    nlist2len = len(nlist2)
    cdef int i,j,l, amin, amax, amin2, amax2
    cdef int natm = X.shape[0]
    cdef double rij, rik, rij2, rik2, Utemp, Utemp2, dot, dot2, cosjik, cosjik2, hjik, hjik2
    cdef np.ndarray[double] disij, disik, disij2, disik2
    cdef double t0 = time.clock()

#    cdef np.ndarray[double,ndim=2] Rij_new = np.copy(Rij)
#    cdef np.ndarray[double,ndim=2] Cij_new = np.copy(Cij)
#    cdef np.ndarray[double,ndim=2] X_new = np.copy(X)
#    X_new[atm1,:] = Xi

    cdef double U2_old = 0
    cdef double U2_new = 0
    cdef double U3_old = 0
    cdef double U3_new = 0

    #First calculate difference in energy from pair bonds
    #2 body potential
    for i in range(nlist2p[atm1],nlist2p[atm1+1]):
        atmj = nlist2[i]

        disij = X[atmj,:]-X[atm1,:]
        disij2 = X[atmj,:]-Xi # vectors from i to j, i to k

        for l in range(3): 
            if disij[l] > Lb[l]/2:
                disij[l] = disij[l] - Lb[l]
            elif -disij[l] > Lb[l]/2:
                disij[l] = disij[l] + Lb[l]

        for l in range(3): 
            if disij2[l] > Lb[l]/2:
                disij2[l] = disij2[l] - Lb[l]
            elif -disij2[l] > Lb[l]/2:
                disij2[l] = disij2[l] + Lb[l]


        #old distance
        rij = c_norm2_3d(disij)*isigmaSi
        #new distance
        rij2 = c_norm2_3d(disij2)*isigmaSi

        #checking for distances outside of cutoff raidus: outside will result in a potential of ~0            
        if rij > al:
            cij = -10e20
        else: 
            cij = 1/(rij-al)

        if rij2 > al:
            cij2 = -10e20
        else: 
            cij2 = 1/(rij2-al)

        Utemp = A*(B*1.0/(rij*rij*rij*rij)-1)*exp(cij)
        Utemp2 = A*(B*1.0/(rij2*rij2*rij2*rij2)-1)*exp(cij2)

        U2_old += Utemp
        U2_new += Utemp2

#        if(flag > 10):
#            print('atom1: %1.3i atomj: %1.3i old U2 %1.5f: \t'%(atm1,atmj,Utemp))
        
#            print('new U2: \t' + str(U2_new))
#            if U2_old > 100:
#                print('atom1: %1.3i atomj: %1.3i old Rij %1.5f: \t'%(atm1,atmj,rij))

#                exit()

    # end 2 body loop

#    t1 = time.clock()
#    print("Time required for one atom 2 body potential:\t" + str(t1-t0))

    #triplet potetial contribution
    for i in range(nlist3p[atm1,0]):
        #extract the triplet array using the pointer list's index
        atmi, atmj, atmk = nlist3[nlist3p[atm1,i+1]]

        if atmi == atm1:
            disij = X[atmj,:]-X[atmi,:]
            disik = X[atmk,:]-X[atmi,:]
            disij2 = X[atmj,:]-Xi
            disik2 = X[atmk,:]-Xi
        elif atmj == atm1:
            disij = X[atmj,:]-X[atmi,:]
            disik = X[atmk,:]-X[atmi,:]
            disij2 = Xi-X[atmi,:]
            disik2 = disik
        else:
            disij = X[atmj,:]-X[atmi,:]
            disik = X[atmk,:]-X[atmi,:]
            disij2 = disij
            disik2 = Xi-X[atmi,:]

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
        rij = c_norm2_3d(disij)
        rik = c_norm2_3d(disik)

        #new distance
        rij2 = c_norm2_3d(disij2)
        rik2 = c_norm2_3d(disik2)

#        if rik == 0:
#            print('Atom i and Atom k: \t' + str(atmi)+' '+str(atmk))
#            print(amin2,amax2)
#        if rij == 0:
#            print('Atom i and Atom j: \t' + str(atmi)+' '+str(atmj))
#            print(amin,amax)

        #old cos of angle
        dot = c_dot3d(disij,disik)
        cosjik = dot/(rij*rik)

        #new cos of angle
        dot2 = c_dot3d(disij2,disik2)
        cosjik2 = dot2/(rij2*rik2)

        #normalize distances
        rij = rij*isigmaSi
        rik = rik*isigmaSi
        rij2 = rij2*isigmaSi
        rik2 = rik2*isigmaSi


        #checking for distances outside of cutoff raidus: outside will result in a potential of ~0            
        if rij > al:
            cij = -10e20
        else: 
            cij = 1/(rij-al)

        if rik > al:
            cik = -10e20
        else: 
            cik = 1/(rik-al)

        #checking for distances outside of cutoff raidus: outside will result in a potential of ~0            
        if rij2 > al:
            cij2 = -10e20
        else: 
            cij2 = 1/(rij2-al)

        if rik2 > al:
            cik2 = -10e20
        else: 
            cik2 = 1/(rik2-al)


        #extract cij from 2 body calculations
#        cij = Cij[amin,amax]
#        cik = Cij[amin2,amax2]
#        cij2 = Cij_new[amin,amax]
#        cik2 = Cij_new[amin2,amax2]
    
#        print(cij,cik)
        hjik = exp(gamma*(cij+cik))*(cosjik+1./3.)*(cosjik+1./3.)
        hjik2 = exp(gamma*(cij2+cik2))*(cosjik2+1./3.)*(cosjik2+1./3.)
        U3_old += hjik
        U3_new += hjik2
#    if(flag > 10):
#        print('old U3: \t' + str(U3_old))
#        print('new U3: \t' + str(U3_new))

#    t2 = time.clock()
#    print("Time required for system 3 body potential:\t" + str(t2-t1))

    U_old = U2_old+U3_old*lambdaSi
    U_new = U2_new+U3_new*lambdaSi
    dPot = (U_new-U_old)*epsil
    return dPot

#Not bothering to optimize, since it's only called once
def SWPotAll(nlist2,nlist2p,nlist3,X,Lb):
#    t0 = time.clock()
    Natm = np.shape(X)[0]
    U2 = 0 #initial system 2 body potential energy scalar
    U3 = 0 #initial system 3 body potential energy scalar
    #stored distances
#    Rij = np.zeros((Natm,Natm))

    #stored components of exponential terms
#    Cij = np.zeros((Natm,Natm))

    #2 body potential
    for i in range(Natm):
        atmi = i
        for j in range(nlist2p[i],nlist2p[i+1]):
            atmj = nlist2[j]
            #only calculate each pair's energy once
            if atmi > atmj: continue

#            rij = Rij[atmi,atmj]
#            cij = Cij[atmi,atmj]
            disij = X[atmj,:]-X[atmi,:] # vectors from i to j, i to k

            # loop through x,y,z distance components and find nearest images
            for l in range(3): 
                if disij[l] > Lb[l]/2:
                    disij[l] = disij[l] - Lb[l]
                elif -disij[l] > Lb[l]/2:
                    disij[l] = disij[l] + Lb[l]
                # end if
            rij = np.linalg.norm(disij)*isigmaSi            

            if rij > al:
                cij = -10e20
            else: 
                cij = 1/(rij-al)

            Utemp = A*(B*1.0/(rij*rij*rij*rij)-1)*exp(cij)
#            print('Utemp 2 body:\t' + str(Utemp))
            U2 += Utemp
#        print(U2)
        # end for
    # end for 2 body loop
#    t1 = time.clock()
#    print("Time required for system 2 body potential:\t" + str(t1-t0))

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

        rij = c_norm2_3d(disij)
        rik = c_norm2_3d(disik)

        dot = c_dot3d(disij,disik)
        cosjik = dot/(rij*rik)

        rij = rij*isigmaSi
        rik = rik*isigmaSi

        #checking for distances outside of cutoff raidus: outside will result in a potential of ~0            
        if rij > al:
            cij = -10e20
        else: 
            cij = 1/(rij-al)

        if rik > al:
            cik = -10e20
        else: 
            cik = 1/(rik-al)

        hjik = lambdaSi*exp(gamma*(cij+cik))*(cosjik+1./3.)*(cosjik+1./3.)
        U3 += hjik
#    print('U3: \t'+str(U3))
    #end for 3 body loops
#    t2 = time.clock()
#    print("Time required for system 3 body potential:\t" + str(t2-t1))

    #total 2 and 3 body potentials
    U = (U2+U3)*epsil

    U_average = U/Natm
    U2_average = U2/Natm
    U3_average = U3/Natm

#    print(U_average)
#    print("Average potential per atom: %1.10f" % U_average)
#    print("Average 2 body potential: %1.10f" % U2_average)
#    print("Average 3 body potential: %1.10f" % U3_average)
    return U


#optimizing for cython
@cython.boundscheck(False)
#@cython.wraparound(False)
def nlist2(
    np.ndarray[double] bx,
    np.ndarray[double] by,
    np.ndarray[double] bz,
    double rc, double rs,
    np.ndarray[double,ndim=2] x):

    cdef int nx,ny,nz,d,m,i,j,k,m2,natm,ncntf,pcntf,amax,amin
    cdef int binx,biny,binz,binx2,biny2,binz2,nlen,atm1,atm2
    
    cdef double t0,blx,bly,blz,idnx,idny,idnz
    cdef np.ndarray[double] x1,x2

    t0 = time.clock()
    #number of bins in each direction, dictated by box dimension and cutoff radius
    nx = int(np.floor((bx[1]-bx[0])/(rc+rs)))
    ny = int(np.floor((by[1]-by[0])/(rc+rs)))
    nz = int(np.floor((bz[1]-bz[0])/(rc+rs)))

    rc2 = (rc+rs)*(rc+rs)  #compare distances squared to avoid a sqrt calculation on each distance

    #sub-bins, 2 per rc division
    d = 2

    #(trying to reduce number per bin for looping)
    nx = nx*d
    ny = ny*d
    nz = nz*d

    idnx = nx/(bx[1]-bx[0])
    idny = ny/(by[1]-by[0])
    idnz = nz/(bz[1]-bz[0])

#    print('bin size: ' +str(1/idnx) +', '+str(1/idny) +', '+ str(1/idnz) )
    blx = bx[1]-bx[0]
    bly = by[1]-by[0]
    blz = bz[1]-bz[0]


#    print(nx,ny,nz)
    #counter for particles placed in each bin
    cdef np.ndarray[np.int_t,ndim=3] bincnt = np.zeros((nx,ny,nz),dtype=np.int)

    #storage for pointers to particle IDs, estimate ~1-2 per bin but give extra space
    cdef np.ndarray[np.int_t,ndim=4] bins = np.zeros((nx,ny,nz,5),dtype=np.int)

    #load xyz file; particle coordinates in first three columns, header info first 9 rows
    natm = np.shape(x)[0]

    #also create a map for particle # to bin location
    cdef np.ndarray[np.int_t,ndim=2] atmbin = np.zeros((natm,3),dtype=np.int)

    #neighborlist & nlist pointer initializaton
    cdef np.ndarray[np.int_t] nlistf = np.zeros((natm*50),dtype=np.int)

    cdef np.ndarray[np.int_t] nlistpf = np.zeros(natm+1,dtype=np.int)
    #counter for total number of neighbors found
    ncntf = 0

    #storage for per-atom neighbor count, mostly for debugging/optimization
    cdef np.ndarray[np.int_t] ncntp = np.zeros(natm,dtype=np.int)

    #pointer index count
    pcntf = 1

    #Initialize some Rij, Cij matrices for use in potential functions
#    cdef np.ndarray[double,ndim=2] Rij = np.zeros((natm,natm))

    #stored components of exponential terms
#    cdef np.ndarray[double,ndim=2] Cij = np.zeros((natm,natm))

#    print('Number of atoms in list:\t' + str(natm))

    for i in range(natm):
        binx = int(np.floor((x[i,0]-bx[0])*idnx))
        biny = int(np.floor((x[i,1]-by[0])*idny))
        binz = int(np.floor((x[i,2]-bz[0])*idnz))

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
        x1 = x[atm1]
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
                        x2 = np.copy(x[atm2])

                        if (iflagm == True): x2[0] =x2[0]-blx
                        if (iflagp == True): x2[0] =x2[0]+blx
                        if (jflagm == True): x2[1] =x2[1]-bly
                        if (jflagp == True): x2[1] =x2[1]+bly
                        if (kflagm == True): x2[2] =x2[2]-blz
                        if (kflagp == True): x2[2] =x2[2]+blz

                        #squared distance for squared rc
                        dx = c_dot3d(x1-x2,x1-x2)
                        if(dx<rc2):
                            #place second atom's index on the full neighbor list #includes duplicate pairs
                            ncntp[atm1] = ncntp[atm1]+1
                            nlistf[ncntf] = atm2
                            ncntf += 1

                        #now calculate and store quantities that need to be initialized for potential
#                            if(atm1<atm2):
#                            rij = sqrt(dx)
#                            Rij[atm1,atm2] = rij*isigmaSi
#                            print('scaled rij \t' + str(rij*isigmaSi))
                            #checking for distances outside of cutoff radius: outside will result in a potential of ~0
#                            if rij > rc:
#                                cij = -1e20
#                            else: 
#                                cij = 1/(Rij[atm1,atm2]-al)
#                            Cij[atm1,atm2] = cij
#                            print('cij \t'+str(cij))
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
#    print('Full Neighbors found: '+str(ncntf))
#    print('Time elapsed for 2body lists: ' +str(time.clock()-t0))
#    print('Maximum neighbor count: ' +str(int(max(ncntp))))
    print('Average neighbor count: ' +str(np.average(ncntp)))
#    print(int(sum(ncntp)))
    return nlistf, nlistpf

#function for creating a 3 body neighborlist out of the full 2 body list (includes 3 of each triplet, once each for each central atom numbering)
#optimized for Cython
def nlist3(
    np.ndarray[DTYPE_t] nlistf,
    np.ndarray[DTYPE_t] nlistpf
    ):

    cdef int natm,cnt3,i,j,k,atm1,atm2,atm3
    cdef double t0
    cdef np.ndarray[np.int_t,ndim=2] nlist,nlistp

    t0 = time.clock()

    natm = np.shape(nlistpf)[0]-1
    nlist = np.zeros((natm*500,3),dtype = np.int)

    #3body pointer stores the triplet indices of each triplet involved with atm1
    #first index out of 120*4 is how many triplet indices follow
    nlistp = np.zeros((natm,378*4),dtype = np.int)
#    nlistp[:,0] = 0
    #counter for triplets found
    cnt3 = 0
    #counter for where each central atom's triplet listings begin in the nlist

    for i in range(natm):
        atm1 = i
        for j in range(nlistpf[i],nlistpf[i+1]):
            atm2 = nlistf[j]
            for k in range(j+1,nlistpf[i+1]):
                atm3 = nlistf[k]

                nlistp[atm1,0] = nlistp[atm1,0]+1
                nlistp[atm2,0] = nlistp[atm2,0]+1
                nlistp[atm3,0] = nlistp[atm3,0]+1

                nlist[cnt3,:] = atm1,atm2,atm3
                nlistp[atm1,nlistp[atm1,0]] = cnt3
                nlistp[atm2,nlistp[atm2,0]] = cnt3
                nlistp[atm3,nlistp[atm3,0]] = cnt3

                cnt3 += 1

    nlist = np.delete(nlist,np.s_[cnt3::],axis=0)

#    print('Time elapsed for 3body lists: ' +str(time.clock()-t0))
#    print('Triplets found:' +str(cnt3))
    return nlist,nlistp


