#function file for initializing silicon crystal lattice

#nx,ny,nz are number of unit cells in each direction; a is lattice constant (width of cubic cells)

def PureSi(nx,ny,nz,a):
      import numpy as np
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



