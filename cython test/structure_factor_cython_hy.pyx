import matplotlib.pyplot as plt
import numpy as np

cimport numpy as np
cimport cython
cdef int natom = 216  # number of atoms

@cython.boundscheck(False)
@cython.wraparound(False)
def RDF(np.ndarray[double,ndim =2] X, 
    np.ndarray[double] Lb, 
    double dr):

    cdef int tot_bin_num = int(np.ceil(Lb[0] / 2 / dr))  # total number of binning windows
    cdef np.ndarray[double] radial = np.zeros(tot_bin_num) # initialize RDF
    cdef np.ndarray[double] bins   = np.zeros(tot_bin_num) # initialize r for plot RDF
    cdef int i,j,ig
    cdef double dist

    for i in range(natom - 1):
        for j in range(i + 1, natom):
            dist = np.linalg.norm(X[j,:]-X[i,:]) # calculate distance between iat and jat 
            if dist <= Lb[0] / 2: # judge if the distance is smaller than half box length
                ig = int(np.floor(dist / dr))
                radial[ig] += 2 # consider two atoms

    for i in range(tot_bin_num):
        radial[i] = radial[i] * (1 / float(natom)) / (4/3 * np.pi * ((i + 1)**3 - i**3) * dr**3 *float(natom) / (Lb[0]**3)) # normalization
        bins[i] = (i + 0.5) * dr
    
#    plt.figure(1)
#    plt.plot(bins,radial)
#    plt.xlabel('r')
#    plt.ylabel('g(r)')

    return radial, bins
# end def RDF

# function computes a list of the legal k-vectors

@cython.boundscheck(False)
@cython.wraparound(False)
def legal_kvecs(maxk,Lb):
    cdef int n_x, n_y, n_z
    cdef np.ndarray[np.int_t] vector
    kvecs = []
    # calculate a list of legal k vectors
    for n_x in range(maxk):
        for n_y in range(maxk):
            for n_z in range(maxk):
                vector = np.array([n_x, n_y, n_z],dtype=np.int)
                kvecs.append([2 * np.pi / Lb[0] * n for n in vector])
    return np.array(kvecs)
# end def legal_kvecs

# function to compute the Fourier transform of the density for a given wavevector


@cython.boundscheck(False)
@cython.wraparound(False)
def rhok(kvec,X):
    cdef int i
    value = complex(0.0,0.0)
    cdef double dot_product,cos,sin
    #computes \sum_j \exp(i * k \dot r_j)
    for i in range(natom):
        dot_product = np.dot(kvec,X[i,:])
        cos = np.cos(dot_product)
        sin = np.sin(dot_product)
        value += complex(cos , sin)
    return value
# end def rhok

# function to compute the structure factor

@cython.boundscheck(False)
@cython.wraparound(False)
def Sk(np.ndarray[double,ndim=2] kvecs, 
    np.ndarray[double, ndim=2] X):
    """ computes structure factor for all k vectors in kList
     and returns a list of them """
#    cdef double kvec
    sk_list = []
    for kvec in kvecs:
        sk_list.append(abs(rhok(kvec, X) * rhok(-kvec , X)) / natom)

#    plt.figure(1)
#    kvecs = legal_kvecs(5,Lb)
    cdef np.ndarray[np.int_t] kmags = np.zeros(np.shape(kvecs)[0],dtype=np.int)
    for i in range(np.shape(kvecs)[0]):
        kmags[i] = np.linalg.norm(kvecs[i,:])
    # sk_list = Sk(kvecs, X)
    cdef np.ndarray[double] sk_arr = np.array(sk_list) # convert to numpy array if not already so
    # average S(k) if multiple k-vectors have the same magnitude
    cdef np.ndarray[np.int_t] unique_kmags = np.unique(kmags)
    cdef np.ndarray[double] unique_sk    = np.zeros(len(unique_kmags))
    cdef int iukmag
    cdef double kmag
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    # end for iukmag

    # visualize
#    plt.plot(unique_kmags[1:],unique_sk[1:])
    return sk_list
# end def Sk

@cython.boundscheck(False)
@cython.wraparound(False)
def BLF( np.ndarray[double,ndim=2] X):

    cdef double BLF = 0.0 # bond length fluctuation
    cdef double mean = 0.0 # mean value of dist
    cdef double sqmean = 0.0 # mean value of dist square
    cdef int i,j
    cdef double dist
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            dist = np.linalg.norm(X[j,:]-X[i,:]) # calculate distance between iat and jat 
            mean += dist/natom
            sqmean += dist**2/natom
        # end for
    # end for
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            BLF += np.sqrt(abs(sqmean-mean**2))/mean * 2/(natom*(natom-1))

        #end for
    # end for

    return BLF
# end BLF
