import matplotlib.pyplot as plt
import numpy as np
natom = 216  # number of atoms

def RDF(X, Lb, dr):

    tot_bin_num = int(np.ceil(Lb / 2 / dr))  # total number of binning windows
    radial = np.zeros(tot_bin_num) # initialize RDF
    bins   = np.zeros(tot_bin_num) # initialize r for plot RDF
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            dist = np.linalg.norm(X[j,:]-X[i,:]) # calculate distance between iat and jat 
            if dist <= Lb / 2: # judge if the distance is smaller than half box length
                ig = int(np.floor(dist / dr))
                radial[ig] += 2 # consider two atoms

    for i in range(tot_bin_num):
        radial[i] = radial[i] * (1 / float(natom)) / (4/3 * np.pi * ((i + 1)**3 - i**3) * dr**3 *float(natom) / (Lb**3)) # normalization
        bins[i] = (i + 0.5) * dr
    
    plt.figure(1)
    plt.plot(bins,radial)
    plt.xlabel('r')
    plt.ylabel('g(r)')

    return radial, bins
# end def RDF

# function computes a list of the legal k-vectors
def legal_kvecs(maxk,Lb):
    kvecs = []
    # calculate a list of legal k vectors
    for n_x in range(maxk):
        for n_y in range(maxk):
            for n_z in range(maxk):
                vector = [n_x, n_y, n_z]
                kvecs.append([2 * np.pi / Lb * n for n in vector])
    return np.array(kvecs)
# end def legal_kvecs

# function to compute the Fourier transform of the density for a given wavevector

def rhok(kvec,X):
    value = 0.0
    #computes \sum_j \exp(i * k \dot r_j)
    for i in range(natom):
        dot_product = np.dot(kvec,X[i,:])
        cos = np.cos(dot_product)
        sin = np.sin(dot_product)
        value += cos + sin * 1j
    return value
# end def rhok

# function to compute the structure factor
def Sk(kvecs, X):
    """ computes structure factor for all k vectors in kList
     and returns a list of them """

    sk_list = []
    for kvec in kvecs:
        sk_list.append(abs(rhok(kvec, X) * rhok(-kvec , X)) / natom)

    plt.figure(1)
    kvecs = legal_kvecs(5,Lb)
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_list = Sk(kvecs, X)
    sk_arr = np.array(sk_list) # convert to numpy array if not already so
    # average S(k) if multiple k-vectors have the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    # end for iukmag

    # visualize
    plt.plot(unique_kmags[1:],unique_sk[1:])
    return sk_list
# end def Sk

def BLF(X):

    BLF = 0 # bond length fluctuation
    mean = 0 # mean value of dist
    sqmean = 0 # mean value of dist square
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            dist = np.linalg.norm(X[j,:]-X[i,:]) # calculate distance between iat and jat 
            mean += dist/natom
            sqmean += dist**2/natom
        # end for
    # end for
    for i in range(natom - 1):
        for j in range(i + 1, natom):
            BLF += np.sqrt(sqmean-mean**2)/mean * 2/(natom*(natom-1))

        #end for
    # end for

    return BLF
# end BLF
