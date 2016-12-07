#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import copy

Lb         = [1.60531e-9/3,1.60531e-9/3,1.60531e-9/3] # box size, Stillinger-Weber compressed domain
Pi         = np.pi
srPi       = np.sqrt(Pi)
kB         = 1.381e-23    # k-Boltzmann (J/K)
eps0       = 8.854187e-12 # permittivity constant (F/m = C^2/J m)
ec         = 1.60217646e-19 # elementary charge (C)
ao         = 0.053e-09 # Bohr radius
massSi     = 46.637063e-27

# Stillinger-Weber Constants
sigmaSi    = 2.0951e-10
A          = 7.049556277
B          = 0.6022245584
psi        = 4.0
qsi        = 0.0
al         = 1.8
lambdaSi   = 21.0
gamma      = 1.2

# Initialize parameters

Nc1        = 3 # x dimension # cells
Nc2        = 3 # y dimension # cells
Nc3        = 3 # z dimension # cells
Natm       = Nc1*Nc2*Nc3*8 # total atom count, 8 atoms per diamond cubic cell
rc         = al*sigmaSi # Stillinger-Weber Cutoff radius
ml1        = 50 # max defalut neighbor count (per atom)
print("rc = %d" % rc)

PE         = np.zeros(Natm)
KE         = np.zeros(Natm)

F          = np.zeros((Natm,3)) # force arrays
V          = np.zeros((Natm,3)) # velocity arrays

# random seed
seed       = 1
np.random.seed(seed)

# Initiliaze crystal
uc         = np.zeros((8,3))
X          = np.zeros((Natm,3)) # position arrays

# Values are for perfect diamond cubic in [001] direction
# uc contains x,y,z position of each atom in the cubic cell

uc[0,0] = 0.00         
uc[0,1] = 0.00            
uc[0,2] = 0.00
uc[1,0] = 0.00         
uc[1,1] = 0.50*Lb[1]    
uc[1,2] = 0.50*Lb[2]
uc[2,0] = 0.250*Lb[0] 
uc[2,1] = 0.250*Lb[1] 
uc[2,2] = 0.750*Lb[2]
uc[3,0] = 0.250*Lb[0]
uc[3,1] = 0.750*Lb[1]  
uc[3,2] = 0.250*Lb[2]
uc[4,0] = 0.50*Lb[0]
uc[4,1] = 0.0*Lb[1] 
uc[4,2] = 0.50*Lb[2]
uc[5,0] = 0.50*Lb[0]
uc[5,1] = 0.50*Lb[1]  
uc[5,2] = 0.0*Lb[2]
uc[6,0] = 0.750*Lb[0]  
uc[6,1] = 0.250*Lb[1]
uc[6,2] = 0.250*Lb[2]
uc[7,0] = 0.750*Lb[0]
uc[7,1] = 0.750*Lb[1] 
uc[7,2] = 0.750*Lb[2]

# map cubic cells to the whole domain, probably a 3*3*3 cell grid
n = 0
for k in range(Nc3):
	for j in range(Nc2):
		for i in range(Nc1):
			for l in range(8):
				
				X[n,0] = Lb[0] * i + uc[l,0]
				X[n,1] = Lb[1] * j + uc[l,1]
				X[n,2] = Lb[2] * k + uc[l,2]
				n += 1 # count # atoms places
			# end for
		# end for
	# end for
# end for

Lb[0] = Lb[0] * Nc1
Lb[1] = Lb[1] * Nc2
Lb[2] = Lb[2] * Nc3

print("Si atoms: %d" %n)
print("Box Dimensions: %f * %f * %f (nm)" % (Lb[0]*1e9,Lb[1]*1e9,Lb[2]*1e9))

maxx = np.amax(X[:,0])
maxy = np.amax(X[:,1])


dis = np.zeros(3)

nlist = np.zeros((Natm,ml1))
nlistcnt = np.zeros(Natm, dtype=np.int8)

for i in range(Natm):
	cnt1 = 0

	for j in range(i,Natm): # Sorting from high to low to make 3 body pointer list pretty
	# checking one dimension at a time in order to optimize excution time
		dis[0] = np.abs(X[i,0]-X[j,0])

		if dis[0] > Lb[0]/2:
			dis[0] = np.abs(dis[0]-Lb[0])
		# end if

		if dis[0] > rc:
			continue
		# end if

		dis[1] = np.abs(X[i,1]-X[j,1])

		if dis[1] > Lb[1]/2:
			dis[1] = np.abs(dis[1]-Lb[1])
		# end if

		if dis[1] > rc:
			continue
		# end if		

		dis[2] = np.abs(X[i,2]-X[j,2])

		if dis[2] > Lb[2]/2:
			dis[2] = np.abs(dis[2]-Lb[2])
		# end if

		if dis[2] > rc:
			continue
		# end if

		# If within sphere, then count as neighbor
		dist = np.sqrt(dis[0]**2 + dis[1]**2 + dis[2]**2)
		if dist <= rc:
			# If within sphere, then count as neighbor


			if cnt1 > ml1:
				print("neighbor list length past limit, increase ml1")
				exit()
			# end if

			nlist[i,cnt1] = j
			cnt1 = cnt1 + 1 # increment counter for current particle's # of neighbors
		# end if
	# end for

	nlistcnt[i] = cnt1 # # of neighbors in 2 body list for particle # i
# END FOR

# create full neighborlist from 2 body list
# (Full, as in, includes duplicate pairs)
nlistf = copy.deepcopy(nlist)
nlistcntf = copy.deepcopy(nlistcnt)

for i in range(Natm):
	for j in range(nlistcnt[i]):
		nlistf[nlist[i,j],nlistcntf[nlist[i,j]]] = i+1
		nlistcntf[nlist[i,j]] += 1
	# end for
# end for

# SWPotential
dis = np.zeros(3)
disij = np.zeros(3)
disik = np.zeros(3)
disjk = np.zeros(3)

U2 = np.zeros((Natm,np.amax(nlistcnt)))

r2 = np.zeros((Natm,np.amax(nlistcnt))) # stored distances
dis2 = np.zeros((Natm,np.amax(nlistcnt),3))
# 2 body potential
for i in range(Natm):
	for j in range(nlistcnt[i]):
		dis = X[nlist[i,j],:] - X[i,:] # vector from i to nlist[i,j]
		for k in range(3): # loop through x,y,z distances and move to nearest particle image
			if dis[k] > Lb[k]/2:
				dis[k] = dis[k]-Lb[k]
			elif -dis[k] > Lb[k]/2:
				dis[k] = dis[k]+Lb[k]
			# end if
			dis2[i,j,k] = dis[k]/sigmaSi
		# end for
		r = np.sqrt(np.dot(dis,dis))/sigmaSi # temporary for immediate calculations
		print(r)
		r2[i,j] = copy.deepcopy(r) # update stored interaction distance

		Utemp = A*(B*r**(-psi)-r**(-qsi))*np.exp((r-al)**(-1))
		# print(Utemp)
		U2[i,j] = copy.deepcopy(Utemp) # Store potential energy per neighbor pair, same scheme as neighbotlist
	# end for
# end for

# 3 body potential
cnt3 = 0
for i in range(np.amax(nlistcnt)-1):
	cnt3 = cnt3+i
# end for

U3 = np.zeros((Natm,cnt3,3)) # allocate U3 space for each triplet
                           # rank 3 will store hijk, hijk, hikj
dis3 = np.zeros((Natm,cnt3,3,3)) # interaction displacement vectors disij,disik,disji
							   # rank 3 is vector designation, rank 4 is vector component (x,y,z)
r3 = np.zeros((Natm,cnt3,3)) # interaction displacement magnitude rij, rik, rji
cos3 = np.zeros((Natm,cnt3,3)) # cosjik, cosijk, cosikj terms for each triplet


for i in range(Natm): # primary atom
	
	for j in range(nlistcnt[i]-1): # Second term
		# print("j = %d" %j)
		cnt3 = 0
		for k in range(j,nlistcnt[i]): # third atom, avoiding same pairs
			# print("k = %d" %k)
			disij = X[nlist[i,j],:]-X[i,:] # vectors from i to j, i to k, and j to k
			disik = X[nlist[i,k],:]-X[i,:]
			disjk = X[nlist[i,k],:]-X[nlist[i,j],:]


			for l in range(3): # loop through x,y,z distances and move to nearest particle image
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
				if disjk[l] > Lb[l]/2:
					disjk[l] = disjk[l] - Lb[l]
				elif -disjk[l] > Lb[l]/2:
					disjk[l] = disjk[l] + Lb[l]
				# end if
				dis3[i,cnt3,0,l] = disij[l]/sigmaSi
				dis3[i,cnt3,1,l] = disik[l]/sigmaSi
				dis3[i,cnt3,2,l] = disjk[l]/sigmaSi
			# end for

			rij = np.sqrt(disij[0]**2+disij[1]**2+disij[2]**2)
			rik = np.sqrt(disik[0]**2+disik[1]**2+disik[2]**2)
			rjk = np.sqrt(disjk[0]**2+disjk[1]**2+disjk[2]**2)

			dot1 = np.dot(disij,disik)
			dot2 = np.dot(disjk,disik)
			dot3 = np.dot(-disij,disjk)

			cosjik = dot1/(rij*rik)
			cosikj = dot2/(rjk*rik)
			cosijk = dot3/(rij*rjk)

			cos3[i,cnt3,0] = cosjik
			cos3[i,cnt3,1] = cosijk
			cos3[i,cnt3,2] = cosikj

			rij = rij/sigmaSi
			rik = rik/sigmaSi
			rjk = rjk/sigmaSi

			r3[i,cnt3,0] = rij
			r3[i,cnt3,1] = rik
			r3[i,cnt3,2] = rjk

			for l in range(3):
				dis3[i,cnt3,0,l] = dis3[i,cnt3,0,l] / r3[i,cnt3,0]
				dis3[i,cnt3,1,l] = dis3[i,cnt3,1,l] / r3[i,cnt3,1]
				dis3[i,cnt3,2,l] = dis3[i,cnt3,2,l] / r3[i,cnt3,2]
			# end for

			if rij > al:
				rij = al-1e-10
			# end if
			if rik > al:
				rik = al-1e-10
			# end if
			if rjk > al:
				rjk = al-1e-10
			# end if

			hjik = lambdaSi*np.exp(gamma)*np.exp((rij-al)**(-1)+(rik-al)**(-1))*(cosjik+1./3.)**2
			hijk = lambdaSi*np.exp(gamma)*np.exp((rjk-al)**(-1)+(rij-al)**(-1))*(cosijk+1./3.)**2
			hikj = lambdaSi*np.exp(gamma)*np.exp((rik-al)**(-1)+(rjk-al)**(-1))*(cosikj+1./3.)**2
			# print(cnt3)
			U3[i,cnt3,0] = hjik
			U3[i,cnt3,1] = hijk
			U3[i,cnt3,2] = hikj
			cnt3 += 1 # keep track of # triplets for each primary atom
		# end for
	# end for
# end for

U = np.sum(U2) + np.sum(U3)

U_average = U/Natm
U2_average = np.sum(U2)/Natm
U3_average = np.sum(U3)/Natm

print(U_average)

print("Average potential per atom: %d" % U_average)
print("Average 2 body potential: %d" % U2_average)
print("Average 3 body potential: %d" % U3_average)

