from sortedcontainers import SortedSet

def periodic_disp(pos1, pos2):
  disij = pos2 - pos1
  for l in range(3):
    if disij[l] > Lb[l]/2:
      disij[l] -= Lb[l]
    elif -disij[l] > Lb[l]/2:
      disij[l] += Lb[l]

def twobody_sanity(atom1, x, rs, rc):
    V_pairs = SortedSet([])
    l_pairs = SortedSet([])
    for atom2 in range(atom1):
      dist = np.linalg.norm(periodic_disp(x[atom1], x[atom2]))
      if (dist <= rc):
        V_pairs.add(atom2)
      if (dist <= rs):
        l_pairs.add(atom2)
    return (V_pairs, l_pairs)

def threebody_sanity(atom1, x, rs, rc):
    return None
    others1 = x[:atom1]
    others2 = x[(atom1+1):]
    others = np.concatenate(others1. others2, axis = 0)
    for atom2 in others1:
      for atom2 in others2:
      
    

        
