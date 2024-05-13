import numpy as np
from numba import jit, prange
from datetime import datetime
import argparse


def xsf_parser(filename):

    with open(filename, "r") as main:
        line = main.readline()
        while line and line.strip() != "BEGIN_DATAGRID_3D_UNKNOWN":
            line = main.readline()  # go to data block

        n_size = np.array([int(x) for x in main.readline().split()])

        origin = np.array([float(x) for x in main.readline().split()])

        vecs = np.zeros((3, 3))        
        for i in range(3):
            vecs[i] = np.array([float(x) for x in main.readline().split()])

        W = []
        for token in main.read().split():
            try:
                W.append(float(token))
            except ValueError:
                break
            
        assert len(W) == np.prod(n_size)
        W = np.array(W)

        print("File", filename, "was scanned successfully")

    return W, n_size, origin, vecs


def normalize(W):
    norm = np.sum(W**2)
    
    return W/np.sqrt(norm)


@jit(nopython=True) 
def size_reduction(W1, W2, n_size, vecs, r_center, r_cut):
    
    n_tot = n_size[0] * n_size[1] * n_size[2]
    
    W1_new = []
    W2_new = []
    r_new = []
    
    norm_1 = 0.0
    norm_2 = 0.0
    
    for i in range(n_tot):
        c = int(i // (n_size[0] * n_size[1]))
        a = int((i - (n_size[0] * n_size[1]) * c) % (n_size[0]))
        b = int((i - (n_size[0] * n_size[1]) * c) // (n_size[0]))
        
        r = (vecs[0] * a) / n_size[0] + (vecs[1] * b) / n_size[1] + (vecs[2] * c) / n_size[2]
        
        if(np.linalg.norm(r_center - r) < r_cut):
            W1_new.append(W1[i])
            W2_new.append(W2[i])
            r_new.append(r)
            
            norm_1 += W1[i]*W1[i]
            norm_2 += W1[i]*W1[i]
                    
    W1_new = np.array(W1_new)
    W2_new = np.array(W2_new)
    
    n_tot_new = W1_new.shape[0]  
    print("Size reduction  leads to W1 and W2 norms: ", norm_1, norm_2 )
    print("Size reduction factor: ", int(100 * (n_tot - n_tot_new)/n_tot), "%")
    print("WARNING: norms after size reduction should not be far from  1!!!")
            
    return r_new, W1_new, W2_new 



@jit(nopython=True, parallel=True)
def compute_Coulomb(mc_steps, n_tot, W1, W2, r):

    coulomb_U = 0.0
    coulomb_V = 0.0
    coulomb_J = 0.0

    for n in prange(mc_steps):
        local_coulomb_U = 0.0
        local_coulomb_V = 0.0
        local_coulomb_J = 0.0

        i = np.random.randint(0, n_tot)
        j = np.random.randint(0, n_tot)
        
        if(i != j):
            distance = np.linalg.norm(r[i] - r[j])
            local_coulomb_U += (W1[i] * W1[i]) * (W1[j] * W1[j]) / distance
            local_coulomb_V += (W1[i] * W1[i]) * (W2[j] * W2[j]) / distance
            local_coulomb_J += (W1[i] * W2[i]) * (W1[j] * W2[j]) / distance


        coulomb_U += local_coulomb_U
        coulomb_V += local_coulomb_V
        coulomb_J += local_coulomb_J

    return  14.3948 * (n_tot * n_tot / mc_steps) * np.array([coulomb_U, coulomb_V, coulomb_J])


def main():

    #play with this parameter to reach the required accuracy
    mc_steps = int(1E9)
    # set the center r_center for size reduction
    # set the cutoff distance to increase the accuracy of MC sampling
    # keep in mind that norm_1 and norm_2 should be close to 1 after size reduction!!!
    r_center = np.array([0, 0, 9.5176]) 
    r_cut = 25

    print("Program Wannier_Hund.x v.2.0 starts on  ", datetime.now())
    print('=' * 69)

    parser = argparse.ArgumentParser(prog='Wannier_Coulomb.py', usage='%(prog)s WF1.xsf WF2.xsf')
    parser.add_argument("WF1")
    parser.add_argument("WF2")
    args = parser.parse_args()


    W1, n_size, origin, vecs = xsf_parser(args.WF1)
    W2, n_size2, origin2, vecs2 = xsf_parser(args.WF2)

    assert((n_size == n_size2).all())
    assert((origin == origin2).all())
    assert((vecs == vecs2).all())

    W1 = normalize(W1)
    W2 = normalize(W2)

    print("Dimensions are:", n_size[0], n_size[1], n_size[2])
    print("Origin is:", '{:.3f}'.format(origin[0]), '{:.3f}'.format(origin[1]), '{:.3f}'.format(origin[2]))
    
    print("Span_vectors are:")
    for i in range(3):
        print('{:.3f}'.format(vecs[i,0]), '{:.3f}'.format(vecs[i,1]), '{:.3f}'.format(vecs[i,2])) 
        
    r_new, W1_new, W2_new  = size_reduction(W1, W2, n_size, vecs, r_center, r_cut)
    r_new = np.array(r_new)
    n_tot_new = r_new.shape[0] 

    coulomb = compute_Coulomb(mc_steps, n_tot_new, W1_new, W2_new, r_new) 

    print("Coulomb_U: ", '{:.4f}'.format(coulomb[0]), " eV")
    print("Coulomb_V: ", '{:.4f}'.format(coulomb[1]), " eV")
    print("Coulomb_J: ", '{:.4f}'.format(coulomb[2]), " eV")

    print('\n')
    print(f'This run was terminated on: {datetime.now()}')
    print(f'JOB DONE')
    print('=' * 69)


if __name__ == '__main__':
    main()