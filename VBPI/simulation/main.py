import sys
import numpy as np
from models import SBN_VI_EMP
from utils import get_support_from_samples
import pandas as pd 
import dill
import argparse
import os


def get_args():
    '''Add argument by command line'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--nparticles',type=int, default=10)
    parser.add_argument('--method',type=str, default='rwsvr')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    path = f'results/{args.method}-{args.nparticles}'
    os.makedirs(path, exist_ok=True)
    
    # load empirical data
    with open('../../data/simulation/simulation_emp_tree_freq.dill', 'rb') as readin:
        emp_tree_freq = dill.load(readin, encoding='iso-8859-1')
    
    tree_dict, tree_wts = zip(*emp_tree_freq.items())
    tree_dict = {'tree_'+str(i): tree for i, tree in enumerate(tree_dict)}
    tree_names = ['tree_'+str(i) for i in range(len(tree_wts))]

    taxa = 'ABCDEFGH'
    # get subsplit support
    rootsplit_supp_dict, subsplit_supp_dict = get_support_from_samples(taxa, tree_dict, tree_names)
    # vbpi model
    sbn = SBN_VI_EMP(taxa, emp_tree_freq, rootsplit_supp_dict, subsplit_supp_dict)

    # run!
    if args.method == 'vimco':
        test_kl_div, test_lower_bound = sbn.vimco(0.001, n_particles=args.nparticles, sgd_solver='adam')
    elif args.method == 'rws':
        test_kl_div, test_lower_bound = sbn.rws(0.002, n_particles=args.nparticles, sgd_solver='amsgrad', sample_particle=False)
    elif args.method == 'rwsvr':
        test_kl_div, test_lower_bound = sbn.rwsvr(0.002, n_particles=args.nparticles, sgd_solver='amsgrad', sample_particle=False)
    
    np.save(os.path.join(path, 'kl_div.npy'), test_kl_div)
    np.save(os.path.join(path, 'lower_bound.npy'), test_lower_bound)
    
        
if __name__ == '__main__':
    main()