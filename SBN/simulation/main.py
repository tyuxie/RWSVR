import os
import numpy as np
from collections import defaultdict
from models import SBN
from utils import generate, get_support_from_mcmc
import argparse
import pickle
import torch
from multiprocessing import Pool
from copy import deepcopy

EPS = np.finfo(float).eps

def get_args():
    '''Add argument by command line'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', default=0.004, type=float)
    parser.add_argument('--n_trees', default=500, type=int)
    parser.add_argument('--k_all', default=1000, type=int)
    parser.add_argument('--method', default='EM', choices=['EM', 'SEM', 'SEMVR', 'SGA', 'SVRG'])
    parser.add_argument('--emp',default=False, action='store_true', help='use the empirical probability')
    parser.add_argument('--niters', default=200, type=int)
    args = parser.parse_args()
    return args

def main(args):
    taxa = list('ABCDEFGH')
    all_tree = generate(taxa)

    tree_space_cap = len(all_tree)
    os.makedirs('results/simulation_{}/'.format(args.k_all) + args.method, exist_ok=True)
    
    samp_freq = np.random.dirichlet(args.beta*np.ones(tree_space_cap))
    sorted_samp_freq = np.argsort(samp_freq)[::-1]

    samp_trees = [all_tree[i] for i in sorted_samp_freq[:args.n_trees]]

    if args.emp:
        emp_tree_freq = {tree:samp_freq[sorted_samp_freq[i]] for i, tree in enumerate(samp_trees)}
    else:
        emp_tree_freq = {}
        for i, tree in enumerate(all_tree):
            emp_tree_freq[tree] = samp_freq[i]

    tree_dict = {'tree_{}'.format(i+1): tree for i, tree in enumerate(samp_trees)}
    tree_names = list(tree_dict.keys())
    tree_wts = np.array([emp_tree_freq[tree] for tree in tree_dict.values()])
    tree_wts = list(tree_wts / np.sum(tree_wts))
    rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict, tree_names, tree_wts)
    
    if args.method == 'EM':
        model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict, emp_tree_freq)
        logp, kldivs = model.EM_learn(tree_dict, tree_names, tree_wts, maxiter=args.niters, miniter=args.niters, abstol=1e-05, monitor=True, start_from_uniform=False, report_kl=True, alpha=0.0)
        results = {'logp': logp, 'kldivs': kldivs, 'size': len(tree_dict)}
        save_to_path = 'results/simulation_{}/EM/'.format(args.k_all)+'beta_'+str(args.beta)+'_k_'+str(args.n_trees)+'.pkl'
    if args.method == 'SEM':
        model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict, emp_tree_freq)
        logp, kldivs = model.SEMVR_learn(tree_dict, tree_names, tree_wts, maxiter=args.niters, miniter=args.niters, monitor=True, ema_rate=0.999, start_from_uniform=False, biter=args.k_all, alpha=0.0, report_kl=True)
        results = {'logp': logp, 'kldivs': kldivs, 'size': len(tree_dict)}
        save_to_path = 'results/simulation_{}/SEM/'.format(args.k_all)+'beta_'+str(args.beta)+'_k_'+str(args.n_trees)+'.pkl'
    if args.method == 'SEMVR':
        model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict, emp_tree_freq)
        logp, kldivs = model.SEMVR_learn(tree_dict, tree_names, tree_wts, maxiter=args.niters, miniter=args.niters, monitor=True, ema_rate=0.99, start_from_uniform=False, biter=args.k_all, alpha=0.0, report_kl=True, vr=True)
        results = {'logp': logp, 'kldivs': kldivs, 'size': len(tree_dict)}
        save_to_path = 'results/simulation_{}/SEMVR/'.format(args.k_all)+'beta_'+str(args.beta)+'_k_'+str(args.n_trees)+'.pkl'
    if args.method == 'SGA':
        model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict, emp_tree_freq)
        logp, kldivs = model.SGA_learn(0.0001, tree_dict, tree_names, tree_wts, batch=1, maxiter=args.niters, miniter=args.niters, abstol=1e-05, biter=args.k_all, ar=0.75, af=10, start_from_uniform=False, monitor=True, report_kl=True, vr=False)
        results = {'logp': logp, 'kldivs': kldivs, 'size': len(tree_dict)}
        save_to_path = 'results/simulation_{}/SGA/'.format(args.k_all)+'beta_'+str(args.beta)+'_k_'+str(args.n_trees)+'.pkl'
    if args.method == 'SVRG':
        model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict, emp_tree_freq)
        logp, kldivs = model.SGA_learn(0.001, tree_dict, tree_names, tree_wts, batch=1, maxiter=args.niters, miniter=args.niters, abstol=1e-05, biter=args.k_all, ar=0.75, af=10, start_from_uniform=False, monitor=True, report_kl=True, vr=True)
        results = {'logp': logp, 'kldivs': kldivs, 'size': len(tree_dict)}
        save_to_path = 'results/simulation_{}/SVRG/'.format(args.k_all)+'beta_'+str(args.beta)+'_k_'+str(args.n_trees)+'.pkl'

    with open(save_to_path, 'wb') as f:
        pickle.dump(results, f)
    
    torch.save(model.CPDs, save_to_path.replace('.pkl', '.pt'))

if __name__ == '__main__':
    args = get_args()       
    main(args)

