import sys
sys.path.append('../')
import os
import torch
import numpy as np
import time
import pickle
import logging
import argparse
from models import SBN
from utils import summary, mcmc_treeprob, get_support_from_mcmc

def run_sem(args, stat):
    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.SEMVR_learn(args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter,miniter=args.miniter, biter=args.biter, batch=args.batch, abstol=args.abstol, ema_rate=args.ema_rate, vr=False, report_kl=args.report_kl, monitor=args.monitor)
    stat['logp'] = logp
    if args.report_kl:
        stat['kldiv'] = kldiv
        print('{} SEM ema_rate:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch, kldiv[-1]))
    else:
        print('{} SEM ema_rate:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch))
    torch.save(model.CPDs, args.ckptpath)

    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.SEMVR_learn(args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter,miniter=args.miniter, biter=args.biter, batch=args.batch, abstol=args.abstol, ema_rate=args.ema_rate,vr=False, monitor=args.monitor,report_kl=args.report_kl, alpha=0.0001)
    stat['logp-alpha'] = logp
    if args.report_kl:
        stat['kldiv-alpha'] = kldiv
        print('{} SEM-alpha ema_rate:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch, kldiv[-1]))
    else:
        print('{} SEM-alpha ema_rate:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch))
    torch.save(model.CPDs, args.ckptpath.replace('SEM', 'SEMalpha'))

def run_semvr(args, stat):
    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.SEMVR_learn(args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter, miniter=args.miniter, biter=args.biter, batch=args.batch, abstol=args.abstol, ema_rate=args.ema_rate,report_kl=args.report_kl, monitor=args.monitor)
    stat['logp'] = logp
    if args.report_kl:
        stat['kldiv'] = kldiv
        print('{} SEMVR ema_rate:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch, kldiv[-1]))
    else:
        print('{} SEMVR ema_rate:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch))
    torch.save(model.CPDs, args.ckptpath)

    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.SEMVR_learn(args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter, miniter=args.miniter, biter=args.biter, batch=args.batch, abstol=args.abstol, ema_rate=args.ema_rate,report_kl=args.report_kl, monitor=args.monitor, alpha=0.0001)
    stat['logp-alpha'] = logp
    if args.report_kl:
        stat['kldiv-alpha'] = kldiv
        print('{} SEMVR-alpha ema_rate:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch, kldiv[-1]))
    else:
        print('{} SEMVR-alpha ema_rate:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.ema_rate, args.biter, args.batch))
    torch.save(model.CPDs, args.ckptpath.replace('SEMVR', 'SEMVRalpha'))


def run_em(args, stat):
    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.EM_learn(args.tree_dict, args.tree_names, args.tree_wts, miniter=args.miniter, maxiter=args.maxiter, abstol=args.abstol,report_kl=args.report_kl,monitor=args.monitor)
    stat['logp'] = logp
    if args.report_kl:
        stat['kldiv'] = kldiv
        print('{} EM finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), kldiv[-1]))
    else:
        print('{} EM finished'.format(time.asctime(time.localtime(time.time()))))
    torch.save(model.CPDs, args.ckptpath)

    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.EM_learn(args.tree_dict, args.tree_names, args.tree_wts, miniter=args.miniter, maxiter=args.maxiter, abstol=args.abstol,report_kl=args.report_kl,monitor=args.monitor, alpha=0.0001)
    stat['logp-alpha'] = logp
    if args.report_kl:
        stat['kldiv-alpha'] = kldiv
        print('{} EM-alpha finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), kldiv[-1]))
    else:
        print('{} EM-alpha finished'.format(time.asctime(time.localtime(time.time()))))
    torch.save(model.CPDs, args.ckptpath.replace('EM', 'EMalpha'))


def run_svrg(args, stat):
    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.SGA_learn(args.lr, args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter,miniter=args.miniter, biter=args.biter, batch=args.batch, abstol=args.abstol,report_kl=args.report_kl, monitor=args.monitor, vr=True)
    stat['logp'] = logp
    if args.report_kl:
        stat['kldiv'] = kldiv
        print('{} SVRG lr:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.lr, args.biter, args.batch, kldiv[-1]))
    else:
        print('{} SVRG lr:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.lr, args.biter, args.batch))
    torch.save(model.CPDs, args.ckptpath)

def run_sga(args, stat):
    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.SGA_learn(args.lr, args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter,miniter=args.miniter, biter=args.biter, batch=args.batch, abstol=args.abstol,report_kl=args.report_kl, monitor=args.monitor, vr=False)
    stat['logp'] = logp
    if args.report_kl:
        stat['kldiv'] = kldiv
        print('{} SGA lr:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.lr, args.biter, args.batch, kldiv[-1]))
    else:
        print('{} SGA lr:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.lr, args.biter, args.batch))
    torch.save(model.CPDs, args.ckptpath)

def run_ga(args, stat):
    model = SBN(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.emp_tree_freq)
    logp, kldiv = model.GA_learn(args.lr, args.tree_dict, args.tree_names, args.tree_wts, maxiter=args.maxiter,miniter=args.miniter, abstol=args.abstol,report_kl=args.report_kl, monitor=args.monitor)
    stat['logp'] = logp
    if args.report_kl:
        stat['kldiv'] = kldiv
        print('{} GA lr:{}, biter:{}, batch:{} finished. achieves the KL divergence: {}.'.format(time.asctime(time.localtime(time.time())), args.lr, args.biter, args.batch, kldiv[-1]), flush=True)
    else:
        print('{} GA lr:{}, biter:{}, batch:{} finished.'.format(time.asctime(time.localtime(time.time())), args.lr, args.biter, args.batch), flush=True)
    torch.save(model.CPDs, args.ckptpath)

def add_arguments():

    parser = argparse.ArgumentParser(description='The arguments for SBN')
    parser.add_argument('--method', type=str, default='SEMVR', help='the method for SBN probability estimation')
    parser.add_argument('--dataset', type=int, required=True, help='DS1-DS8')
    parser.add_argument('--repo', type=int, required=True, help='repo1-repo10')
    parser.add_argument('--biter', type=int, default=1000, help='batch iteration. The # of iterations per loop')
    parser.add_argument('--batch', type=int, default=1, help='the number of samples used in one biter')
    parser.add_argument('--ema_rate', type=float, default=0.99, help='The ema_rate for exponential moving average for SEMVR, SEM')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate for SVRG, SGA')
    parser.add_argument('--abstol', type=float, default=1e-5, help='stopping criteria')
    parser.add_argument('--maxiter', type=int, default=300, help='the maximum epoch')
    parser.add_argument('--miniter',type=int, default=50, help='the minimum epoch')
    parser.add_argument('--monitor', default=False, action='store_true')
    parser.add_argument('--report_kl', default=False, action='store_true')
    args = parser.parse_args()

    return args

def main():
    args = add_arguments()
    date = str(time.asctime(time.localtime(time.time()))).replace(' ', '-')
    args.result_folder = os.path.join('results', 'DS{}'.format(args.dataset), 'repo{}'.format(args.repo))
    os.makedirs(args.result_folder, exist_ok=True)
    args.statpath = args.result_folder + '/{}_{}.pkl'.format(args.method, date)
    args.ckptpath = args.result_folder + '/{}_{}.pt'.format(args.method, date)

    print('Training with parameters {}'.format(args), flush=True)
    if args.report_kl:
        print('{} dataset DS{}, golden run loading ...'.format(time.asctime(time.localtime(time.time())), args.dataset), flush=True)
        tree_dict_total, tree_names_total, tree_wts_total = summary('DS{}'.format(args.dataset), '../../data/raw_data_DS1-4/')
        args.emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
    else:
        args.emp_tree_freq = None
    
    print("{} loading DS{} rep {} ...".format(time.asctime(time.localtime(time.time())), args.dataset, args.repo), flush=True)
    args.tree_dict, args.tree_names, args.tree_wts = mcmc_treeprob('../../data/short_run_data_DS1-4/DS' + str(args.dataset) + '/rep_{}/DS'.format(args.repo) + str(args.dataset) + '.trprobs', 'nexus')
    args.tree_wts = np.array(args.tree_wts)/sum(args.tree_wts) 
    args.taxa = args.tree_dict[args.tree_names[0]].get_leaf_names()
    args.rootsplit_supp_dict, args.subsplit_supp_dict = get_support_from_mcmc(args.taxa, args.tree_dict, args.tree_names)

    print("{} DS{} rep {}: {} unique trees".format(time.asctime(time.localtime(time.time())), args.dataset, args.repo, len(args.tree_wts)), flush=True)

    stat = dict()

    method_dict = {'SGA': run_sga, 'SEM': run_sem, 'SVRG': run_svrg, 'EM': run_em, 'SEMVR': run_semvr, 'GA': run_ga}
    method_dict[args.method](args, stat)
    
    with open(args.statpath, 'wb') as f:
        pickle.dump(stat, f)

if __name__ == '__main__':
    main()
    sys.exit(0)