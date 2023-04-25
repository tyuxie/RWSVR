import argparse
import os

from dataManipulation import *
from utils import summary, summary_raw, get_support_from_mcmc
from vbpi import VBPI
import time
import numpy as np
import datetime
import logging
import torch 

def get_args():
    '''Add argument by command line'''
    parser = argparse.ArgumentParser()

    ######### Data arguments
    parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
    parser.add_argument('--empFreq', default=False, action='store_true', help='emprical frequence for KL computation')
    parser.add_argument('--rws', default=False, action='store_true')
    parser.add_argument('--rwsvr', default=False, action='store_true')

    ######### Model arguments
    parser.add_argument('--flow_type', type=str, default='identity', help=' identity | planar | realvnp ')
    parser.add_argument('--psp', type=bool, default=True, help=' turn on psp branch length feature, default=True')
    parser.add_argument('--nf', type=int, default=2, help=' branch length feature embedding dimension ') 
    parser.add_argument('--sh', type=list, default=[100], help=' list of the hidden layer sizes for permutation invariant flow ')
    parser.add_argument('--Lnf', type=int, default=5, help=' number of layers for permutation invariant flow ')


    ######### Optimizer arguments
    parser.add_argument('--stepszTree', type=float, default=0.001, help=' stepsz for tree topology parameters ')
    parser.add_argument('--stepszBranch', type=float, default=0.001, help=' stepsz for branch length parameters ')
    parser.add_argument('--maxIter', type=int, default=200000, help=' number of iterations for training, default=200000') 
    parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule, default=0.001')
    parser.add_argument('--nwarmStart', type=float, default=100000, help=' number of warm start iterations, default=100000')
    parser.add_argument('--nParticle', type=int, default=10, help='number of particles for variational objectives, default=10')
    parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75') 
    parser.add_argument('--af', type=int, default=20000, help='step size anneal frequency, default=20000')
    parser.add_argument('--tf', type=int, default=100, help='monitor frequency during training, default=1000')
    parser.add_argument('--lbf', type=int, default=1000, help='lower bound test frequency, default=5000')
    ######### vr arguments
    parser.add_argument('--maxFIter', type=int, default=2000, help=' number of iterations for training in RWSVR, default=2000')
    parser.add_argument('--biter', type=int, default=100)
    parser.add_argument('--batch_F', type=int, default=1000)

    args = parser.parse_args()
    if args.rws:
        args.result_folder = 'results/' + args.dataset + '/rws_' 
    elif args.rwsvr:
        args.result_folder = 'results/' + args.dataset + '/rwsvr_' 
    else:
        args.result_folder = 'results/' + args.dataset + '/vimco_' 
    if args.flow_type != 'identity':
        args.result_folder = args.result_folder + args.flow_type + '_' +  str(args.Lnf)+ '/nparticle_' +str(args.nParticle)
    else:
        args.result_folder = args.result_folder + 'base' + '/nparticle_' +str(args.nParticle)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    args.log_path = args.result_folder + '/training.log'
    args.logger = logging.getLogger()
    args.logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(args.log_path)
    filehandler.setLevel(logging.INFO)
    args.logger.addHandler(filehandler)

    return args

def get_data(args):
    ###### Load Data
    args.logger.info('\nLoading Data set: {} ......'.format(args.dataset))
    run_time = -time.time()

    tree_dict_ufboot, tree_names_ufboot = summary_raw(args.dataset, '../../data/ufboot_data_DS1-4/')
    data, taxa = loadData('../../data/hohna_datasets_fasta/' + args.dataset + '.fasta', 'fasta')
    args.data = data
    args.taxa = taxa

    run_time += time.time()
    args.logger.info('Support loaded in {:.1f} seconds'.format(run_time))

    if args.empFreq:
        args.logger.info('\nLoading empirical posterior estimates ......')
        run_time = -time.time()
        tree_dict_total, tree_names_total, tree_wts_total = summary(args.dataset, '../../data/raw_data_DS1-4/') 
        emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)} 
        run_time += time.time()
        args.logger.info('Empirical estimates from MrBayes loaded in {:.1f} seconds'.format(run_time))
    else:
        emp_tree_freq = None
    args.emp_tree_freq = emp_tree_freq

    rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_ufboot, tree_names_ufboot)  
    del tree_dict_ufboot, tree_names_ufboot
    args.rootsplit_supp_dict = rootsplit_supp_dict
    args.subsplit_supp_dict = subsplit_supp_dict 

def get_model(args):
    model = VBPI(args.taxa, args.rootsplit_supp_dict, args.subsplit_supp_dict, args.data, pden=np.ones(4)/4., subModel=('JC', 1.0),
                    emp_tree_freq=args.emp_tree_freq, feature_dim=args.nf, hidden_sizes=args.sh, num_of_layers_nf=args.Lnf,
                    flow_type=args.flow_type, logger=args.logger)   

    args.logger.info('VBPI running, Flow type: {}'.format(args.flow_type))
    return model


def main():
    args = get_args()
    get_data(args)
    model = get_model(args)
    if args.rwsvr:
        model.rws_vr_learn({'tree':args.stepszTree,'branch':args.stepszBranch}, maxiter=args.maxFIter, test_freq=args.tf, lb_test_freq=args.lbf, n_particles=args.nParticle, anneal_freq=args.af, anneal_rate=args.ar, init_inverse_temp=args.invT0, biter=args.biter, batch_F=args.batch_F, warm_start_interval=args.nwarmStart)
    else:
        model.learn({'tree':args.stepszTree,'branch':args.stepszBranch}, args.maxIter, test_freq=args.tf, lb_test_freq=args.lbf, n_particles=args.nParticle, anneal_freq=args.af, anneal_rate=args.ar, init_inverse_temp=args.invT0,warm_start_interval=args.nwarmStart, rws=args.rws)

if __name__ == '__main__':
    main()