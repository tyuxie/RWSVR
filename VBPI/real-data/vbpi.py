from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import random
import numpy as np
from utils import namenum
from deep_branchModel import DeepModel
from vector_sbnModel import SBN
from phyloModel import PHY
import logging
import matplotlib.pyplot as plt
import pickle

class VBPI(nn.Module):
    
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden, subModel, emp_tree_freq=None,
                 scale=0.1, psp=True, feature_dim=2, hidden_sizes=[50], flow_type='planar', num_of_layers_nf=16, logger=None):
        super().__init__()
        torch.set_num_threads(1)
        self.EPS = 1e-40 
        self.taxa, self.emp_tree_freq = taxa, emp_tree_freq
        if emp_tree_freq:
            self.trees, self.emp_freqs = zip(*emp_tree_freq.items())
            self.emp_freqs = np.array(self.emp_freqs)
            self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        
        self.ntips = len(data)
        self.scale = scale
        self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale)
        self.log_p_tau = - np.sum(np.log(np.arange(3, 2*self.ntips-3, 2)))
        
        self.tree_model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.rs_embedding_map, self.ss_embedding_map = self.tree_model.rs_map, self.tree_model.ss_map 
        
        self.branch_model = DeepModel(self.ntips, self.rs_embedding_map, self.ss_embedding_map, psp=psp, hidden_sizes=hidden_sizes, feature_dim=feature_dim, flow_type=flow_type, num_of_layers_nf=num_of_layers_nf)
        
        self.logger = logger
        
    def load_from(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path))
        self.eval()
        self.tree_model.update_CPDs()
    
    def load_CPD_params(self, CPD_params):
        self.tree_model.CPD_params += CPD_params
        self.tree_model.update_CPDs()

    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.items():
            tree_cp = deepcopy(tree)
            kl_div += wt * np.log(max(np.exp(self.tree_model.loglikelihood(tree_cp)), self.EPS))
            del tree_cp
        kl_div = self.negDataEnt - kl_div
        return kl_div
        
    def exclusive_kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.items():
            tree_cp = deepcopy(tree)
            ll = self.tree_model.loglikelihood(tree_cp)
            del tree_cp
            kl_div += np.exp(ll) * (ll - np.log(max(wt, self.EPS)))
        return kl_div
    
    def logq_tree(self, tree, CPDs_f=None):
        return self.tree_model(tree, CPDs_f)
    
    def lower_bound(self, n_particles=1, n_runs=1000):
        lower_bounds = []
        with torch.no_grad():
            for run in range(n_runs):
                samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
                [namenum(tree, self.taxa) for tree in samp_trees]    
                samp_log_branch, logq_branch = self.branch_model(samp_trees)

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])       
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0))            
            
            lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()
    
    def tree_lower_bound(self, tree, n_particles=1, n_runs=1000):
        lower_bounds = []
        namenum(tree, self.taxa)
        with torch.no_grad():
            for run in range(n_runs):
                test_trees = [tree for particle in range(n_particles)]
                samp_log_branch, logq_branch = self.branch_model(test_trees) 

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, test_tree) for log_branch, test_tree in zip(*[samp_log_branch, test_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch) 
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_branch, 0) - math.log(n_particles))
                
            lower_bound = torch.stack(lower_bounds).mean()

        return lower_bound.item()
    
    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]  
        [namenum(tree, self.taxa) for tree in samp_trees] 
        
        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])]) 
        logp_prior = self.phylo_model.logprior(samp_log_branch) 
        logp_joint = inverse_temp * logll + logp_prior 
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0) 
        
        l_signal = logp_joint - logq_tree - logq_branch 
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)  
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)  
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)
        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)

    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10, CPDs_f=None):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        if CPDs_f == None:
            logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        else:
            logq_tree, logq_tree_f = [], []
            for tree in samp_trees:
                q1, q2 = self.logq_tree(tree, CPDs_f)
                logq_tree.append(q1)
                logq_tree_f.append(q2)
            logq_tree = torch.stack(logq_tree)
            logq_tree_f = torch.stack(logq_tree_f)

        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree.detach() - logq_branch
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0) 
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)
        if CPDs_f == None:
            return temp_lower_bound, rws_fake_term, lower_bound, torch.max(logll)
        
        rws_fake_term_f = torch.sum(snis_wts.detach() * logq_tree_f, dim=0)
        return temp_lower_bound, rws_fake_term, rws_fake_term_f, lower_bound, torch.max(logll)


    def rws_lower_bound_vr(self, n_particles=1000):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        logq_tree, grad_tree = [], []
        for tree in samp_trees:
            logq_tree.append(self.logq_tree(tree))
            grad_tree.append(-torch.autograd.grad(logq_tree[-1], self.tree_model.CPD_params)[0])
            self.tree_model.update_CPDs()
        grad_tree = torch.stack(grad_tree, dim=0).detach()
        logq_tree = torch.stack(logq_tree).detach()
        with torch.no_grad():
            samp_log_branch, logq_branch = self.branch_model(samp_trees)
            logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
            logp_prior = self.phylo_model.logprior(samp_log_branch)
        return logp_prior, logll, logq_tree, logq_branch, grad_tree

    def learn(self, stepsz, maxiter=100000, test_freq=1000, lb_test_freq=5000, anneal_freq=20000, anneal_rate=0.75, n_particles=10, init_inverse_temp=0.001, warm_start_interval=50000, rws=False):
        lbs, lls = [], []
        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz}
        
        optimizer = torch.optim.Adam([
                    {'params': self.tree_model.parameters(), 'lr':stepsz['tree']},
                    {'params': self.branch_model.parameters(), 'lr': stepsz['branch']}
                ])
        run_time = -time.time()
        for it in range(1, maxiter+1):
            inverse_temp = min(1., init_inverse_temp + it * 1.0/warm_start_interval) 
            if not rws: 
                temp_lower_bound, vimco_fake_term, lower_bound, logll = self.vimco_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - vimco_fake_term
            else:
                temp_lower_bound, rws_fake_term, lower_bound, logll = self.rws_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - rws_fake_term
            lbs.append(lower_bound.item())
            lls.append(logll.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.tree_model.update_CPDs()
            
            if it % test_freq == 0:
                run_time += time.time()
                if self.emp_tree_freq:
                    self.logger.info('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}| KL : {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls), self.kl_div()))
                else:
                    self.logger.info('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls)))
                
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    lb1 = self.lower_bound(n_particles=1)
                    run_time += time.time()
                    self.logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(it, run_time, lb1))

                run_time = -time.time()
                lbs, lls = [], []
            
            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate

    def rws_vr_learn(self, stepsz, maxiter=2000, test_freq=100, lb_test_freq=1000, anneal_freq=20000, anneal_rate=0.75, n_particles=10, init_inverse_temp=0.001, warm_start_interval=100000, biter=100, batch_F=1000):
        lbs, lls = [], []
        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz}

        optimizer = torch.optim.Adam([
                    {'params': self.tree_model.parameters(), 'lr':stepsz['tree']},
                    {'params': self.branch_model.parameters(), 'lr': stepsz['branch']}
                ])
        run_time = -time.time()
        for it in range(1, maxiter+1):
            CPD_params_f = torch.tensor(self.tree_model.CPD_params.tolist(), requires_grad=True)
            logp_prior_F, logll_F, logq_tree_F, logq_branch_F, grad_tree_F = self.rws_lower_bound_vr(batch_F)
            for bit in range(1, biter+1):
                tit = (it-1)*biter + bit
                inverse_temp = min(1.0, init_inverse_temp + tit / warm_start_interval)
                snis_wts_F = torch.softmax(inverse_temp * logll_F + logp_prior_F - logq_tree_F - logq_branch_F, dim=0)
                grad_F = torch.matmul(snis_wts_F, grad_tree_F)

                CPD_f = self.tree_model._update_CPDs(CPD_params_f)
                temp_lower_bound, rws_fake_term, rws_fake_term_f, lower_bound, logll = self.rws_lower_bound(inverse_temp, n_particles, CPD_f)
                grad_f = - torch.autograd.grad(rws_fake_term_f, CPD_params_f)[0]
                loss = - temp_lower_bound - rws_fake_term - torch.sum((grad_f-grad_F).detach() * self.tree_model.CPD_params)
                lbs.append(lower_bound.item())
                lls.append(logll.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.tree_model.update_CPDs()
                if tit % test_freq == 0:
                    run_time += time.time()
                    if self.emp_tree_freq:
                        self.logger.info('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}| KL : {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls), self.kl_div()))
                    else:
                        self.logger.info('Iter {}:({:.1f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls)))
                    if tit % lb_test_freq == 0:
                        run_time = -time.time()
                        lb1 = self.lower_bound(n_particles=1)
                        run_time += time.time()
                        self.logger.info('>>> Iter {}:({:.1f}s) Test Lower Bound: {:.4f}'.format(tit, run_time, lb1))
                        
                    run_time = -time.time()
                    lbs, lls = [], []
                if tit % anneal_freq == 0:
                    for g in optimizer.param_groups:
                        g['lr'] *= anneal_rate
            del CPD_params_f, logp_prior_F, logll_F, logq_tree_F, logq_branch_F, grad_tree_F
