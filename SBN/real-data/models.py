"""
For theory and comments, see "Generalizing Tree Probability Estimation via Bayesian Networks", Zhang & Matsen

Notes:
* Assume the standard total order on bitarrays.
* Addition on bitarrays is concatenation.
* A "composite" bitarray represents a subsplit. Say we have n taxa, and a
  well-defined parent node and child node. The first n bits represent the clade
  of the child node's sister (the parent node's other child) and the second n
  bits represent the clade of the child node itself.
* To "decompose" a composite bitarray means to cut it into two.
"""



import numpy as np
from collections import defaultdict
from bitarray import bitarray
from copy import deepcopy
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = np.finfo(float).eps

class ParamParser(object):
    def __init__(self):
        self.start_and_end = {}
        self.num_params = 0
        self.num_params_in_dicts = 0
        self.dict_name_list = []
    
    def add_item(self, name):
        start = self.num_params
        self.num_params += 1
        self.start_and_end[name] = start
    
    def add_dict(self, name, record_name=True):
        start = self.num_params_in_dicts
        self.num_params_in_dicts = self.num_params
        self.start_and_end[name] = (start, self.num_params)
        if name == 'rootsplit':
            self.num_rootsplit_params = self.num_params - start
        if record_name:
            self.dict_name_list.append(name)
    
    def get(self, tensor, name):
        start, end = self.start_and_end[name]
        return tensor[start: end]

    def get_scalar(self, tensor, name):
        start = self.start_and_end[name]
        return tensor[start].item()
    
    def get_index(self, name):
        return self.start_and_end[name]
    
    def check_item(self, name):
        return name in self.start_and_end

class SBN(nn.Module):
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, emp_tree_freq=None):
        super().__init__()
        torch.set_num_threads(1)
        self.taxa, self.ntaxa = taxa, len(taxa)

        self.map = {taxon: i for i, taxon in enumerate(taxa)}
        self.rootsplit_supp_dict = rootsplit_supp_dict
        self.subsplit_supp_dict = subsplit_supp_dict

        self.CPDParser = ParamParser()
        for split in self.rootsplit_supp_dict:
            self.CPDParser.add_item(split)
        self.CPDParser.add_dict('rootsplit', record_name=False)
        self.rs_len = len(self.rootsplit_supp_dict) 

        ss_mask, ss_max_len = [], 0
        for parent in self.subsplit_supp_dict: 
            ss_len = len(self.subsplit_supp_dict[parent])
            if ss_len > 1:
                for child in self.subsplit_supp_dict[parent]:
                    self.CPDParser.add_item(parent+child)
                self.CPDParser.add_dict(parent)
                ss_mask.append(torch.ones(ss_len, dtype=torch.uint8))
                ss_max_len = max(ss_len, ss_max_len)
        
        self.idx_map = np.append(np.arange(self.CPDParser.num_params), [-2,-1])
        self.rs_map = {split: i for i, split in enumerate(self.rootsplit_supp_dict.keys())}
        self.rs_reverse_map = {i: split for i, split in enumerate(self.rootsplit_supp_dict.keys())}
        
        
        self.subsplit_parameter_set = set(self.CPDParser.dict_name_list)
        self.ss_name_map = {parent: i for i, parent in enumerate(self.CPDParser.dict_name_list)} 
                
        self.ss_map = {} 
        self.ss_reverse_map = {} 
        for parent in self.subsplit_supp_dict:
            self.ss_map[parent] = {child: i for i, child in enumerate(self.subsplit_supp_dict[parent].keys())}
            self.ss_reverse_map[parent] = {i: child for i, child in enumerate(self.subsplit_supp_dict[parent].keys())} 
            
        self.ss_mask = torch.stack([F.pad(mask, (0, ss_max_len - mask.size(0)), 'constant', 0) for mask in ss_mask], dim=0)

        if emp_tree_freq is None:
            self.emp_tree_freq = {}
        else:
            self.emp_tree_freq = emp_tree_freq
            self.emp_tree_prob = torch.tensor(list(emp_tree_freq.values()), dtype=torch.float32)
            self.negDataEnt = torch.sum(self.emp_tree_prob * self.emp_tree_prob.clamp(EPS).log())
        
            for tree in self.emp_tree_freq:
                for node in tree.traverse('postorder'):
                    node.clade_bitarr = self.clade_to_bitarr(node.get_leaf_names())

    def clade_to_bitarr(self, clade):
        """Creates an indicator bitarray from a collection of taxa.

        :param clade: collection containing elements the SBN object's taxa list.
        :return: bitarray indicating which taxa are in clade.
        """
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))

    def set_clade_to_bitarr(self, tree_dict, tree_names):
        for name in tree_names:
            tree = tree_dict[name]
            for node in tree.traverse('preorder'):
                node.clade_bitarr = self.clade_to_bitarr(node.get_leaf_names())

    def update_rootsplit_CPDs(self):
        self.rs_CPDs = F.softmax(self.CPDParser.get(self.CPD_params, 'rootsplit'), 0)
        if torch.isnan(self.rs_CPDs).any():
            raise Exception('Invalid rootsplit probability! Check self.rs:(max {:.4f}, min {:.4f})'.format(np.max(self.rs.detach().numpy()), np.min(self.rs.detach().numpy())))
        
    def update_subsplit_CPDs(self):
        temp_mat = torch.zeros(self.ss_mask.size())
        temp_mat.masked_scatter_(self.ss_mask.bool(), self.CPD_params[self.rs_len:])
        masked_temp_mat = temp_mat.masked_fill((1-self.ss_mask).bool(), -float('inf'))
        masked_CPDs = F.softmax(masked_temp_mat, dim=1) 
        
        return masked_CPDs.masked_select(self.ss_mask.bool()), masked_CPDs
    
    def update_CPDs(self):
        self.update_rootsplit_CPDs()
        ss_CPDs, self.ss_masked_CPDs = self.update_subsplit_CPDs()
        self.CPDs = torch.cat((self.rs_CPDs, ss_CPDs))

    def update_other_CPDs(self, CPD_params):
        rs_CPDs = F.softmax(self.CPDParser.get(CPD_params, 'rootsplit'), 0)
        if torch.isnan(rs_CPDs).any():
            raise Exception('Invalid rootsplit probability! Check self.rs:(max {:.4f}, min {:.4f})'.format(np.max(self.rs.detach().numpy()), np.min(self.rs.detach().numpy())))
        
        temp_mat = torch.zeros(self.ss_mask.size())
        temp_mat.masked_scatter_(self.ss_mask.bool(), CPD_params[self.rs_len:])
        masked_temp_mat = temp_mat.masked_fill((1-self.ss_mask).bool(), -float('inf')) 
        masked_CPDs = F.softmax(masked_temp_mat, dim=1)  #2d

        ss_CPDs = masked_CPDs.masked_select(self.ss_mask.bool())
        return torch.cat((rs_CPDs, ss_CPDs))

    def CPDs_normalize(self):
        rs_CPDs = self.CPDParser.get(self.CPDs, 'rootsplit')
        rs_CPDs = rs_CPDs / torch.sum(rs_CPDs)

        temp_mat = torch.zeros(self.ss_mask.size())
        temp_mat.masked_scatter_(self.ss_mask.bool(), self.CPDs[self.rs_len:])
        norm_const = torch.sum(temp_mat, dim=-1)
        masked_CPDs = (temp_mat.T / norm_const).T
        ss_CPDs = masked_CPDs.masked_select(self.ss_mask.bool())

        self.CPDs = torch.cat((rs_CPDs, ss_CPDs))

    def get_emp_CPDs(self):
        rs_CPDs = self.CPDParser.get(self.CPDs, 'rootsplit')
        rs_CPDs = rs_CPDs / torch.sum(rs_CPDs)

        temp_mat = torch.zeros(self.ss_mask.size())
        temp_mat.masked_scatter_(self.ss_mask.bool(), self.CPDs[self.rs_len:])
        norm_const = torch.sum(temp_mat, dim=-1)
        lens = torch.sum(temp_mat!=0.0, dim=-1)
        masked_CPDs = ((norm_const / lens).repeat(temp_mat.shape[1], 1)).T
        ss_CPDs = masked_CPDs.masked_select(self.ss_mask.bool())

        return torch.cat((rs_CPDs, ss_CPDs))

    def CPDs_update(self, tree, wts, root_wts=None):
        subsplit_idxes_inorder = []
        root_idxes_inorder = []
        for node in tree.traverse('postorder'):
            if node.is_root():
                continue
            if not node.is_leaf():
                node.leaf_to_root_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                if not node.up.is_root():
                    parent_bipart_bitarr = min(node.clade_bitarr, ~node.clade_bitarr)
                    parent_bipart_bitarr_idx = self.CPDParser.get_index(parent_bipart_bitarr.to01())
                    comb_parent_bipart_bitarr = node.get_sisters()[0].clade_bitarr + node.clade_bitarr
                    ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.leaf_to_root_bipart_bitarr.to01()
                    ss_name = ss_parent + ss_child
                    if self.CPDParser.check_item(ss_name):
                        subsplit_idxes_inorder.append(self.CPDParser.get_index(ss_name))
                        root_idxes_inorder.append(parent_bipart_bitarr_idx)
            
        for node in tree.traverse('preorder'):
            if node.is_root():
                continue
            parent_bipart_bitarr = min(node.clade_bitarr, ~node.clade_bitarr)
            parent_bipart_bitarr_idx = self.CPDParser.get_index(parent_bipart_bitarr.to01())
            if node.up.is_root():
                node.root_to_leaf_bipart_bitarr = min(sister.clade_bitarr for sister in node.get_sisters())
                for sister in node.get_sisters():
                    if not sister.is_leaf():
                        comb_parent_bipart_bitarr = ((~node.clade_bitarr) ^ sister.clade_bitarr) + sister.clade_bitarr  
                        ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), sister.leaf_to_root_bipart_bitarr.to01()
                        ss_name = ss_parent + ss_child
                        if self.CPDParser.check_item(ss_name):
                            subsplit_idxes_inorder.append(self.CPDParser.get_index(ss_name))
                            root_idxes_inorder.append(parent_bipart_bitarr_idx)
            else:
                sister = node.get_sisters()[0]
                node.root_to_leaf_bipart_bitarr = min(sister.clade_bitarr, ~node.up.clade_bitarr)
                comb_parent_bipart_bitarr = sister.clade_bitarr + ~node.up.clade_bitarr
                ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.up.root_to_leaf_bipart_bitarr.to01()
                ss_name = ss_parent + ss_child
                if self.CPDParser.check_item(ss_name):
                    subsplit_idxes_inorder.append(self.CPDParser.get_index(ss_name))
                    root_idxes_inorder.append(parent_bipart_bitarr_idx)

                if not sister.is_leaf():
                    comb_parent_bipart_bitarr = ~node.up.clade_bitarr + sister.clade_bitarr
                    ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), sister.leaf_to_root_bipart_bitarr.to01()
                    ss_name = ss_parent + ss_child
                    if self.CPDParser.check_item(ss_name):
                        subsplit_idxes_inorder.append(self.CPDParser.get_index(ss_name))
                        root_idxes_inorder.append(parent_bipart_bitarr_idx)
            
            if not node.is_leaf():
                comb_parent_bipart_bitarr = ~node.clade_bitarr + node.clade_bitarr
                ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.leaf_to_root_bipart_bitarr.to01()
                ss_name = ss_parent + ss_child
                if self.CPDParser.check_item(ss_name):
                    subsplit_idxes_inorder.append(self.CPDParser.get_index(ss_name))
                    root_idxes_inorder.append(parent_bipart_bitarr_idx)
            
            comb_parent_bipart_bitarr = node.clade_bitarr + ~node.clade_bitarr
            ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.root_to_leaf_bipart_bitarr.to01()
            ss_name = ss_parent + ss_child
            if self.CPDParser.check_item(ss_name):
                subsplit_idxes_inorder.append(self.CPDParser.get_index(ss_name))
                root_idxes_inorder.append(parent_bipart_bitarr_idx)
            
            subsplit_idxes_inorder.append(parent_bipart_bitarr_idx)

        mapped_subsplit_idxes_inorder = torch.LongTensor(subsplit_idxes_inorder)
        self.CPDs[mapped_subsplit_idxes_inorder] += wts / (2 * self.ntaxa - 3.0)

    def CPDs_train_prob(self, tree_dict, tree_names, tree_wts):
        self.CPDs = torch.zeros(size=(self.CPDParser.num_params,), requires_grad=False)
        self.samp_tree_freq = torch.zeros(size=(len(tree_wts),), requires_grad=False)
        for i, tree_name in enumerate(tree_names):
            tree = tree_dict[tree_name]
            wts = tree_wts[i]

            self.CPDs_update(tree, wts)
            self.samp_tree_freq[i] = wts

        self.CPDs_emp = self.get_emp_CPDs().clone()

    def check_parent_child(self, parent, child=None):
        if parent not in self.ss_map:
            return False
        else:
            if child and child not in self.ss_map[parent]:
                return False
        return True

    def node_subsplit_idxes_update(self, node_subsplit_idxes, ss_parent, ss_child):
        if not self.check_parent_child(ss_parent, ss_child):
            node_subsplit_idxes.append(-1)
        else:
            ss_name = ss_parent + ss_child
            if self.CPDParser.check_item(ss_name):
                node_subsplit_idxes.append(self.CPDParser.get_index(ss_name))
            else:
                node_subsplit_idxes.append(-2)

    def EM_learn(self, tree_dict, tree_names, tree_wts, maxiter=100, miniter=50, abstol=1e-05, monitor=False, start_from_uniform=False, report_kl=False, alpha=0.0):
        self.set_clade_to_bitarr(tree_dict, tree_names)
        if start_from_uniform:
            self.CPDs = torch.ones(self.CPDParser.num_params)
            self.CPDs_normalize()
        else:
            self.CPDs_train_prob(tree_dict, tree_names, tree_wts)
        self.CPDs_normalize()
        logp, kldivs = [], []
        if monitor:
            t = time.time()
        for k in range(maxiter):
            curr_logp = 0.0
            CPDs = torch.zeros(size=(self.CPDParser.num_params,), requires_grad=False)
            for i, tree_name in enumerate(tree_names):
                tree = tree_dict[tree_name]
                wts = tree_wts[i]
                CPDsi, log_est_prob = self.em_update(tree)
                CPDs += wts * CPDsi
                curr_logp += wts * log_est_prob
            logp.append(curr_logp)
            if monitor:
                if report_kl:
                    kldivs.append(self.kl_div())
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.09f} | KL div {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp, kldivs[-1]))
                else:
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.09f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp))
                t = time.time()
            self.CPDs = (CPDs + alpha * self.CPDs_emp).clamp(EPS)
            self.CPDs_normalize()
            if k > miniter and abs(logp[-1] - logp[-2]) < abstol:
                break
        return logp, kldivs

    def SEMVR_learn(self, tree_dict, tree_names, tree_wts, maxiter=100, miniter=50, abstol=1e-05, biter=1000, batch=1, ema_rate=0.99, ar=0.75, af=50,  monitor=True, vr=True, start_from_uniform=False, report_kl=False, alpha=0.0, clip_increment=False, clip_value=1.0):
        self.set_clade_to_bitarr(tree_dict, tree_names)
        if start_from_uniform:
            self.CPDs = torch.ones(self.CPDParser.num_params)
            self.CPDs_normalize()
        else:
            self.CPDs_train_prob(tree_dict, tree_names, tree_wts)
        logp, kldivs = [], []
        if monitor:
            t = time.time()
        prenorm_CPDs = self.CPDs.clone()
        self.CPDs_normalize()
        for k in range(maxiter):
            curr_logp = 0.0
            CPDs_ctrl0 = self.CPDs.clone()
            CPDs_ctrl2 = torch.zeros(size=(self.CPDParser.num_params,), requires_grad=False)
            for i, tree_name in enumerate(tree_names):
                tree = tree_dict[tree_name]
                wts = tree_wts[i]
                tree_cp = deepcopy(tree)
                CPDsi, log_est_prob = self.em_update(tree_cp)
                del tree_cp
                CPDs_ctrl2 += wts * CPDsi
                curr_logp += wts * log_est_prob
            logp.append(curr_logp)
            
            if vr:
                for i in range(biter):
                    tree_ids = np.random.choice(len(tree_wts), p=tree_wts, size=(batch,), replace=False)
                    CPDs_ctrl1 = torch.zeros(size=(self.CPDParser.num_params,), requires_grad=False)
                    CPDs = torch.zeros(size=(self.CPDParser.num_params,), requires_grad=False)
                    for tree_id in tree_ids:
                        tree_name = tree_names[tree_id]
                        tree = tree_dict[tree_name]
                        tree_cp = deepcopy(tree)
                        CPDsi, CPDsi_ctrl1, log_est_prob = self.em_update(tree_cp, CPDs_ctrl0)
                        del tree_cp
                        CPDs_ctrl1 += CPDsi_ctrl1 / batch
                        CPDs += CPDsi / batch

                    norm = torch.sqrt(torch.sum((CPDs-CPDs_ctrl1 + CPDs_ctrl2-prenorm_CPDs)**2))
                    if clip_increment and norm > clip_value:
                        prenorm_CPDs = prenorm_CPDs + (1-ema_rate) * (CPDs-CPDs_ctrl1 + CPDs_ctrl2-prenorm_CPDs) / norm * clip_value
                    else:
                        prenorm_CPDs = prenorm_CPDs + (1-ema_rate) * (CPDs-CPDs_ctrl1 + CPDs_ctrl2-prenorm_CPDs)
                    if alpha == 0.0:
                        self.CPDs = prenorm_CPDs.clamp(EPS)
                    else:
                        self.CPDs = (prenorm_CPDs + alpha * self.CPDs_emp).clamp(EPS)
                    self.CPDs_normalize()
            else:
                if (k+1) % af == 0:
                    ema_rate = 1.0 - (1.0-ema_rate) * ar
                for i in range(biter):
                    tree_ids = np.random.choice(len(tree_wts), p=tree_wts, size=(batch,), replace=False)
                    CPDs = torch.zeros(size=(self.CPDParser.num_params,), requires_grad=False)
                    for tree_id in tree_ids:
                        tree_name = tree_names[tree_id]
                        tree = tree_dict[tree_name]
                        tree_cp = deepcopy(tree)
                        CPDsi, log_est_prob = self.em_update(tree_cp)
                        del tree_cp
                        CPDs += CPDsi / batch
                    prenorm_CPDs = ema_rate * prenorm_CPDs + (1-ema_rate) * CPDs
                    if alpha == 0.0:
                        self.CPDs = prenorm_CPDs.clamp(EPS)
                    else:
                        self.CPDs = (prenorm_CPDs + alpha * self.CPDs_emp).clamp(EPS)
                    self.CPDs_normalize()
            if monitor:
                if report_kl:
                    kldivs.append(self.kl_div())
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.06f} | KL div {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp, kldivs[-1]))
                else:
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp))
                t = time.time()

            if k > miniter and abs(logp[-1] - logp[-2]) < abstol:
                break
        return logp, kldivs

    def SGA_learn(self, stepsz, tree_dict, tree_names, tree_wts, batch=1, maxiter=500, miniter=50, abstol=1e-05, biter=1000, ar=0.75, af=50, start_from_uniform=False, monitor=True, report_kl=False, vr=True):
        self.set_clade_to_bitarr(tree_dict, tree_names)
        if start_from_uniform:
            self.CPD_params = nn.Parameter(torch.zeros(self.CPDParser.num_params), requires_grad=True)
        else:
            self.CPDs_train_prob(tree_dict, tree_names, tree_wts)
            self.CPDs_normalize()
            self.CPD_params = nn.Parameter(torch.log(self.CPDs.clamp(EPS)), requires_grad=True)
        self.update_CPDs()
        logp, kldivs = [], []
        assert isinstance(stepsz, float)
        optimizer = torch.optim.Adam(params = self.parameters(), lr = stepsz)

        if monitor:
            t = time.time()

        for k in range(maxiter):
            if vr:
                curr_logp = 0.0
                for i, tree_name in enumerate(tree_names):
                    tree = tree_dict[tree_name]
                    wts = tree_wts[i]
                    tree_cp = deepcopy(tree)
                    log_est_prob = self.tree_prob(tree_cp)
                    del tree_cp
                    curr_logp += wts * log_est_prob
                grad_F = torch.autograd.grad(-curr_logp, self.CPD_params)[0]
                CPD_params_f = nn.Parameter(torch.tensor(self.CPD_params.tolist()), requires_grad=True)
                self.update_CPDs()
                logp.append(curr_logp.item())
                for it in range(1, biter+1):
                    CPDs_f = self.update_other_CPDs(CPD_params_f)
                    tree_ids = np.random.choice(len(tree_wts), p=tree_wts, size=(batch,), replace=False)
                    samp_trees = [tree_dict[tree_names[idx]] for idx in tree_ids]
                    loglls, loglls_f = [], []
                    for tree in samp_trees:
                        tree_cp = deepcopy(tree)
                        logll, logll_f = self.tree_prob(tree_cp, CPDs_f)
                        del tree_cp
                        loglls.append(logll)
                        loglls_f.append(logll_f)
                    ll = torch.mean(torch.stack(loglls))
                    grad_f = torch.autograd.grad(-torch.mean(torch.stack(loglls_f)), CPD_params_f)[0]
                    loss = -ll - torch.sum((grad_f-grad_F).detach() * self.CPD_params)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.update_CPDs()
                    del CPDs_f
            else:
                if (k+1) % af == 0:
                    for g in optimizer.param_groups:
                        g['lr'] *= ar
                curr_logp = 0.0
                with torch.no_grad():
                    for i, tree_name in enumerate(tree_names):
                        tree = tree_dict[tree_name]
                        wts = tree_wts[i]
                        tree_cp = deepcopy(tree)
                        log_est_prob = self.tree_prob(tree_cp)
                        del tree_cp
                        curr_logp += wts * log_est_prob
                    logp.append(curr_logp)
                for it in range(1, biter+1):
                    tree_ids = np.random.choice(len(tree_wts), p=tree_wts, size=(batch,), replace=False)
                    samp_trees = [tree_dict[tree_names[idx]] for idx in tree_ids]
                    loglls = []
                    for tree in samp_trees:
                        tree_cp = deepcopy(tree)
                        logll = self.tree_prob(tree_cp)
                        del tree_cp
                        loglls.append(logll)
                    ll = torch.mean(torch.stack(loglls))
                    loss = -ll
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.update_CPDs()
            if monitor:
                if report_kl:
                    kldivs.append(self.kl_div())
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.06f} | KL div {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp, kldivs[-1]))
                else:
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp))
                t = time.time()
                
            if k > miniter and abs(logp[-1] - logp[-2]) < abstol:
                break
        return logp, kldivs


    def GA_learn(self, stepsz, tree_dict, tree_names, tree_wts, maxiter=500, miniter=50, abstol=1e-05, start_from_uniform=False, monitor=True, report_kl=False):
        self.set_clade_to_bitarr(tree_dict, tree_names)
        if start_from_uniform:
            self.CPD_params = nn.Parameter(torch.zeros(self.CPDParser.num_params), requires_grad=True)
        else:
            self.CPDs_train_prob(tree_dict, tree_names, tree_wts)
            self.CPDs_normalize()
            self.CPD_params = nn.Parameter(torch.log(self.CPDs.clamp(EPS)), requires_grad=True)
        self.update_CPDs()
        logp, kldivs = [], []
        assert isinstance(stepsz, float)
        optimizer = torch.optim.Adam(params = self.parameters(), lr = stepsz)

        if monitor:
            t = time.time()

        for k in range(maxiter):
            curr_logp = 0.0
            for i, tree_name in enumerate(tree_names):
                tree = tree_dict[tree_name]
                wts = tree_wts[i]
                tree_cp = deepcopy(tree)
                log_est_prob = self.tree_prob(tree_cp)
                del tree_cp
                curr_logp += wts * log_est_prob
            loss = - curr_logp
            logp.append(curr_logp.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.update_CPDs()

            if monitor:
                if report_kl:
                    kldivs.append(self.kl_div())
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.06f} | KL div {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp, kldivs[-1]))
                else:
                    print('{} Iter {} ({:.02f} seconds): current log likelihood {:.06f}'.format(time.asctime(time.localtime(time.time())), k + 1, time.time()-t, curr_logp))
                t = time.time()
                
            if k > miniter and abs(logp[-1] - logp[-2]) < abstol:
                break
        return logp, kldivs
    
    def grab_subsplit_idxes(self, tree):
            subsplit_idxes_inorder = []
            root_idxes_inorder = [] ##root at this node
            up_root_idxes_inorder = [] ##at or upper this node
            down_root_idxes_inorder = []  ##at or under this node
            cum_rootsplit_idxes_list = []  ##root at or under this node
            for node in tree.traverse("postorder"):
                if not node.is_root():
                    node.cum_rootsplit_idxes = [] 
                    node.leaf_to_root_subsplit_idxes = []
                    if not node.is_leaf():
                        parent_bipart_bitarr = min(node.clade_bitarr, ~node.clade_bitarr)
                        if parent_bipart_bitarr.to01() not in self.rs_map:
                            parent_bipart_bitarr_idx = -1 
                        else:
                            parent_bipart_bitarr_idx = self.CPDParser.get_index(parent_bipart_bitarr.to01())
                        node.cum_rootsplit_idxes.append(parent_bipart_bitarr_idx)

                        node.leaf_to_root_child_subsplit_idxes = []
                        for child in node.children:
                            node.leaf_to_root_child_subsplit_idxes.extend(child.leaf_to_root_subsplit_idxes)
                            node.cum_rootsplit_idxes.extend(child.cum_rootsplit_idxes)

                        node.leaf_to_root_bipart_bitarr = min(child.clade_bitarr for child in node.children)
                        node.leaf_to_root_subsplit_idxes.extend(node.leaf_to_root_child_subsplit_idxes)
                        
                        upnode = node.up
                        if not upnode.is_root():
                            comb_parent_bipart_bitarr = node.get_sisters()[0].clade_bitarr + node.clade_bitarr
                            ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.leaf_to_root_bipart_bitarr.to01()
                            self.node_subsplit_idxes_update(node.leaf_to_root_subsplit_idxes, ss_parent, ss_child)
                            if node.leaf_to_root_subsplit_idxes[-1] not in [-1, -2]:
                                subsplit_idxes_inorder.append(node.leaf_to_root_subsplit_idxes[-1])
                                root_idxes_inorder.append(-1)
                                parent_bipart_bitarr = min(upnode.clade_bitarr, ~upnode.clade_bitarr)
                                if parent_bipart_bitarr.to01() not in self.rs_map:
                                    parent_bipart_bitarr_idx = -1
                                else:
                                    parent_bipart_bitarr_idx = self.CPDParser.get_index(parent_bipart_bitarr.to01())
                                up_root_idxes_inorder.append(parent_bipart_bitarr_idx)
                                down_root_idxes_inorder.append(-1)
                    else:
                        parent_bipart_bitarr = min(node.clade_bitarr, ~node.clade_bitarr)
                        if parent_bipart_bitarr.to01() not in self.rs_map:
                            node.cum_rootsplit_idxes.append(-1)
                        else:
                            node.cum_rootsplit_idxes.append(self.CPDParser.get_index(parent_bipart_bitarr.to01()))
                    cum_rootsplit_idxes_list.append(node.cum_rootsplit_idxes+[-1]*(2*self.ntaxa-len(node.cum_rootsplit_idxes)))

            subsplit_idxes_list = []          
            for node in tree.traverse("preorder"):
                if not node.is_root():
                    node.root_to_leaf_subsplit_idxes = []
                    parent_bipart_bitarr = min(node.clade_bitarr, ~node.clade_bitarr)
                    if parent_bipart_bitarr.to01() not in self.rs_map:
                        parent_bipart_bitarr_idx = -1
                    else:
                        parent_bipart_bitarr_idx = self.CPDParser.get_index(parent_bipart_bitarr.to01())

                    if node.up.is_root():
                        node.root_to_leaf_bipart_bitarr = min(sister.clade_bitarr for sister in node.get_sisters())
                        for sister in node.get_sisters():
                            if not sister.is_leaf():
                                node.root_to_leaf_subsplit_idxes.extend(sister.leaf_to_root_subsplit_idxes)
                                comb_parent_bipart_bitarr = ((~node.clade_bitarr) ^ sister.clade_bitarr) + sister.clade_bitarr
                                ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), sister.leaf_to_root_bipart_bitarr.to01()
                                self.node_subsplit_idxes_update(node.root_to_leaf_subsplit_idxes, ss_parent, ss_child)
                                if node.root_to_leaf_subsplit_idxes[-1] not in [-1, -2]:
                                    subsplit_idxes_inorder.append(node.root_to_leaf_subsplit_idxes[-1])
                                    root_idxes_inorder.append(-1)
                                    up_root_idxes_inorder.append(-1)
                                    down_root_idxes_inorder.append(parent_bipart_bitarr_idx)
                    else:
                        sister = node.get_sisters()[0]
                        node.root_to_leaf_bipart_bitarr = min(sister.clade_bitarr, ~node.up.clade_bitarr)
                        node.root_to_leaf_subsplit_idxes.extend(node.up.root_to_leaf_subsplit_idxes)
                        comb_parent_bipart_bitarr = sister.clade_bitarr + ~node.up.clade_bitarr
                        ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.up.root_to_leaf_bipart_bitarr.to01()
                        self.node_subsplit_idxes_update(node.root_to_leaf_subsplit_idxes, ss_parent, ss_child)
                        if node.root_to_leaf_subsplit_idxes[-1] not in [-1, -2]:
                            subsplit_idxes_inorder.append(node.root_to_leaf_subsplit_idxes[-1])
                            root_idxes_inorder.append(-1)
                            up_root_idxes_inorder.append(-1)
                            down_root_idxes_inorder.append(parent_bipart_bitarr_idx)

                        if not sister.is_leaf():
                            node.root_to_leaf_subsplit_idxes.extend(sister.leaf_to_root_child_subsplit_idxes)
                            comb_parent_bipart_bitarr = ~node.up.clade_bitarr + sister.clade_bitarr
                            ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), sister.leaf_to_root_bipart_bitarr.to01()
                            self.node_subsplit_idxes_update(node.root_to_leaf_subsplit_idxes, ss_parent, ss_child)
                            if node.root_to_leaf_subsplit_idxes[-1] not in [-1, -2]:
                                subsplit_idxes_inorder.append(node.root_to_leaf_subsplit_idxes[-1])
                                root_idxes_inorder.append(-1)
                                up_root_idxes_inorder.append(-1)
                                down_root_idxes_inorder.append(parent_bipart_bitarr_idx)

                    node_subsplit_idxes = [parent_bipart_bitarr_idx]
                        
                    if not node.is_leaf(): 
                        node_subsplit_idxes.extend(node.leaf_to_root_child_subsplit_idxes)
                        comb_parent_bipart_bitarr = ~node.clade_bitarr + node.clade_bitarr
                        ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.leaf_to_root_bipart_bitarr.to01()
                        self.node_subsplit_idxes_update(node_subsplit_idxes, ss_parent, ss_child)
                        if node_subsplit_idxes[-1] not in [-1, -2]:
                            subsplit_idxes_inorder.append(node_subsplit_idxes[-1])
                            root_idxes_inorder.append(parent_bipart_bitarr_idx)
                            up_root_idxes_inorder.append(-1)
                            down_root_idxes_inorder.append(-1)
                    
                    node_subsplit_idxes.extend(node.root_to_leaf_subsplit_idxes) 
                    comb_parent_bipart_bitarr = node.clade_bitarr + ~node.clade_bitarr
                    ss_parent, ss_child = comb_parent_bipart_bitarr.to01(), node.root_to_leaf_bipart_bitarr.to01()
                    self.node_subsplit_idxes_update(node_subsplit_idxes, ss_parent, ss_child)
                    if node_subsplit_idxes[-1] not in [-1, -2]:
                        subsplit_idxes_inorder.append(node_subsplit_idxes[-1])
                        root_idxes_inorder.append(parent_bipart_bitarr_idx)
                        up_root_idxes_inorder.append(-1)
                        down_root_idxes_inorder.append(-1)

                    subsplit_idxes_list.append(node_subsplit_idxes)                        
            
            return cum_rootsplit_idxes_list, subsplit_idxes_list, subsplit_idxes_inorder, root_idxes_inorder, up_root_idxes_inorder, down_root_idxes_inorder


    def tree_prob(self, tree, other_CPDs=None):
        cum_rootsplit_idxes_list, subsplit_idxes_list, subsplit_idxes_inorder, root_idxes_inorder, up_root_idxes_inorder, down_root_idxes_inorder = self.grab_subsplit_idxes(tree)
        mapped_idxes_list = torch.LongTensor(subsplit_idxes_list)
        def get_estprob(CPDs):
            nowCPDs = torch.cat((CPDs, torch.tensor([1.0, 0.0])))
            root_prob = nowCPDs[mapped_idxes_list].prod(1)
            estprob = root_prob.sum(0)
            return estprob.clamp(EPS).log()
        if other_CPDs == None:
            return get_estprob(self.CPDs)
        else:
            return get_estprob(self.CPDs), get_estprob(other_CPDs)

    def em_update(self, tree, otherCPDs=None):
        cum_rootsplit_idxes_list, subsplit_idxes_list, subsplit_idxes_inorder, root_idxes_inorder, up_root_idxes_inorder, down_root_idxes_inorder = self.grab_subsplit_idxes(tree)
        mapped_idxes_list = torch.LongTensor(subsplit_idxes_list)
        preorder_rootsplit_idxes_list = mapped_idxes_list[:,0] 
        mapped_cum_rootsplit_idxes_list = torch.LongTensor(cum_rootsplit_idxes_list) 
        postorder_rootsplit_idxes_list = mapped_cum_rootsplit_idxes_list[:,0]
        def get_nextCPDs(CPDs):
            nowCPDs = torch.cat((CPDs, torch.tensor([1.0, 0.0])))
            log_root_prob = nowCPDs[mapped_idxes_list].log().sum(1)  
            log_estprob = log_root_prob.logsumexp(0)
            root_prob = torch.exp(log_root_prob - log_estprob)
            del nowCPDs
            
            nextCPDs = torch.zeros(self.CPDParser.num_params+1) 
            nextCPDs[preorder_rootsplit_idxes_list] += root_prob 
            cum_root_prob = nextCPDs[mapped_cum_rootsplit_idxes_list].sum(1) 

            tmpCPDs = torch.zeros(size=(3,self.CPDParser.num_rootsplit_params+1))
            tmpCPDs[0, preorder_rootsplit_idxes_list] += root_prob
            tmpCPDs[2, postorder_rootsplit_idxes_list] += cum_root_prob
            tmpCPDs[1, postorder_rootsplit_idxes_list] += 1 - cum_root_prob + tmpCPDs[0, postorder_rootsplit_idxes_list]
            nextCPDs[torch.LongTensor(subsplit_idxes_inorder)] += tmpCPDs[0, torch.LongTensor(root_idxes_inorder)] + tmpCPDs[1, torch.LongTensor(up_root_idxes_inorder)] + tmpCPDs[2, torch.LongTensor(down_root_idxes_inorder)] 
            del tmpCPDs

            return nextCPDs[:-1], log_estprob
        
        if otherCPDs == None:
            return get_nextCPDs(self.CPDs)
        else:
            self_nextCPDs, self_log_estprob = get_nextCPDs(self.CPDs)
            other_nextCPDs, other_log_estprob = get_nextCPDs(otherCPDs)
            return self_nextCPDs, other_nextCPDs, self_log_estprob

    def kl_div(self):
        log_model_prob = []
        with torch.no_grad():
            for tree in self.emp_tree_freq:
                tree_cp = deepcopy(tree)
                log_model_prob.append(self.tree_prob(tree_cp))
                del tree_cp
            log_model_prob = torch.tensor(log_model_prob)
            kldiv = self.negDataEnt - torch.sum(self.emp_tree_prob * log_model_prob)
        return kldiv.item()