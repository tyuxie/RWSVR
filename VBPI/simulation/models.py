import numpy as np
from bitarray import bitarray
from ete3 import Tree
from collections import defaultdict
from optimizers import SGD_Server
from utils import softmax, softmax_parser, dict_sum, upper_clip, logmeanexp, BitArray, ParamParser
import time


class SBN(object):
    """
    Subsplit Bayesian Networks (SBNs) for distributions over 
    phylogenetic tree topologies. SBNs utilize the similarity 
    among tree topologies to provide a familty of flexibile 
    distributions over the entire tree topology space. 
    
    Parameters
    ----------
    taxa : ``list``, a list of the labels of the sequences.
    rootsplit_supp_dict : ``dict``, a dictionary of rootsplit support, 
                           usually obtained from some bootstrap run.
    subsplit_supp_dict: ``dict``, a dictionary of subsplit support,
                         obtained similarly as above.
    
    References
    ----------
    .. [1] Cheng Zhang and Frederick A. Matsen IV. "Generalizing Tree 
    Probability Estimation via Bayesian Networks", Advances in Neural 
    Information Processing Systems 32, 2018. (https://papers.nips.cc/
    paper/7418-generalizing-tree-probability-estimation-via-bayesian-
    networks)
    
    """
    
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, support_only=True):
        self.taxa, self.ntaxa = taxa, len(taxa)
        self.toBitArr = BitArray(taxa)
        self.CPDs_parser = ParamParser()
        self.dict_names = []
        self.rootsplit_supp_dict = rootsplit_supp_dict
        self.subsplit_supp_dict = subsplit_supp_dict
        if not support_only:
            init_CPDs = []
        for split in rootsplit_supp_dict:
            self.CPDs_parser.add_item(split)
            if not support_only:
                init_CPDs.append(rootsplit_supp_dict[split])
        self.CPDs_parser.add_dict('rootsplits')
        self.dict_names.append('rootsplits')
        for parent in subsplit_supp_dict:
            for child in subsplit_supp_dict[parent]:
                self.CPDs_parser.add_item(parent + child)
                if not support_only:
                    init_CPDs.append(subsplit_supp_dict[parent][child])
            self.CPDs_parser.add_dict(parent)
            self.dict_names.append(parent)
        
        self.num_CPDs = self.CPDs_parser.num_params
        self._CPDs = np.zeros(self.num_CPDs)
        if support_only:
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
        else:
            self.CPDs = init_CPDs
        
    def get_CPDs(self, name):
        return self.CPDs_parser.get(self.CPDs, name)
        
    def assign_CPDs(self, vect, name, value):
        self.CPDs_parser.assign(vect, name, value)
        
    @property
    def rootsplit_CPDs(self):
        return {split: self.get_CPDs(split) for split in self.rootsplit_supp_dict}
        
    def subsplit_CPDs(self, parent):
        return {child: self.get_CPDs(parent+child) for child in self.subsplit_supp_dict[parent]}
        
    
    def check_item(self, name):
        return self.CPDs_parser.check_item(name)
        
    def node2bitMap(self, tree, bit_type='split'):
        if bit_type == 'split':
            return {node: self.toBitArr.minor(self.toBitArr.from_clade(node.get_leaf_names())).to01() for node in tree.traverse('postorder') if not node.is_root()}
        elif bit_type == 'clade':
            return {node: self.toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
    
    def check_subsplit_pair(self, subsplit_pair):
        # if self.CPDs_parser.get(self.CPDs, subsplit_pair) == 0.0:
        if self.get_CPDs(subsplit_pair) == 0.0:
            return False
        return True
    
    def rooted_tree_probs(self, tree, nodetobitMap=None):
        """
        Compute the logprobs of all compatible rooted tree topologies
        of the unrooted tree via a two pass algorithm. 
        The overall computational cost is O(N) where N is the number of
        species. 
        """
        sbn_est_up = {node: 0.0 for node in tree.traverse('postorder') if not node.is_root()}
        Up = {node: 0.0 for node in tree.traverse('postorder') if not node.is_root()}

        if not nodetobitMap:
            nodetobitMap = self.node2bitMap(tree, 'clade')
            
        bipart_bitarr_up = {}
        bipart_bitarr_prob = {}
        bipart_bitarr_node = {}
        zero_bubble_up = {node:0 for node in tree.traverse('postorder') if not node.is_root()}
        zero_bubble_Up = {node:0 for node in tree.traverse('postorder') if not node.is_root()}
        
        for node in tree.traverse('postorder'):
            if not node.is_leaf() and not node.is_root():
                for child in node.children:
                    Up[node] += sbn_est_up[child]
                    zero_bubble_Up[node] += zero_bubble_up[child]
                    
                bipart_bitarr = min(nodetobitMap[child] for child in node.children)
                bipart_bitarr_up[node] = bipart_bitarr
                if not node.up.is_root():
                    sbn_est_up[node] += Up[node]
                    zero_bubble_up[node] += zero_bubble_Up[node]
                    comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                    item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr.to01()
                    if not self.check_subsplit_pair(item_name):
                        zero_bubble_up[node] += 1
                    else:
                        sbn_est_up[node] += np.log(self.get_CPDs(item_name))
        
        sbn_est_down = {node: 0.0 for node in tree.traverse('preorder') if not node.is_root()}
        zero_bubble_down = {node: 0.0 for node in tree.traverse('preorder') if not node.is_root()}
        zero_bubble = defaultdict(int)
        bipart_bitarr_down = {}
        
        for node in tree.traverse('preorder'):
            if not node.is_root():
                if node.up.is_root():
                    parent_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                    bipart_bitarr_down[node] = parent_bipart_bitarr
                    
                    for sister in node.get_sisters():
                        if not sister.is_leaf():
                            sbn_est_down[node] += Up[sister]
                            zero_bubble_down[node] += zero_bubble_Up[sister]
                            comb_parent_bipart_bitarr = ((~nodetobitMap[node]) ^ nodetobitMap[sister]) + nodetobitMap[sister]
                            item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_up[sister].to01()
                            if not self.check_subsplit_pair(item_name):
                                zero_bubble_down[node] += 1
                            else:
                                sbn_est_down[node] += np.log(self.get_CPDs(item_name))
                else:
                    sister = node.get_sisters()[0]
                    parent_bipart_bitarr = min([nodetobitMap[sister], ~nodetobitMap[node.up]])
                    bipart_bitarr_down[node] = parent_bipart_bitarr
                    sbn_est_down[node] += sbn_est_down[node.up]
                    zero_bubble_down[node] += zero_bubble_down[node.up]
                    comb_parent_bipart_bitarr = nodetobitMap[sister] + ~nodetobitMap[node.up]
                    item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_down[node.up].to01()
                    if not self.check_subsplit_pair(item_name):
                        zero_bubble_down[node] += 1
                    else:
                        sbn_est_down[node] += np.log(self.get_CPDs(item_name))
                    
                    if not sister.is_leaf():
                        sbn_est_down[node] += Up[sister]
                        zero_bubble_down[node] += zero_bubble_Up[sister]
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[sister]
                        item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_up[sister].to01()
                        if not self.check_subsplit_pair(item_name):
                            zero_bubble_down[node] += 1
                        else:
                            sbn_est_down[node] += np.log(self.get_CPDs(item_name))
                
                parent_bipart_bitarr = self.toBitArr.minor(nodetobitMap[node])
                bipart_bitarr_node[node] = parent_bipart_bitarr
                if parent_bipart_bitarr.to01() not in self.rootsplit_supp_dict or self.get_CPDs(parent_bipart_bitarr.to01()) == 0.0:
                    zero_bubble[parent_bipart_bitarr.to01()] += 1
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0
                else:
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] = np.log(self.get_CPDs(parent_bipart_bitarr.to01()))
                    
                if not node.is_leaf():
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] += Up[node]
                    zero_bubble[parent_bipart_bitarr.to01()] += zero_bubble_Up[node]
                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_up[node].to01()
                    if not self.check_subsplit_pair(item_name):
                        zero_bubble[parent_bipart_bitarr.to01()] += 1
                    else:
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] += np.log(self.get_CPDs(item_name))
                
                bipart_bitarr_prob[parent_bipart_bitarr.to01()] += sbn_est_down[node]
                zero_bubble[parent_bipart_bitarr.to01()] += zero_bubble_down[node]
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_down[node].to01()
                if not self.check_subsplit_pair(item_name):
                    zero_bubble[parent_bipart_bitarr.to01()] += 1
                else:
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] += np.log(self.get_CPDs(item_name))
                
        bipart_bitarr_prob_real = {key: value if zero_bubble[key]==0 else -np.inf for key, value in bipart_bitarr_prob.items()}
        bipart_bitarr_prob_mask = {key: value if zero_bubble[key]<2 else -np.inf for key, value in bipart_bitarr_prob.items()}
        return bipart_bitarr_prob_real, bipart_bitarr_prob_mask, bipart_bitarr_node, zero_bubble
        
    
    def cum_root_probs(self, tree, bipart_bitarr_prob, bipart_bitarr_node, max_bipart_bitarr_prob=None, log=False, normalized=True):
        """
        Compute the cumulative sums of the rooted tree probabilities
        from the leaves to the root.
        """
        root_prob = {}
        cum_root_prob = defaultdict(float)
        if max_bipart_bitarr_prob is None:
            max_bipart_bitarr_prob = np.max(list(bipart_bitarr_prob.values()))
        for node in tree.traverse('postorder'):
            if not node.is_root():
                bipart_bitarr = bipart_bitarr_node[node]
                cum_root_prob[bipart_bitarr.to01()] += np.exp(bipart_bitarr_prob[bipart_bitarr.to01()] - max_bipart_bitarr_prob)
                if not node.is_leaf():
                    for child in node.children:
                        cum_root_prob[bipart_bitarr.to01()] += cum_root_prob[bipart_bitarr_node[child].to01()]
    
        root_prob_sum = 0.0
        for child in tree.children:
            root_prob_sum += cum_root_prob[bipart_bitarr_node[child].to01()]
    
        if normalized:
            if log:
                root_prob = {key: bipart_bitarr_prob[key] - max_bipart_bitarr_prob - np.log(root_prob_sum) for key in bipart_bitarr_prob}
                cum_root_prob = {key: np.log(cum_root_prob[key]) - np.log(root_prob_sum) if cum_root_prob[key] != 0.0 else -np.inf for key in bipart_bitarr_prob}
            else:
                root_prob = {key: np.exp(bipart_bitarr_prob[key] - max_bipart_bitarr_prob)/root_prob_sum for key in bipart_bitarr_prob}
                cum_root_prob = {key: cum_root_prob[key]/root_prob_sum for key in bipart_bitarr_prob}
    
        return root_prob, cum_root_prob, root_prob_sum
        
      
    def tree_loglikelihood(self, tree, nodetobitMap=None, grad=False, entry_ub=10.0, value_and_grad=False):
        """ Compute the SBN loglikelihood and gradient. """
         
        if not nodetobitMap:
            nodetobitMap = self.node2bitMap(tree, bit_type='clade')
        
        bipart_bitarr_prob_real, bipart_bitarr_prob, bipart_bitarr_node, zero_bubble = self.rooted_tree_probs(tree, nodetobitMap)
        bipart_bitarr_prob_real_vec = np.array(list(bipart_bitarr_prob_real.values()))
        max_bipart_bitarr_prob_real = np.max(bipart_bitarr_prob_real_vec)
        if max_bipart_bitarr_prob_real != -np.inf:
            loglikelihood = np.log(np.sum(np.exp(bipart_bitarr_prob_real_vec - max_bipart_bitarr_prob_real))) + max_bipart_bitarr_prob_real
        else:
            loglikelihood = -np.inf
        max_bipart_bitarr_prob = np.max(list(bipart_bitarr_prob.values()))
        if not grad:
            return loglikelihood
        
        CPDs_grad = np.ones(self.num_CPDs) * (-np.inf)
        root_prob_real, cum_root_prob_real, _ = self.cum_root_probs(tree, bipart_bitarr_prob_real, bipart_bitarr_node, max_bipart_bitarr_prob_real, log=True)
        _, cum_root_prob, root_prob_sum = self.cum_root_probs(tree, bipart_bitarr_prob, bipart_bitarr_node, max_bipart_bitarr_prob, normalized=False)
        
        for node in tree.traverse('postorder'):
            if not node.is_root():
                bipart_bitarr = bipart_bitarr_node[node]
                if bipart_bitarr.to01() in self.rootsplit_supp_dict:
                    if self.get_CPDs(bipart_bitarr.to01()) == 0.0:
                        self.assign_CPDs(CPDs_grad, bipart_bitarr.to01(), upper_clip(bipart_bitarr_prob[bipart_bitarr.to01()] - loglikelihood, entry_ub))
                    else:
                        self.assign_CPDs(CPDs_grad, bipart_bitarr.to01(), root_prob_real[bipart_bitarr.to01()] - np.log(self.get_CPDs(bipart_bitarr.to01())))
                
                if not node.is_leaf():
                    children_bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    if not node.up.is_root():
                        parent_bipart_bitarr = bipart_bitarr_node[node.up]
                        comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                        item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                        if self.check_item(item_name):
                            if self.get_CPDs(item_name) == 0.0:
                                self.assign_CPDs(CPDs_grad, item_name, np.log(root_prob_sum - cum_root_prob[parent_bipart_bitarr.to01()] + \
                                    np.exp(bipart_bitarr_prob[parent_bipart_bitarr.to01()] - max_bipart_bitarr_prob)) + upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                            else:
                                cum_node_prob = 1.0 - np.exp(cum_root_prob_real[parent_bipart_bitarr.to01()]) + np.exp(root_prob_real[parent_bipart_bitarr.to01()])
                                if cum_node_prob == 0.0:
                                    self.assign_CPDs(CPDs_grad, item_name, -np.inf)
                                else:
                                    self.assign_CPDs(CPDs_grad, item_name, np.log(cum_node_prob) - np.log(self.get_CPDs(item_name)))
                                
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[node]
                        item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                        if self.check_item(item_name):
                            if self.get_CPDs(item_name) == 0.0:
                                self.assign_CPDs(CPDs_grad, item_name, np.log(cum_root_prob[bipart_bitarr_node[node.get_sisters()[0]].to01()]) + \
                                                upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                            else:
                                cum_node_prob = cum_root_prob_real[bipart_bitarr_node[node.get_sisters()[0]].to01()]
                                self.assign_CPDs(CPDs_grad, item_name, cum_node_prob - np.log(self.get_CPDs(item_name)))
                    else:
                        for sister in node.get_sisters():
                            comb_parent_bipart_bitarr = nodetobitMap[sister] + nodetobitMap[node]
                            item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                            if self.check_item(item_name):
                                if self.get_CPDs(item_name) == 0.0:
                                    self.assign_CPDs(CPDs_grad, item_name, np.log(cum_root_prob[self.toBitArr.minor((~nodetobitMap[node]) ^ nodetobitMap[sister]).to01()]) + \
                                             upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                                else:
                                    cum_node_prob = cum_root_prob_real[self.toBitArr.minor(nodetobitMap[sister] ^ (~nodetobitMap[node])).to01()]
                                    self.assign_CPDs(CPDs_grad, item_name, cum_node_prob - np.log(self.get_CPDs(item_name)))
                    
                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                    if self.check_item(item_name):
                        if self.get_CPDs(item_name) == 0.0:
                            self.assign_CPDs(CPDs_grad, item_name, upper_clip(bipart_bitarr_prob[bipart_bitarr.to01()] - loglikelihood, entry_ub))
                        else:
                            self.assign_CPDs(CPDs_grad, item_name, root_prob_real[bipart_bitarr.to01()] - np.log(self.get_CPDs(item_name)))
                
                if not node.up.is_root():
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                    
                if not node.is_leaf():
                    for child in node.children:
                        comb_parent_bipart_bitarr = nodetobitMap[child] + ~nodetobitMap[node]
                        item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                        if self.check_item(item_name):
                            if self.get_CPDs(item_name) == 0.0:
                                self.assign_CPDs(CPDs_grad, item_name, np.log(cum_root_prob[self.toBitArr.minor(nodetobitMap[node] ^ nodetobitMap[child]).to01()]) + \
                                        upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                            else:
                                cum_node_prob = cum_root_prob_real[self.toBitArr.minor(nodetobitMap[node] ^ nodetobitMap[child]).to01()]
                                self.assign_CPDs(CPDs_grad, item_name, cum_node_prob - np.log(self.get_CPDs(item_name)))
                
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                if self.check_item(item_name):
                    if self.get_CPDs(item_name) == 0.0:
                        self.assign_CPDs(CPDs_grad, item_name, upper_clip(bipart_bitarr_prob[bipart_bitarr.to01()] - loglikelihood, entry_ub))
                    else:
                        self.assign_CPDs(CPDs_grad, item_name, root_prob_real[bipart_bitarr.to01()] - np.log(self.get_CPDs(item_name)))
        
        CPDs_grad = np.exp(CPDs_grad)
        dict_length = self.CPDs_parser.dict_len
        dict_sum_grad = dict_sum(self.CPDs*CPDs_grad, dict_length)
        CPDs_grad = (CPDs_grad - dict_sum_grad) * self.CPDs
        
        if not value_and_grad:
            return CPDs_grad
        else:
            return loglikelihood, CPDs_grad
        

    def sample_tree(self, rooted=False):
        """ Sampling from SBN (ancestral sampling) """
        
        root = Tree()
        node_split_stack = [(root, '0'*self.ntaxa + '1'*self.ntaxa)]
        for i in range(self.ntaxa-1):
            node, split_bitarr = node_split_stack.pop()
            parent_clade_bitarr = bitarray(split_bitarr[self.ntaxa:])
            if node.is_root():
                split_candidate, split_prob = zip(*self.rootsplit_CPDs.items())
            else:
                split_candidate, split_prob = zip(*self.subsplit_CPDs(split_bitarr).items())
            
            split = np.random.choice(split_candidate, p=split_prob)                
            comp_split = (parent_clade_bitarr ^ bitarray(split)).to01()
            
            c1 = node.add_child()
            c2 = node.add_child()
            if split.count('1') > 1:
                node_split_stack.append((c1, comp_split + split))
            else:
                c1.name = self.taxa[split.find('1')]
            if comp_split.count('1') > 1:
                node_split_stack.append((c2, split + comp_split))
            else:
                c2.name = self.taxa[comp_split.find('1')]
        
        if not rooted:
            root.unroot()
        
        return root          
    
 
    @staticmethod  
    def aggregate_CPDs_grad(wts, CPDs_grad, clip=None, choose_one=False):
        """
        Aggregate gradients from multiple sampled tree topologies.
        
        Parameters
        ----------
        wts : ``np.array``, the weight vector for sampled tree topologies.
        CPDs_grad : ``np.ndarray``, the gradient matrix of CPDs.
        clip : ``float``, the bound for clipping the gradient.
        choose_one: ``boolean``, optional for reweighted weak sleep (RWS).
        """
        
        n_particles = len(CPDs_grad)
        if not choose_one:
            agg_CPDs_grad = np.sum(wts.reshape(n_particles, 1) * CPDs_grad, 0)
        else:
            if choose_one == 'sample':
                samp_index = np.random.choice(np.arange(n_particles), p=wts)
            elif choose_one == 'max':
                samp_index = wts.argmax()
            else:
                raise NotImplementedError
            agg_CPDs_grad = CPDs_grad[samp_index]
        if clip:
            agg_CPDs_grad = np.clip(agg_CPDs_grad, -clip, clip)
        return agg_CPDs_grad
            

class SBN_VI_EMP(SBN):
    """
    Training SBNs for empirical distributions over tree topologies only.
        
    Parameters
    ----------
    taxa : ``list``, a list of the labels of the sequences.
    emp_tree_freq : ``dict``, a dictionary of the empirical probabilities
                     of tree topologies.
    rootsplit_supp_dict : ``dict``, a dictionary of rootsplit support,
                          usually obtained from some bootstrap run.
    subsplit_supp_dict: ``dict``, a dictionary of subsplit support,
                         obtained similarly as above.
    
    References
    ----------
    .. [1] Cheng Zhang and Frederick A. Matsen IV. "Variational Bayesian Phylogenetic
           Inference",  In Proceedings of the 7th International Conference on Learning
           Representations (ICLR), 2019. (https://openreview.net/forum?id=SJVmjjR9FX)
    
    """
    
    EPS = 1e-100
    def __init__(self, taxa, emp_tree_freq, rootsplit_supp_dict, subsplit_supp_dict):
        super(SBN_VI_EMP, self).__init__(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.trees , self.emp_freqs = zip(*emp_tree_freq.items())
        self.emp_freqs = np.array(self.emp_freqs)
        self.emp_tree_freq = emp_tree_freq
        self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        self._data_loglikelihood = defaultdict(lambda: -np.inf)
        
        for tree, value in emp_tree_freq.items():
            if value > 0.0:
                self._data_loglikelihood[tree.get_topology_id()] = np.log(value)
        
        
        
    
    # compute the KL divergence from SBN to the target empirical distribution.   
    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.items():
            kl_div += wt * np.log(max(np.exp(self.tree_loglikelihood(tree)), self.EPS))
        kl_div = self.negDataEnt - kl_div
        return kl_div

    
    def data_loglikelihood(self, tree):
        return self._data_loglikelihood[tree.get_topology_id()]
    
    # n-sample lower bound estimates   
    def lower_bound_estimate(self, n_particles, rooted=False, sample_sz=1000):
        lower_bound = np.empty(sample_sz)
        for sample in range(sample_sz):
            samp_tree_list = [self.sample_tree(rooted=rooted) for particle in range(n_particles)]
            loglikelihood = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])  
            
            lower_bound[sample] = logmeanexp(log_prob_ratios)
        
        return np.mean(lower_bound[~np.isinf(lower_bound)])
        

    def rws(self, stepsz, maxiter=200000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, anneal_freq=20000, anneal_rate=0.75, alpha=1.0,
            n_particles=20, clip=100., momentum=0.9, decay=0.0, sgd_solver='adam', sample_particle=False):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        infer_opt = SGD_Server({'CPDs':self.num_CPDs}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz}
            
        run_time = -time.time()
        for it in range(1, maxiter+1):
            samp_tree_list = [self.sample_tree(rooted) for particle in range(n_particles)]
            loglikelihood = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = np.array(logq_tree), np.concatenate(CPDs_grad).reshape(n_particles, -1)
            # log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            log_prob_ratios = loglikelihood - logq_tree
            particle_wts = softmax(alpha*log_prob_ratios)
            lower_bound = logmeanexp(log_prob_ratios)
            
            # CPDs_grad = self.aggregate_CPDs_grad(particle_wts, samp_tree_list, clip=clip, choose_one=sample_particle)
            CPDs_grad = self.aggregate_CPDs_grad(particle_wts, CPDs_grad, clip=clip, choose_one=sample_particle)
            self._CPDs = self._CPDs + getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs}, {'CPDs': CPDs_grad})['CPDs']
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            
            lls.append(np.max(loglikelihood))
            lbs.append(lower_bound)
            
            if it % anneal_freq == 0:
                # stepsz *= anneal_rate
                for var in stepsz:
                    stepsz[var] *= anneal_rate
            
            if it % test_freq == 0:
                run_time += time.time()
                test_kl_div.append(self.kl_div())
                print('Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1]))
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lower_bound.append(self.lower_bound_estimate(n_particles, sample_sz=lb_test_sampsz)) 
                    run_time += time.time()  
                    print('>>> Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(it, run_time, test_lower_bound[-1]))
                                 
                lbs, lls = [], []
                run_time = -time.time()
            
        return test_kl_div, test_lower_bound
    
    
    def rwsvr(self, stepsz, maxiter=1000, biter=1000, batch_F=1000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, anneal_freq=20000, anneal_rate=0.75, alpha=1.0, n_particles=20, clip=100., momentum=0.9, decay=0.0, sgd_solver='adam', sample_particle=False):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        infer_opt = SGD_Server({'CPDs':self.num_CPDs}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz}
            
        run_time = -time.time()
        for it in range(1, maxiter+1):
            samp_tree_list_F = [self.sample_tree(rooted) for particle in range(batch_F)]
            CPDs_F = self.CPDs
            loglikelihood_F = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list_F])
            logq_tree_F, CPDs_grad_F = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list_F])
            logq_tree_F, CPDs_grad_F = np.array(logq_tree_F), np.concatenate(CPDs_grad_F).reshape(batch_F, -1)
            log_prob_ratios_F = loglikelihood_F - logq_tree_F
            particle_wts_F = softmax(alpha*log_prob_ratios_F)
            lower_bound_F = logmeanexp(log_prob_ratios_F)
            CPDs_grad_F = self.aggregate_CPDs_grad(particle_wts_F, CPDs_grad_F, clip=clip, choose_one=sample_particle)
            
            for bit in range(1, biter+1):
                tit = (it-1)*biter + bit
                CPDs_f = self.CPDs
                samp_tree_list_f = [self.sample_tree(rooted) for particle in range(n_particles)]
                loglikelihood_f = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list_f])
                logq_tree_f, CPDs_grad_f = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list_f])
                logq_tree_f, CPDs_grad_f = np.array(logq_tree_f), np.concatenate(CPDs_grad_f).reshape(n_particles, -1)
                # log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
                log_prob_ratios_f = loglikelihood_f - logq_tree_f
                particle_wts_f = softmax(alpha*log_prob_ratios_f)
                lower_bound_f = logmeanexp(log_prob_ratios_f)
                CPDs_grad_f = self.aggregate_CPDs_grad(particle_wts_f, CPDs_grad_f, clip=clip, choose_one=sample_particle)
                
                self.CPDs = CPDs_F
                loglikelihood_f0 = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list_f])
                logq_tree_f0, CPDs_grad_f0 = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list_f])
                logq_tree_f0, CPDs_grad_f0 = np.array(logq_tree_f0), np.concatenate(CPDs_grad_f0).reshape(n_particles, -1)
                # log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
                log_prob_ratios_f0 = loglikelihood_f0 - logq_tree_f0
                particle_wts_f0 = softmax(alpha*log_prob_ratios_f0)
                lower_bound_f0 = logmeanexp(log_prob_ratios_f0)
                CPDs_grad_f0 = self.aggregate_CPDs_grad(particle_wts_f0, CPDs_grad_f0, clip=clip, choose_one=sample_particle)
                
                self.CPDs = CPDs_f
                self._CPDs = self._CPDs + getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs}, {'CPDs': CPDs_grad_f- CPDs_grad_f0+ CPDs_grad_F})['CPDs']
                self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            
                lls.append(np.max(loglikelihood_f))
                lbs.append(lower_bound_f)
            
                if tit % anneal_freq == 0:
                    # stepsz *= anneal_rate
                    for var in stepsz:
                        stepsz[var] *= anneal_rate
            
                if tit % test_freq == 0:
                    run_time += time.time()
                    test_kl_div.append(self.kl_div())
                    print('Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(tit, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1]))
                    if tit % lb_test_freq == 0:
                        run_time = -time.time()
                        test_lower_bound.append(self.lower_bound_estimate(n_particles, sample_sz=lb_test_sampsz)) 
                        run_time += time.time()  
                        print('>>> Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(tit, run_time, test_lower_bound[-1]))
                                 
                    lbs, lls = [], []
                    run_time = -time.time()
            
        return test_kl_div, test_lower_bound
    

    def vimco(self, stepsz, maxiter=200000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, anneal_freq=20000, anneal_rate=0.75, n_particles=20, clip=100., momentum=0.9, decay=0.0, sgd_solver='adam'):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        infer_opt = SGD_Server({'CPDs': self.num_CPDs}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz}
            
        run_time = -time.time()
        for it in range(1, maxiter+1):
            samp_tree_list = [self.sample_tree(rooted=rooted) for particle in range(n_particles)]
            loglikelihood = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = np.array(logq_tree), np.concatenate(CPDs_grad).reshape(n_particles, -1)
            # log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            log_prob_ratios = loglikelihood - logq_tree
            
            particle_wts = softmax(log_prob_ratios)
            lower_bound = logmeanexp(log_prob_ratios)
            lower_bound_approx = logmeanexp(log_prob_ratios, exclude=True)
            
            if lower_bound == -np.inf:
                update_wts = -particle_wts
            else:
                update_wts = lower_bound - lower_bound_approx - particle_wts
                update_wts[np.isposinf(update_wts)] = 20.
            
            CPDs_grad = self.aggregate_CPDs_grad(update_wts, CPDs_grad, clip=clip)
            self._CPDs = self._CPDs + getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs}, {'CPDs': CPDs_grad})['CPDs']
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            
            lls.append(np.max(loglikelihood))
            lbs.append(lower_bound)
            
            if it % anneal_freq == 0:
                for var in stepsz:
                    stepsz[var] *= anneal_rate
            
            if it % test_freq == 0:
                run_time += time.time()
                test_kl_div.append(self.kl_div())                   
                print('Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1]))
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lower_bound.append(self.lower_bound_estimate(n_particles, sample_sz=lb_test_sampsz))
                    run_time += time.time()
                    print('>>> Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(it, run_time, test_lower_bound[-1]))
                    
                lbs, lls = [], []
                run_time = -time.time()
                    
        return test_kl_div, test_lower_bound