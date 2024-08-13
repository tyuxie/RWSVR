import numpy as np
from Bio import Phylo
# from cStringIO import StringIO
from io import StringIO
from ete3 import Tree
import copy
from collections import defaultdict, OrderedDict
from bitarray import bitarray
EPS = np.finfo(float).eps

class BitArray(object):
    def __init__(self, taxa):
        self.taxa = taxa
        self.ntaxa = len(taxa)
        self.map = {taxon: i for i, taxon in enumerate(taxa)}
        
    def combine(self, arrA, arrB):
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA 
        
    def merge(self, key):
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])
        
    def decomp_minor(self, key):
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))
        
    def minor(self, arrA):
        return min(arrA, ~arrA)
        
    def from_clade(self, clade):
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))

# generate full tree space
def generate(taxa):
    if len(taxa) == 3:
        return [Tree('(' + ','.join(taxa) + ');')]
    else:
        res = []
        sister = Tree('(' + taxa[-1] + ');')
        for tree in generate(taxa[:-1]):
            
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    node.up.add_child(sister)
                    node.detach()
                    sister.add_child(node)
                    res.append(copy.deepcopy(tree))
                    node.detach()
                    sister.up.add_child(node)
                    sister.detach()

        return res


def mcmc_treeprob(filename, data_type, truncate=None):
    mcmc_samp_tree_stats = Phylo.parse(filename, data_type)
    mcmc_samp_tree_dict = {}
    mcmc_samp_tree_name = []
    mcmc_samp_tree_wts = []
    num_hp_tree = 0
    for tree in mcmc_samp_tree_stats:
        handle = StringIO()
        Phylo.write(tree, handle, 'newick')
        mcmc_samp_tree_dict[tree.name] = Tree(handle.getvalue().strip())

        handle.close()
        mcmc_samp_tree_name.append(tree.name)
        mcmc_samp_tree_wts.append(tree.weight)
        num_hp_tree += 1

        if truncate and num_hp_tree >= truncate:
            break

    return mcmc_samp_tree_dict, mcmc_samp_tree_name, mcmc_samp_tree_wts


def summary(dataset, file_path):
    tree_dict_total = {}
    tree_dict_map_total = defaultdict(float)
    tree_names_total = []
    tree_wts_total = [] 
    n_samp_tree = 0 
    for i in range(1, 11):
        tree_dict_rep, tree_name_rep, tree_wts_rep = mcmc_treeprob(file_path + dataset + '/rep_{}/'.format(i) + dataset + '.trprobs', 'nexus')
        tree_wts_rep = np.round(np.array(tree_wts_rep) * 750001)

        for i, name in enumerate(tree_name_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_dict_map_total: 
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]

            tree_dict_map_total[tree_id] += tree_wts_rep[i]

    for key in tree_dict_map_total:
        tree_dict_map_total[key] /= 10 * 750001

    for name in tree_names_total:
        tree_wts_total.append(tree_dict_map_total[tree_dict_total[name].get_topology_id()])

    return tree_dict_total, tree_names_total, tree_wts_total

def get_support_from_mcmc(taxa, tree_dict_total, tree_names_total, tree_wts_total=None):
    rootsplit_supp_dict = OrderedDict()
    subsplit_supp_dict = OrderedDict()
    toBitArr = BitArray(taxa)
    for i, tree_name in enumerate(tree_names_total):
        tree = tree_dict_total[tree_name]
        wts = tree_wts_total[i] if tree_wts_total else 1.0
        nodetobitMap = {node:toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        for node in tree.traverse('levelorder'):
            if not node.is_root():
                rootsplit = toBitArr.minor(nodetobitMap[node]).to01()
                # rootsplit_supp_dict[rootsplit] += wts
                if rootsplit not in rootsplit_supp_dict:
                    rootsplit_supp_dict[rootsplit] = 0.0
                rootsplit_supp_dict[rootsplit] += wts
                if not node.is_leaf():
                    child_subsplit = min([nodetobitMap[child] for child in node.children]).to01()
                    for sister in node.get_sisters():
                        parent_subsplit = (nodetobitMap[sister] + nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                    if not node.up.is_root():
                        parent_subsplit = (~nodetobitMap[node.up] + nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0                        
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                        
                    parent_subsplit = (~nodetobitMap[node] + nodetobitMap[node]).to01()
                    if parent_subsplit not in subsplit_supp_dict:
                        subsplit_supp_dict[parent_subsplit] = OrderedDict()
                    if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                        subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                    subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                
                if not node.up.is_root():
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                child_subsplit = bipart_bitarr.to01()
                if not node.is_leaf():
                    for child in node.children:
                        parent_subsplit = (nodetobitMap[child] + ~nodetobitMap[node]).to01()
                        if parent_subsplit not in subsplit_supp_dict:
                            subsplit_supp_dict[parent_subsplit] = OrderedDict()
                        if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                            subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                
                parent_subsplit = (nodetobitMap[node] + ~nodetobitMap[node]).to01()
                if parent_subsplit not in subsplit_supp_dict:
                    subsplit_supp_dict[parent_subsplit] = OrderedDict()
                if child_subsplit not in subsplit_supp_dict[parent_subsplit]:
                    subsplit_supp_dict[parent_subsplit][child_subsplit] = 0.0
                subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

    return rootsplit_supp_dict, subsplit_supp_dict 