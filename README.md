### <div align="center"><font size=5>Improving Tree Probability Estimation with Stochastic Optimization and Variance Reduction</font><div> 

<div align="center">Tianyu Xie*, Musu Yuan*, Minghua Deng, Cheng Zhang</div>

## Installation
This repository is a light CPU-based implementation of the paper titled "Improving Tree Probability Estimation with Stochastic Optimization and Variance Reduction" ([Paper](https://link.springer.com/article/10.1007/s11222-024-10498-2)) ([Arxiv](https://arxiv.org/abs/2409.05282)).
To create the python environment, use the following Anaconda command:
```
conda env create -f environment.yml
```



## Experiments on Tree Topology Probability Estimation

**Simulation**  You can use the following command to reproduce the simulation studies (e.g., SEMVR, $\beta=0.004$, $K=1500$, $T=1000$):
```
cd ./SBN/simulation
python main.py --method SEMVR --beta 0.004 --n_trees 1500 --k_all 1000
```
Here, the "--method" argument can take string value in "EM, SEM, SEMVR, SGA, SVRG". For ablation studies, you can freely specify the Dirichlet parameter ($\beta$) in "--beta", the number of trees in the training set ($K$) in "--n_trees", and the number of iterations per epoch ($T$) in "--k_all".

**Real Data** You can use the following command to reproduce the experiments on real data (e.g., SEMVR on DS1, repo1):
```
cd ./SBN/real-data
python main.py --method SEMVR --dataset 1 --repo 1 --monitor
```
Here, the "--method" argument can take string value in "EM, SEM, SEMVR, SGA, SVRG", the "--dataset" argument can take integer value from 1 to 8, and the "--repo" argument can take integer value from 1 to 10.
Please pay attention to the learning rates for different methods when running the code, by specifying the "--ema_rate" for EM-based methods and "--lr" for gradient based methods.


## Experiments on Variational Bayesian Phylogenetic Inference (VBPI)

**Simulation** You can use the following command to reproduce the results of simulation studies on VBPI:
```
cd ./VBPI/simulation
python main.py --method rwsvr --nparticles 10
```
Here, the "--method" argument can take string value in "vimco, rws, rwsvr". The "--nParticle" argument ($R$) can be alternatively specified, but a value of 10 is suggested.

**Real Data** You can use the following command to reproduce the experiments on real data (e.g., DS1):
```
cd ./VBPI/real-data
python main.py --dataset DS1 --nParticle 10 --maxIter 200000 ##VIMCO
python main.py --dataset DS1 --rws --nParticle 10 --maxIter 200000 ##RWS
python main.py --dataset DS1 --rwsvr --nParticle 10 --maxFIter 2000 --biter 100 --batch_F 1000 ##RWSVR
```
For the RWSVR method, the learning schedule is controlled by the number of epochs ("--maxFIter"), the number of iterations per epoch ($T$, "--biter"), and the epoch sample size ($F$, "--batch_F").
You can also use the "--empFreq" argument to moniter the KL divergence to the ground truth during the training.

### References
- Zhang, Cheng, and Frederick A. Matsen IV. "Generalizing tree probability estimation via Bayesian networks." Advances in neural information processing systems 31 (2018).
- Zhang, Cheng. "Improved variational Bayesian phylogenetic inference with normalizing flows." Advances in neural information processing systems 33 (2020): 18760-18771.

--- 

If you find this codebase useful, please consider citing our work:
```
@article{xie2024improving,
  title={Improving tree probability estimation with stochastic optimization and variance reduction},
  author={Xie, Tianyu and Yuan, Musu and Deng, Minghua and Zhang, Cheng},
  journal={Statistics and Computing},
  volume={34},
  number={186},
  pages={1--18},
  year={2024},
  publisher={Springer}
}
```