# extending coverage experiments to aircond using a tight CI as a proxy for zstar
from mpi4py import MPI
import argparse
import mpisppy.tests.examples.gbd.gbd
import numpy as np
from mpisppy.confidence_intervals import mmw_ci as mmw_ci
from mpisppy.confidence_intervals import sample_tree
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.utils.amalgomator as ama
import scipy
from mpisppy.utils import sputils
import time
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
global_rank = comm.Get_rank()

def gbd_mmw_coverage_experiment(num_samples, sample_size, zstar, xhat_ones,start=0,solvername='gurobi_direct'):

    refmodel='mpisppy.tests.examples.gbd.gbd'

    nreps=len(xhat_ones)
    num_scens=sample_size
    start+=num_scens
    options = { "EF-2stage": True,
                    "EF_solver_name": solvername,
                    "EF_solver_options":None,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    }
    options['solver_options'] = options['EF_solver_options']

    data = np.zeros((nreps, 9))

    for i in range(nreps):
        mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, {'ROOT':xhat_ones[i]}, num_samples,
            batch_size=sample_size, start=start,verbose=False, confidence_interval=0.9)
        t0 = time.time()
        r=mmw.run(objective_gap=True)
        mmw_time = time.time()-t0

        start+=num_samples*sample_size
        if global_rank == 0:
            print(r)
        lb = r['gap_outer_bound']
        ub = r['gap_inner_bound']
        Gbar = r['Gbar']
        Gstd = r['std']
        zhat = r['zhat_bar']
        zhat_std = r['std_zhat']
        if lb-zstar <= 0 and ub - zstar >= 0:
            msg = 'covered in: '
            covered = 1
        else:
            msg= 'not covered in: '
            covered = 0
        if global_rank==0:
            print(msg, [lb, ub])
            print('\ntrial {} finished in {} seconds\n'.format(i+1, time.time() - t0))

        data[i] = [zhat, zhat_std, lb, ub, covered, Gbar, Gstd, mmw_time, time.time()-t0]

    return data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-name",
                        help="solver name (default gurobi_direct",
                        default="gurobi_direct")
    parser.add_argument("--seed",
                        help="starting seed for experiments (default 0)",
                        dest='start_seed',
                        type=int,
                        default=0)
    parser.add_argument("--MMW-num-batches",
                            help="number of batches used for MMW confidence interval (default 2)",
                            dest="num_batches",
                            type=int,
                            default=2)
    parser.add_argument("--MMW-batch-size",
                            help="batch size used for MMW confidence interval,\
                             if None then batch_size = num_scens (default to None)",
                            dest="batch_size",
                            type=int,
                            default=None) #None means take batch_size=num_scens
    parser.add_argument("--group",
                        help="group used to split across nodes (for use w/ SLURM)",
                        dest="group",
                        type=int,
                        default=0)

    args = parser.parse_args()
    # unpack args
    solvername=args.solver_name
    seed = args.start_seed
    num_samples = args.num_batches
    sample_size = args.batch_size
    group=args.group
    zstar = 1655.63 # solution provides from Seq Samp
    xhat_string = '../results/gbd_seed=0_num_trials=100_num_samples100_sample_size300_stop=BPL_kf_Gs=1_kf_xhat=1_'
    xhat_ones = np.loadtxt(xhat_string+'xhat_ones.txt')[25*(group):25*(group+1)]
    print(xhat_ones)
    
    data = gbd_mmw_coverage_experiment(num_samples, sample_size, zstar, xhat_ones, 
        solvername=solvername,start=group*num_samples*sample_size)


    if global_rank==0:
        savstr = xhat_string
        np.savetxt(savstr+'mmw_coverage_data_group'+str(group)+'.txt', data)
        print(f'calculated coverage, out of {len(data[:,4])} trials: ', sum(data[:,4])/len(data[:,4]))

if __name__ == '__main__':
    main()
