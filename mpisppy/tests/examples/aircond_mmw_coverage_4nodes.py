# extending coverage experiments to aircond using a tight CI as a proxy for zstar
from mpi4py import MPI
import argparse
import aircond_submodels
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

def aircond_mmw_coverage_experiment(bfs, num_batches, batch_size, zstarCI, xhat_ones, nreps = None, start_seed = 0,solvername='gurobi_direct'):

    refmodel='mpisppy.tests.examples.aircond_submodels'
    if nreps == None: # if unspecified, run experiment for every xhat_one
        nreps=len(xhat_ones)

    num_scens=np.prod(bfs)
    seed = start_seed
    seeds_per_batch = sputils.number_of_nodes(bfs)
    options = { "EF-mstage": True,
                    "EF_solver_name": solvername,
                    "EF_solver_options":None,
                    "num_scens": num_scens,
                    "_mpisppy_probability": 1/num_scens,
                    "branching_factors":bfs,
                    "BFs":bfs,
                    "mudev":0,
                    "sigmadev":40
                    }
    options['solver_options'] = options['EF_solver_options']

    data = np.zeros((nreps, 9))

    for i in range(nreps):
        start = seed + num_scens + i * num_batches * seeds_per_batch
        mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, {'ROOT':xhat_ones[i]}, num_batches, start=start,batch_size=batch_size,
                            verbose=False)
        t0 = time.time()
        r=mmw.run(objective_gap=True,confidence_level=0.9)
        mmw_time = time.time()-t0
        if global_rank == 0:
            print(r)
        lb = r['gap_outer_bound']
        ub = r['gap_inner_bound']
        Gbar = r['Gbar']
        Gstd = r['std']
        zhat = r['zhat_bar']
        zhat_std = r['std_zhat']
        if lb-zstarCI[0] <= 0 and ub - zstarCI[1] >= 0:
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
    parser.add_argument("--start-group",
                        help="starting seed for experiments (default 0)",
                        dest='start_group',
                        type=int,
                        default=0)
    parser.add_argument("--BFs",
                        help="Spaces delimited branching factors (default 3 3)",
                        dest="BFs",
                        nargs="*",
                        type=int,
                        default=[3,3])
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
    parser.add_argument("--zstarCI",
                        help="Approximate confidence interval for optimal objective function value",
                        dest="zstarCI",
                        nargs="*",
                        type=float,
                        default=None)
    parser.add_argument("--num-repetitions",
                        help="number of trials of experiment to run \
                        (depends on size of xhat_one file) (default None, in which case all xhat_ones are used)",
                        dest="nreps",
                        type=int,
                        default=None)

    args = parser.parse_args()
    # unpack args
    solvername=args.solver_name
    group = args.start_group
    bfs = args.BFs
    num_batches = args.num_batches
    batch_size = args.batch_size
    zstarCI = args.zstarCI
    nreps = args.nreps
    seeds_per_batch=sputils.number_of_nodes(bfs)

    if batch_size == None: # default to num_scens
        batch_size = np.prod(bfs)

    bf_string = ''
    for factor in bfs:
        bf_string=bf_string+str(factor)+'_'
        
    if args.zstarCI == None:
        # read in samples to construct CI around zstar:
        try: 
            zstars = np.loadtxt('results/aircond_'+bf_string+'zstars.txt')
        except:
            raise RuntimeError('no data file found for these branching factors')
        # construct CI
        candidate_conf_level = 0.99
        zstarbar = np.mean(zstars)
        s_zstar = np.std(np.array(zstars))
        t_zstar = scipy.stats.t.ppf(candidate_conf_level, len(zstars)-1)
        eps_z = t_zstar*s_zstar/np.sqrt(len(zstars))

        zstarCI = [zstarbar - eps_z, zstarbar + eps_z]

    else:
        zstarCI = args.zstarCI
    xhat_ones = np.loadtxt('results/aircond_'+bf_string+'xhat_ones_0.5percent.txt')[50*(group):50*(group+1)]
    data = aircond_mmw_coverage_experiment(bfs, num_batches, batch_size, zstarCI,xhat_ones,
        solvername=solvername,start_seed=group*num_batches*seeds_per_batch)

    if global_rank==0:
        savstr = 'results/aircond_'+bf_string+'group='+str(group)+'_'
        np.savetxt(savstr+'mmw_coverage_data_0.5percent_CL=0.9.txt',data)
        print(f'calculated coverage, out of {len(data[:,3])} trials: ', sum(data[:,4])/len(data[:,4]))

if __name__ == '__main__':
    main()
