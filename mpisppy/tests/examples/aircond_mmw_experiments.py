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

def aircond_mmw_coverage_experiment(bfs, cb_dict, xhat_one, num_batches, batch_size, zstarCI, zhatCI, nreps, start_seed = 0,solvername='gurobi_direct'):

    refmodel='mpisppy.tests.examples.aircond_submodels'

    ub = zhatCI[1]
    lb = zstarCI[0]

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
                    "sigmadev":40,
                    "cb_dict":cb_dict,
                    }
    if cb_dict["start_ups"]:
        options['EF_solver_options'] = {'mipgap':0.005}

    options['solver_options'] = options['EF_solver_options']

    data = np.zeros((nreps, 9))

    for i in range(nreps):
        start = seed + num_scens + i * num_batches * seeds_per_batch

        # run mmw:
        mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, {'ROOT':xhat_one}, num_batches, start=start,batch_size=batch_size,
                            verbose=False)
        t0 = time.time()
        r=mmw.run(objective_gap=True)
        mmw_time = time.time()-t0
        if global_rank == 0:
            print(r)
        G = r['gap_inner_bound']
        print('G is: ', G)

        # lb = r['obj_lower_bound']
        # ub = r['obj_upper_bound']
        Gbar = r['Gbar']
        Gstd = r['std']
        zhat = r['zhat_bar']
        zstar= r['zstar_bar']
        zhat_std = r['std_zhat']
        zstar_std=r['std_zstar']
        #if lb-zstarCI[0] <= 0 and ub - zstarCI[1] >= 0:
        if ub - lb <= G:
            msg = 'covered in: '
            covered = 1
        else:
            msg= 'not covered in: '
            covered = 0
        if global_rank==0:
            print(msg, [ub-G, ub])
            print('\ntrial {} finished in {} seconds\n'.format(i+1, time.time() - t0))

        data[i] = [zhat, zhat_std, zstar, zstar_std, covered, Gbar, Gstd, G, mmw_time]

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
                        default=[30,30])
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
    parser.add_argument("--zhatCI",
                        help="Approximate confidence interval for optimal objective function value",
                        dest="zhatCI",
                        nargs="*",
                        type=float,
                        default=None)
    parser.add_argument("--xhat-one",
                        help="Approximate confidence interval for optimal objective function value",
                        dest="xhat_one",
                        nargs="*",
                        type=float,
                        default=None)
    parser.add_argument("--num-repetitions",
                        help="number of repetitions of experiment to run (default 1)",
                        dest="nreps",
                        type=int,
                        default=1)
    parser.add_argument("--with-start-ups",
                        help="Toggles start-up costs in the aircond model",
                        dest="start_ups",
                        action="store_true")
    parser.set_defaults(start_ups=False)

    args = parser.parse_args()
    # unpack args
    solvername=args.solver_name
    group = args.start_group
    bfs = args.BFs
    num_batches = args.num_batches
    batch_size = args.batch_size
    zstarCI = args.zstarCI
    zhatCI = args.zhatCI
    xhat_one=args.xhat_one
    nreps = args.nreps
    zstarCI = args.zstarCI
    start_ups = args.start_ups
    if start_ups:
        cb_dict={"start_ups":True}
    else:
        cb_dict={"start_ups":False}
    seeds_per_batch=sputils.number_of_nodes(bfs)

    if batch_size == None: # default to num_scens
        batch_size = np.prod(bfs)

    bf_string = ''
    for factor in bfs:
        bf_string=bf_string+str(factor)+'_'
    

    data = aircond_mmw_coverage_experiment(bfs, cb_dict, xhat_one, num_batches, batch_size, zstarCI, zhatCI,nreps,
        solvername=solvername,start_seed=group*num_batches*seeds_per_batch)

    if global_rank==0:
        savstr = 'IndepScensResults/aircond_start_ups='+str(start_ups)+'_'+bf_string+'num_batches='+str(num_batches)+'_xhat_one='+str(xhat_one[0])+'_'+str(xhat_one[1])+'_'
        np.savetxt(savstr+'mmw_coverage_data.txt', data)
        print(f'calculated coverage, out of {len(data[:,4])} trials: ', sum(data[:,4])/len(data[:,4]))

if __name__ == '__main__':
    main()
