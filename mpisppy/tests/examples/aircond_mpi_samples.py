# extending coverage experiments to aircond using a tight CI as a proxy for zstar
from mpi4py import MPI
import aircond_submodels
import numpy as np
from mpisppy.confidence_intervals import mmw_ci as mmw_ci
from mpisppy.confidence_intervals import sample_tree
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.utils.amalgomator as ama
from mpisppy.confidence_intervals.seqsampling import SeqSampling
import scipy
from mpisppy.utils import sputils
import time
import pandas as pd
import sys
from mpisppy.utils.sputils import number_of_nodes
import argparse

comm = MPI.COMM_WORLD
size = comm.Get_size()
global_rank = comm.Get_rank()

def solve_sample_trees_LB(num_batches,  
                           options,
                           start_seed=0,
                           mname="mpisppy.tests.examples.aircond_submodels"):

    bfs = options["BFs"]
    cb_dict = options["cb_dict"]

    if len(bfs) > 1:
        twostage=False
        mstage=True
        solving_type="EF-mstage"
    else:
        twostage=True
        mstage=False
        solving_type="EF-2stage"

    # find a tight CI to use in lieu of zstar:

    ama_options = { "EF-2stage": twostage,
                    "EF-mstage": mstage,
                    "EF_solver_name": options["solvername"],
                    "EF_solver_options":options["solver_options"],
                    "solvername":options["solvername"],
                    "num_scens": np.prod(bfs),
                    "_mpisppy_probability": None,
                    "BFs":bfs,
                    "branching_factors":bfs,
                    "mudev":options['mudev'],
                    "sigmadev":options['sigmadev'],
                    "cb_dict":cb_dict,
                    "verbose":False,
                    }

    seed=start_seed
    zstars=[]
    xhat_ones=[]

    for j in range(num_batches): # number of sample trees to create
        ama_options["start_seed"] = seed
        ama_object = ama.from_module(mname, ama_options,use_command_line=False)
        ama_object.run()
        xhat_one = sputils.nonant_cache_from_ef(ama_object.ef)['ROOT']
        seed+=sputils.number_of_nodes(bfs)
        xhat_ones.append(xhat_one)
        zstars.append(ama_object.EF_Obj)

    return np.array(zstars)


def mpi_solve_sample_trees_LB(nreps, options):

    n = nreps
    zhats = np.zeros(n)
    count = n // size  # number of catchments for each process to analyze
    remainder = n % size  # extra catchments if n is not a multiple of size
    bfs = options["BFs"]
    seed=1000000

    if global_rank < remainder:  # processes with rank < remainder analyze one extra catchment
        start = global_rank * (count + 1)  # index of first catchment to analyze
        stop = start + count + 1  # index of last catchment to analyze
        seed += start * number_of_nodes(bfs)
    else:
        start = global_rank * count + remainder
        stop = start + count
        seed += start * number_of_nodes(bfs)

    #local_results = zhats[start:stop]
    #print(start)
    local_results = solve_sample_trees_LB(stop-start, options, start_seed=seed)

    # send results to rank 0
    if global_rank > 0:
        comm.Send(local_results, dest=0, tag=14)  # send results to process 0
    else:
        zstars = np.copy(local_results)  # initialize final results with results from process 0
        for i in range(1, size):  # determine the size of the array to be received from each process
            if i < remainder:
                rank_size = count + 1
            else:
                rank_size = count
            tmp = np.empty(rank_size, dtype=np.float64)
            #tmp = np.empty((rank_size, final_results.shape[1]), dtype=np.float) # create empty array to receive results
            comm.Recv(tmp, source=i, tag=14)  # receive results from the process
            zstars = np.concatenate((zstars, tmp))  # add the received results to the final results

        bf_string = ''
        for factor in bfs:
            bf_string=bf_string+str(factor)+'_'

        np.savetxt('results/aircond_start_up_'+bf_string+'zstars_nreps='+str(nreps)+'.txt', zstars)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-name",
                        help="solver name (default gurobi_direct)",
                        dest='solver_name',
                        default="gurobi_direct")

    args=parser.parse_args()
    solvername=args.solver_name

    nreps=492
    bfs = [20,20]
    mudev=0
    sigmadev=40
    options={
    "BFs": bfs, 
    "branching_factors":bfs,
    "mudev":mudev, 
    "sigmadev":sigmadev, 
    "cb_dict":{"start_ups":True},
    "solvername":solvername,
    "solver_options":{"mipgap":0.005}
    }
    confidence_level=0.99

    t0=time.time()
    mpi_solve_sample_trees_LB(nreps, options)
    print('time taken: ', time.time() - t0)
    # zstars = solve_sample_trees_LB(nreps, options)
    # print(zstars)
    # np.savetxt('results/aircond_start_up_'+bf_string+'zstars.txt', zstars)

    if global_rank==0:
        bf_string=''
        for factor in bfs:
            bf_string=bf_string+str(factor)+'_'
        zstars = np.loadtxt('results/aircond2'+bf_string+'zstars'+'_nreps='+str(nreps)+'.txt')
        zstarbar = np.mean(zstars)
        s_zstar = np.std(np.array(zstars))
        t_zstar = scipy.stats.t.ppf(confidence_level, len(zstars)-1)
        eps_z = t_zstar*s_zstar/np.sqrt(len(zstars))

        print('99 percent confidence interval: ', [zstarbar - eps_z, zstarbar + eps_z])
