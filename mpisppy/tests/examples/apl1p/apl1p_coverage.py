# replicates coverage experiments from seqsampling for phat
from mpi4py import MPI
import apl1p
import apl1p_exact
import numpy as np
from mpisppy.confidence_intervals import mmw_ci as mmw_ci
from mpisppy.utils.xhat_eval import Xhat_Eval
from mpisppy.utils import sputils
import mpisppy.utils.amalgomator as ama
from mpisppy.confidence_intervals.seqsampling import SeqSampling
import scipy
import pyomo.environ as pyo
import time
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
global_rank = comm.Get_rank()

# exact solution to APL1P test problem given in APL1P paper pg 86 (or see ap1lp.py)
# G1 = 1800
# G2 = 1571.4
# theta = 13513.7
# zstar = 24642.3

# exact solution as given by pyomo EF solver (slightly modified problem)
xstar = {'ROOT': [1800.0, 1571.4285714285716]}
zstar = 24642.320580714182


def compute_exact_solution(solvername='gurobi_direct'):
    '''compute exact solution to EF of finite scenario model'''

    tot_scen_count = 1280
    scenario_names = ['Scenario' + str(i) for i in range(tot_scen_count)]
    solver_name = solvername

    ef = sputils.create_EF(
        scenario_names,
        apl1p_exact.scenario_creator,
        scenario_creator_kwargs=None,
        )

    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        results = solver.solve(tee=True)
    else:
        results = solver.solve(ef, tee=True, symbolic_solver_labels=True)

    zstar = pyo.value(ef.EF_Obj)
    xstar = sputils.nonant_cache_from_ef(ef)

    return xstar, zstar

def apl1p_seqsampling_coverage_experiment(
    num_batches, options, zstar,
    start_seed=0,
    refmodel = 'mpisppy.tests.examples.apl1p.apl1p',
    xhat_gen = apl1p.xhat_generator_apl1p, 
    tot_scen_count=1280,
    stopping_criterion = 'BM',  
    ArRP=1, 
    fname = None,
    print_results=True):

    seed = start_seed
    options['ArRP'] = ArRP
    apl1p_pb = SeqSampling(refmodel,
                            xhat_gen, 
                            options,
                            stopping_criterion=stopping_criterion,
                            stochastic_sampling=False,
                            )
    apl1p_pb.SeedCount=seed

    # data to save:
    xhat_ones = np.zeros((num_batches, 2))
    data = np.zeros((num_batches, 9))

    #set up Xhat_Eval object to compute E(xhat, xi) exactly (for finite scenario models)
    scenario_creator_kwargs = {}
    scenario_creator_kwargs['num_scens'] = tot_scen_count
    scenario_names = ["Scen"+str(j) for j in range(tot_scen_count)]
    
    ev = Xhat_Eval(options,
                    scenario_names,
                    apl1p_exact.scenario_creator,
                    apl1p_exact.scenario_denouement,
                    scenario_creator_kwargs=scenario_creator_kwargs)

    for i in range(num_batches):

        prev_count = apl1p_pb.ScenCount

        t0 = time.time()
        res = apl1p_pb.run()
        time_elapsed = time.time()-t0

        seeds_used = apl1p_pb.ScenCount - prev_count

        xhat_one = res["Candidate_solution"]
        T = res['T']
        G = res["CI"][1]
        print('Gap estimate: ', G)

        # compute exact objective value at xhat using Xhat_Eval
        zhat = ev.evaluate(xhat_one) 
        lb = zhat - res["CI"][1]

        if (zstar < zhat) and (zstar > lb):
            msg = "\ncovered in: "
            covered = 1
        else:
            msg = "not covered in: "
            covered = 0

        if global_rank==0:
            print(msg, [lb, zhat])

        data[i] = [zhat, lb, zhat, covered, G, T, seeds_used, time_elapsed, time.time() - t0]
        xhat_ones[i] = xhat_one['ROOT']

        if global_rank == 0:
            print('\ntrial {} finished in {} seconds\n'.format(i+1, data[i][-1]))

    # for log file
    if global_rank == 0:

        print('option keys are: ', [k for k,v in options.items()])
        print('option values are: ', [v for k,v in options.items()])
        print('{} percent coverage'.format(100 * sum(data[:,3])/num_batches))

        np.savetxt('../apl1pBM_ArRP='+str(ArRP)+'_xhat_ones.txt', xhat_ones)
        np.savetxt('../apl1pBM_ArRP='+str(ArRP)+'_coverage_data.txt', data)


def main():

    solvername='gurobi_direct'
    argcnt=0
    runMMW=False
    runSEQ=False

    if 'mmw' in sys.argv:
        runMMW=True
        argcnt+=1
    if 'seq' in sys.argv:
        runSEQ=True
        argcnt+=1
    if len(sys.argv) > argcnt+1:
        solvername=sys.argv[-1]

    #xstar, zstar = compute_exact_solution()
    if global_rank==0:
        print('\nGlobal optimal solution is: xstar =',xstar['ROOT'])
        print('Global optimal objective value is: zstar =',zstar)
        print('\n')
    # construct an xhat with specified optimality gap using seqsampling:


    if runSEQ:
        # as used in SeqSamp
        optionsBM = {'h':0.317, #0.217,
                    'hprime':0.115, #0.0015, 
                    'eps':2E-7, 
                    'epsprime':1E-7, 
                    "p":1.91E-1,
                    "q":None,#1.2, 
                    "kf_Gs":1,
                    "kf_xhat":1, # not specified in SeqSamp
                    "solvername":solvername,
                    "confidence_level":0.95,
                    "verbose": False,
                    }

        optionsFSP = {'eps': 123.0,
                      'solvername':solvername,
                      "verbose":False,
                      "confidence_level":0.95,
                      "c0":50,
                      "kf_Gs":25,
                      "kf_xhat":25,
                      }
        if global_rank==0:
           print('\n\n STARTING SRP EXPERIMENT\n\n')
        apl1p_seqsampling_coverage_experiment(100, optionsBM, zstar, stopping_criterion='BM', print_results=True)
        # mpi_seqsampling_coverage_apl1p(4, optionsBM, zstar,start_seed=0)
        if global_rank==0:
           print('\n\n DONE WITH SRP EXPERIMENT\n\n')

        print('\n\n STARTING ArRP EXPERIMENT\n\n')
        apl1p_seqsampling_coverage_experiment(100, optionsBM, zstar, stopping_criterion='BM', ArRP=2, print_results=True)
        print('\n\n DONE WITH ArRP EXPERIMENT\n\n')

    else:
        if global_rank==0:
            print('\nDid not run Sequential Sampling experiment\n')

    if runMMW:
        if global_rank==0:
            print('\n\n STARTING MMW EXPERIMENT\n\n')
        apl1p_mmw_coverage_experiment(100, 10, 100, 'apl1p_ArRP=2', confidence_level=0.95, solver_name=solvername,print_results=False)
    else:
        if global_rank==0:
            print('\nDid not run MMW experiment.\n')


        
if __name__ == '__main__':
    main()
