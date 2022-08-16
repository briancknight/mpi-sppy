# extending coverage experiments to aircond using a tight CI as a proxy for zstar
from mpi4py import MPI
import argparse
import aircond_submodels
import numpy as np
from mpisppy.confidence_intervals import mmw_ci as mmw_ci
from mpisppy.confidence_intervals import sample_tree
from mpisppy.confidence_intervals import ciutils
from mpisppy.utils.xhat_eval import Xhat_Eval
import mpisppy.utils.amalgomator as ama
from mpisppy.confidence_intervals.seqsampling import SeqSampling
import scipy
from mpisppy.utils import sputils
import time
import sys
from mpisppy.confidence_intervals.multi_seqsampling import IndepScens_SeqSampling

comm = MPI.COMM_WORLD
size = comm.Get_size()
global_rank = comm.Get_rank()

def evaluate_sample_trees(xhat_one, 
                          num_samples,
                          ama_options,  
                          SeedCount=0,  
                          mname='mpisppy.tests.examples.aircond_submodels'):
    ''' creates batch_size sample trees with first-stage solution xhat_one
    using SampleSubtree class from sample_tree
    used to approximate E_{xi_2} phi(x_1, xi_2) for confidence interval coverage experiments
    '''
    seed = SeedCount
    zhats=[]
    bfs = ama_options["branching_factors"]
    cb_dict=ama_options['cb_dict']
    solvername = ama_options["EF_solver_name"]
    #sampling_bfs = ciutils.scalable_BFs(batch_size, bfs) # should i be using this? How to compute variance?
    xhat_eval_options = {"iter0_solver_options": None,
                     "iterk_solver_options": None,
                     "display_timing": False,
                     "solvername": solvername,
                     "verbose": False,
                     "solver_options":{}}

    for j in range(num_samples): # number of sample trees to create
        scenario_creator_kwargs={'branching_factors':bfs, 'start_seed':seed,'cb_dict':cb_dict}
        samp_tree = sample_tree.SampleSubtree(mname,
                                              xhats =[],
                                              root_scen=None,
                                              starting_stage=1, 
                                              BFs=bfs,
                                              seed=seed, 
                                              options=scenario_creator_kwargs,
                                              solvername=solvername,
                                              solver_options={})
        samp_tree.run()
        # seed += sputils.number_of_nodes(bfs)
        ama_object = samp_tree.ama
        ama_options = ama_object.options
        # ama_options['start_seed'] = seed
        ama_options['verbose'] = False
        scenario_creator_kwargs=ama_object.kwargs
        if len(samp_tree.ef._ef_scenario_names)>1:
            local_scenarios = {sname:getattr(samp_tree.ef,sname) for sname in samp_tree.ef._ef_scenario_names}
        else:
            local_scenarios = {samp_tree.ef._ef_scenario_names[0]:samp_tree.ef}

        xhats,seed = sample_tree.walking_tree_xhats(mname,
                                                    local_scenarios,
                                                    xhat_one,
                                                    bfs,
                                                    seed,
                                                    ama_options,
                                                    solvername=solvername,
                                                    solver_options=None)
        # for Xhat_Eval
        # scenario_creator_kwargs = ama_object.kwargs
        scenario_names = ama_object.scenario_names
        all_nodenames = sputils.create_nodenames_from_BFs(bfs)

        # Evaluate objective value of feasible policy for this tree
        ev = Xhat_Eval(xhat_eval_options,
                        scenario_names,
                        ama_object.scenario_creator,
                        aircond_submodels.scenario_denouement,
                        scenario_creator_kwargs=scenario_creator_kwargs,
                        all_nodenames=all_nodenames)

        zhats.append(ev.evaluate(xhats))

    return np.array(zhats), seed


def seqsampling_coverage_aircond(num_trials, 
                                 num_samples, 
                                 options, 
                                 zstarCI, 
                                 start_seed=0,
                                 confidence_level=0.9,
                                 stopping_criterion='BPL',
                                 print_results=True):
    '''
    num_batches is number of repetitions of the experiment to run
    batch_size is the number of sample trees to use in each rep
    '''
    # # just in case we lose this information:
    if global_rank==0:
        print('\nseqsampling_coverage_aircond args: ', 
            num_trials, num_samples, options, 
            zstarCI, start_seed, confidence_level, print_results)

    if not print_results:
        print("\n\nWARNING: this will not save final results\n\n")

    mname="mpisppy.tests.examples.aircond_submodels"
    bfs = options["BFs"]
    solvername=options["solvername"]
    solver_options=options["solver_options"]

    start_ups = options["cb_dict"]["start_ups"]
    seed=start_seed

    # set up ama_options for sampling to approximate E phi(x_1, xi_2)):
    if len(bfs) > 1:
        twostage=False
        mstage=True
        solving_type="EF-mstage"
    else:
        twostage=True
        mstage=False
        solving_type="EF-2stage"

    options["EF-2stage"] = twostage
    options["EF-mstage"] = mstage
    print
    ama_options = { "EF-2stage": twostage,
                    "EF-mstage": mstage,
                    "EF_solver_name": solvername,
                    "EF_solver_options":solver_options,
                    "solvername": solvername,
                    "num_scens": np.prod(bfs),
                    "_mpisppy_probability": None,
                    "branching_factors":bfs,
                    "mudev":options['xhat_gen_options']['mudev'],
                    "sigmadev":options['xhat_gen_options']['sigmadev'],
                    "cb_dict":options['cb_dict'],
                    "verbose":False
                    }
    # only BPL is converging for multi-stage currently...
    aircond_pb = IndepScens_SeqSampling(mname,
                            aircond_submodels.xhat_generator_aircond, 
                            options,
                            stopping_criterion=stopping_criterion,
                            )

    aircond_pb.SeedCount=seed
    data = np.zeros((num_trials, 5))

    if start_ups:
        xhat_ones = np.zeros((num_trials,3))
    else:
        xhat_ones = np.zeros((num_trials,2))

    # run seq sampling procedure num_batches times
    for i in range(num_trials):
        prev_count = aircond_pb.SeedCount
        t0 = time.time()
        res = aircond_pb.run()
        time_elapsed = time.time()-t0
        #increase start seed
        seeds_used = aircond_pb.SeedCount - prev_count
        seed += seeds_used
        xhat_ones[i] = res["Candidate_solution"]['ROOT']
        G = res["CI"][1]
        T = res["T"]

        # # sample batch_size scenario trees for each batch:
        # # jump to next multiple of num_scens for xhat_eval object
        # seed += ama_options['num_scens'] - seed%ama_options['num_scens']
        # zhats, seed = evaluate_sample_trees(xhat_ones[i], num_samples, ama_options, SeedCount=seed) 
        # # Find average zhat over all sample trees, CI (approximate E phi(x_1, xi_2))
        # zhatbar = np.mean(zhats)
        # s_zhat = np.std(np.array(zhats))
        # t_zhat = scipy.stats.t.ppf(confidence_level, len(zhats)-1)
        # eps_z = t_zhat*s_zhat/np.sqrt(len(zhats))

        # ub = zhatbar+eps_z
        # lb = zhatbar - G - eps_z

        # # check if this CI contains tight CI zstarCI from construct_candidate_CI:
        # if zstarCI[1] < zhatbar + eps_z  and zstarCI[0] > zhatbar - G - eps_z:
        #     covered = 1
        #     msg = 'covered in: '
        # else: 
        #     covered = 0
        #     msg = 'not covered in: '

        # data[i] = [zhatbar, lb, ub, covered, G, T, seeds_used, time_elapsed, time.time() - t0]
        data[i] = [G, T, seeds_used, time_elapsed, time.time() - t0]

        if global_rank==0:
            print(res)
            # print(msg,[zhatbar - G - eps_z, zhatbar + eps_z ])
            print('\ntrial {} finished in {} seconds\n'.format(i+1, time.time() - t0))

    if print_results:
        if global_rank==0:

            bf_string=''
            for factor in bfs:
                bf_string=bf_string+str(factor)+'_'

            if stopping_criterion=='BPL':
                savstr = 'IndepScensResults/aircond_'+str(start_ups)+'_'+bf_string+'seed='+\
                str(start_seed)+'_num_trials='+str(num_trials)+\
                '_num_samples'+str(num_samples)+'_sample_size'+str(num_samples)+\
                '_eps='+str(options['eps'])
            else:
                savstr = 'IndepScensResults/aircond_'+str(start_ups)+'_'+bf_string+'seed='+\
                str(start_seed)+'_num_trials='+str(num_trials)+\
                '_num_samples'+str(num_samples)+'_sample_size'+str(num_samples)+\
                '_h='+str(options['h'])+'_h='+str(options['hprime'])+'_'

            # np.savetxt(savstr+'coverage_data.txt', data)
            np.savetxt(savstr+'ss_data.txt', data)
            np.savetxt(savstr+'xhat_ones.txt', xhat_ones)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--solver-name',
                        help="solver name (default 'gurobi_direct'",
                        default='gurobi_direct')
    parser.add_argument('--num-trials',
                        help='number of repetitions of the experiment to perform',
                        dest='num_trials',
                        type=int,
                        default=0)
    parser.add_argument('--num-samples',
                        help='number of sample trees to use in each approximation of xhat evaluation',
                        dest='num_samples',
                        type=int,
                        default=10)
    parser.add_argument('--confidence-level',
                    help='number of sample trees to use in each approximation of xhat evaluation',
                    dest='confidence_level',
                    type=float,
                    default=0.9)
    parser.add_argument('--seed',
                        help="starting seed for experiments (default 0)",
                        dest='start_seed',
                        type=int,
                        default=0)
    parser.add_argument("--BFs",
                        help="Spaces delimited branching factors (default 3 3)",
                        dest="BFs",
                        nargs="*",
                        type=int,
                        default=[3,3])
    parser.add_argument("--zstarCI",
                        help="Approximate confidence interval for optimal objective function value",
                        dest="zstarCI",
                        nargs="*",
                        type=float,
                        default=None)
    parser.add_argument("--gap-percent",
                        help="Tolerance percentage (0-100) for optimality gap (for fixed width seq samp) (default 2)",
                        dest="gap_percent",
                        type=float,
                        default=2)
    parser.add_argument("--with-start-ups",
                        help="Toggles start-up costs in the aircond model",
                        dest="start_ups",
                        action="store_true")
    parser.set_defaults(start_ups=False)


    args = parser.parse_args()
    solvername=args.solver_name
    seed = args.start_seed
    bfs = args.BFs
    num_trials = args.num_trials
    num_samples = args.num_samples
    confidence_level = args.confidence_level
    gap_percent = args.gap_percent
    start_ups = args.start_ups

    if start_ups:
        solver_options={'mipgap':0.015}
    else:
        solver_options = {}

    if args.zstarCI == None:
        # read in samples to construct CI around zstar:
        try: 
            bf_string = ''
            for factor in bfs:
                bf_string=bf_string+str(factor)+'_'
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




    # options for seq sampling FW interval for coverage tests: (i.e. do we have nested intervals?)
    eps = np.floor( (zstarCI[1]+zstarCI[0])/2 * gap_percent/100) # gap_percent of approximate solution
    print('eps: ', eps)
    optionsFSP = {'eps': eps,
                    'solvername': solvername,
                    'solver_options':solver_options,
                    "c0":100,
                    "verbose":False,
                    "BFs":bfs,
                    "branching_factors": bfs,
                    "mudev":0,
                    "sigmadev":40,
                    "cb_dict":{"start_ups":start_ups},
                    "xhat_gen_options":{"BFs": bfs, "mudev":0, 'sigmadev':40,"cb_dict":{"start_ups":start_ups}},
                    }

    optionsBM_aircond1 =  { 'h':0.8,
                'hprime':0.55, 
                'eps':0.5, 
                'epsprime':0.4, 
                "p":0.2,
                "q":1.2,
                "solvername":solvername,
                "solver_options":{},
                "BFs":bfs,
                "branching_factors": bfs,
                "cb_dict":{"start_ups":False},
                "xhat_gen_options":{"BFs":bfs,"mudev":0, 'sigmadev':40,"cb_dict":{"start_ups":False}}}
    
    optionsBM_aircond2 =  {'h':1.1,
                'hprime':0.85, 
                'eps':0.5, 
                'epsprime':0.4, 
                "p":0.2,
                "q":1.2,
                "solvername":solvername,
                "solver_options":{"mipgap":0.005},
                "BFs":bfs,
                "branching_factors": bfs,
                "cb_dict":{"start_ups":True},
                "xhat_gen_options":{"BFs":bfs,"mudev":0, 'sigmadev':40,"cb_dict":{"start_ups":True}}}

    if start_ups:
        optionsBM=optionsBM_aircond2
    else:
        optionsBM=optionsBM_aircond1

    seqsampling_coverage_aircond(num_trials, 
                                 num_samples,
                                 optionsBM, 
                                 zstarCI, 
                                 start_seed = seed,
                                 confidence_level=confidence_level, 
                                 stopping_criterion='BM',
                                 print_results=True)


if __name__ == '__main__':
    main()