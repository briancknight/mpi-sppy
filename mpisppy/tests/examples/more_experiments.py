from mpisppy.tests.examples import aircond_submodels 
from mpisppy.confidence_intervals.multi_seqsampling import IndepScens_SeqSampling
from mpisppy.confidence_intervals import mmw_ci
from mpisppy.tests.examples.aircond_coverage_sample_trees import evaluate_sample_trees
import numpy as np
import time
import scipy.stats
import argparse

def run_samples(xhat_one, ama_options, num_samples,bfs=[10,10]):

        ama_options['branching_factors']=bfs
        zhats,seed = evaluate_sample_trees(xhat_one, num_samples, ama_options, SeedCount=810000)

        if ama_options["cb_dict"]["start_ups"]:
            np.savetxt('IndepScensResults/aircond2_xhat_one='+\
                str(xhat_one[0])+'_'+str(xhat_one[1])+'_zhatstars_' + 'v2.txt', zhats)
        else:
            np.savetxt('IndepScensResults/aircond1_xhat_one='+\
                str(xhat_one[0])+'_'+str(xhat_one[1])+'_zhatstars_' + 'v3.txt', zhats)        

        confidence_level=.99
        zhatbar = np.mean(zhats)
        s_zhat = np.std(np.array(zhats))
        t_zhat = scipy.stats.t.ppf(confidence_level, len(zhats)-1)
        eps_z = t_zhat*s_zhat/np.sqrt(len(zhats))

        print('zhatbar: ', zhatbar)
        print('estimate: ', [zhatbar-eps_z, zhatbar+eps_z])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--solver-name",
                        help="solver name (default gurobi_direct)",
                        dest='solver_name',
                        default="gurobi_direct")
    parser.add_argument("--with-start-ups",
                        help="Toggles start-up costs in the aircond model",
                        dest="start_ups",
                        action="store_true")
    parser.set_defaults(start_ups=False)
    parser.add_argument("--BFs",
                        help="Spaces delimited branching factors (default 3 3)",
                        dest="BFs",
                        nargs="*",
                        type=int,
                        default=[10,10])
    parser.add_argument("--xhat-one",
                        help="Approximate confidence interval for optimal objective function value",
                        dest="xhat_one",
                        nargs="*",
                        type=float,
                        default=None)
    args = parser.parse_args()

    bfs = args.BFs
    solver_name = args.solver_name
    start_ups = args.start_ups
    xhat_one=args.xhat_one

    solver_options={}
    if start_ups:
        solver_options['mipgap']=0.015
    cb_dict = {"start_ups":start_ups}

    ama_options = { "EF-mstage": True,
                "EF_solver_name": solver_name,
                "EF_solver_options":solver_options,
                "branching_factors":bfs,
                "BFs":bfs,
                "mudev":0,
                "sigmadev":40,
                "cb_dict":cb_dict,
                "start_seed":0
                }

    refmodel = "mpisppy.tests.examples.aircond_submodels" # WARNING: Change this in SPInstances

    run_samples(xhat_one, ama_options, 3000, bfs)




