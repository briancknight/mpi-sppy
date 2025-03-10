# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# updated april 26
# mpiexec -np 2 python -m mpi4py ccopf2wood.py 2 3
# (see the first lines of main() to change instances)
import os
import numpy as np
# Hub and spoke SPBase classes
from mpisppy.opt.ph import PH
# Hub and spoke SPCommunicator classes
from mpisppy.cylinders.xhatspecific_bounder import XhatSpecificInnerBound
from mpisppy.cylinders.hub import PHHub
# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils.xhat_eval import Xhat_Eval

# the problem
import ACtree as etree
from ccopf_multistage import pysp2_callback,\
    scenario_denouement, _md_dict, FixFast, FixNever, FixGaussian
import rho_setter

import pyomo.environ as pyo
import socket
import sys
import datetime as dt

import mpi4py.MPI as mpi
comm_global = mpi.COMM_WORLD
global_rank = comm_global.Get_rank()
n_proc = comm_global.Get_size()

# =========================


def main():
    start_time = dt.datetime.now()
    # start options
    casename = "pglib-opf-master/pglib_opf_case14_ieee.m"
    # pglib_opf_case3_lmbd.m
    # pglib_opf_case118_ieee.m
    # pglib_opf_case300_ieee.m
    # pglib_opf_case2383wp_k.m
    # pglib_opf_case2000_tamu.m
    # do not use pglib_opf_case89_pegase
    egret_path_to_data = "/thirdparty/"+casename
    number_of_stages = 3
    stage_duration_minutes = [5, 15, 30]
    if len(sys.argv) != number_of_stages + 3:
        print ("Usage: python ccop_multistage bf1 bf2 iters scenperbun(!) solver")
        exit(1)
    branching_factors = [int(sys.argv[1]), int(sys.argv[2])]
    PHIterLimit = int(sys.argv[3])
    scenperbun = int(sys.argv[4])
    solvername = sys.argv[5]

    seed = 1134
    a_line_fails_prob = 0.1
    repair_fct = FixFast
    verbose = False
    # end options
    # create an arbitrary xhat for xhatspecific to use
    xhat_dict = {"ROOT": "Scenario_1"}
    for i in range(branching_factors[0]):
        xhat_dict["ROOT_"+str(i)] = "Scenario_"+str(1 + i*branching_factors[1])

    scenario_creator_kwargs = {
        "convex_relaxation": True,
        "epath": egret_path_to_data,
    }
    if scenario_creator_kwargs["convex_relaxation"]:
        # for initialization solve
        solver = pyo.SolverFactory(solvername)
        scenario_creator_kwargs["solver"] = None
        ##if "gurobi" in solvername:
            ##solver.options["BarHomogeneous"] = 1
    else:
        solver = pyo.SolverFactory(solvername)
        if "gurobi" in solvername:
            solver.options["BarHomogeneous"] = 1
        scenario_creator_kwargs["solver"] = solver
    md_dict = _md_dict(scenario_creator_kwargs["epath"])

    if verbose:
        print("start data dump")
        print(list(md_dict.elements("generator")))
        for this_branch in md_dict.elements("branch"): 
            print("TYPE=",type(this_branch))
            print("B=",this_branch)
            print("IN SERVICE=",this_branch[1]["in_service"])
        print("GENERATOR SET=",md_dict.attributes("generator"))
        print("end data dump")

    lines = list()
    for this_branch in md_dict.elements("branch"):
        lines.append(this_branch[0])

    acstream = np.random.RandomState()
        
    scenario_creator_kwargs["etree"] = etree.ACTree(number_of_stages,
                                    branching_factors,
                                    seed,
                                    acstream,
                                    a_line_fails_prob,
                                    stage_duration_minutes,
                                    repair_fct,
                                    lines)
    scenario_creator_kwargs["acstream"] = acstream

    all_scenario_names=["Scenario_"+str(i)\
                        for i in range(1,len(scenario_creator_kwargs["etree"].\
                                             rootnode.ScenarioList)+1)]
    all_nodenames = scenario_creator_kwargs["etree"].All_Nodenames()

    options = dict()
    if scenario_creator_kwargs["convex_relaxation"]:
        options["solvername"] = solvername
        if "gurobi" in options["solvername"]:
            options["iter0_solver_options"] = {"BarHomogeneous": 1}
            options["iterk_solver_options"] = {"BarHomogeneous": 1}
        else:
            options["iter0_solver_options"] = None
            options["iterk_solver_options"] = None
    else:
        options["solvername"] = solvername  # needs to be ipopt
        options["iter0_solver_options"] = None
        options["iterk_solver_options"] = None
    options["PHIterLimit"] = PHIterLimit
    options["defaultPHrho"] = 1
    options["convthresh"] = 0.001
    options["subsolvedirectives"] = None
    options["verbose"] = False
    options["display_timing"] = False
    options["display_progress"] = True
    options["iter0_solver_options"] = None
    options["iterk_solver_options"] = None
    options["branching_factors"] = branching_factors

    # try to do something interesting for bundles per rank
    if scenperbun > 0:
        nscen = branching_factors[0] * branching_factors[1]
        options["bundles_per_rank"] = int((nscen / n_proc) / scenperbun)
    if global_rank == 0:
        appfile = "acopf.app"
        if not os.path.isfile(appfile):
            with open(appfile, "w") as f:
                f.write("datetime, hostname, BF1, BF2, seed, solver, n_proc")
                f.write(", bunperank, PHIterLimit, convthresh")
                f.write(", PH_IB, PH_OB")
                f.write(", PH_lastiter, PH_wallclock, aph_frac_needed")
                f.write(", APH_IB, APH_OB")
                f.write(", APH_lastiter, APH_wallclock")
        if "bundles_per_rank" in options:
            nbunstr = str(options["bundles_per_rank"])
        else:
            nbunstr = "0"
        oline = "\n"+ str(start_time)+","+socket.gethostname()
        oline += ","+str(branching_factors[0])+","+str(branching_factors[1])
        oline += ", "+str(seed) + ", "+str(options["solvername"])
        oline += ", "+str(n_proc) + ", "+ nbunstr
        oline += ", "+str(options["PHIterLimit"])
        oline += ", "+str(options["convthresh"])

        with open(appfile, "a") as f:
            f.write(oline)
        print(oline)


    # PH hub
    options["tee-rank0-solves"] = True
    hub_dict = {
        "hub_class": PHHub,
        "hub_kwargs": {"options": None},
        "opt_class": PH,
        "opt_kwargs": {
            "options": options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": pysp2_callback,
            'scenario_denouement': scenario_denouement,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            "rho_setter": rho_setter.ph_rhosetter_callback,
            "extensions": None,
            "all_nodenames":all_nodenames,
        }
    }

    xhat_options = options.copy()
    xhat_options['bundles_per_rank'] = 0 #  no bundles for xhat
    xhat_options["xhat_specific_options"] = {"xhat_solver_options":
                                          options["iterk_solver_options"],
                                          "xhat_scenario_dict": xhat_dict,
                                          "csvname": "specific.csv"}

    ub2 = {
        'spoke_class': XhatSpecificInnerBound,
        "spoke_kwargs": dict(),
        "opt_class": Xhat_Eval,
        'opt_kwargs': {
            'options': xhat_options,
            'all_scenario_names': all_scenario_names,
            'scenario_creator': pysp2_callback,
            'scenario_denouement': scenario_denouement,
            "scenario_creator_kwargs": scenario_creator_kwargs,
            'all_nodenames': all_nodenames
        },
    }
    
    list_of_spoke_dict = [ub2]
    wheel_spinner = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel_spinner.spin()
    spcomm = wheel_spinner.spcomm
    if wheel_spinner.global_rank == 0:
        ph_end_time = dt.datetime.now()
        IB = wheel_spinner.BestInnerBound
        OB = wheel_spinner.BestOuterBound
        print("BestInnerBound={} and BestOuterBound={}".\
              format(IB, OB))
        with open(appfile, "a") as f:
            f.write(", "+str(IB)+", "+str(OB)+", "+str(spcomm.opt._PHIter))
            f.write(", "+str((ph_end_time - start_time).total_seconds()))

                
if __name__=='__main__':
    main()
