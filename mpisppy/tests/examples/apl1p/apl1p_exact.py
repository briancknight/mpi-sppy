# This solves the exact EF for the APL1P model, used in sequential sampling / MMW experiements; Aug 2021

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgomator as amalgomator
import mpisppy.utils.xhat_eval as xhat_eval
import linecache


def APL1P_exact_model_creator(seed):

    scenario_str = linecache.getline('apl1p_data/apl1p_unique_scenarios.txt',seed+1)[:-1] # drop '\n'
    scenario_info = np.fromstring(scenario_str, dtype=float, sep= ' ', count=6)
    
    #
    # Model
    #
    
    model = pyo.ConcreteModel()
    
    #
    # Parameters
    #
    
    # generator
    model.G = [1,2]
    
    # Demand level
    model.DL = [1,2,3]
    
    model.Availability = pyo.Param(model.G, within=pyo.NonNegativeReals,mutable=True)
    model.Availability[1]=scenario_info[0]
    model.Availability[2]=scenario_info[1]
      
    # Min Capacity
    cmin_init = 1000
    model.Cmin = pyo.Param(model.G, within=pyo.NonNegativeReals, initialize=cmin_init)
    
    
    # Investment, aka Capacity costs
    invest = np.array([4.,2.5])
    def investment_init(m,g):
        return(invest[g-1])

    model.Investment = pyo.Param(model.G, within=pyo.NonNegativeReals,
                             initialize=investment_init)
    
    # Operating Cost
    op_cost = np.array([[4.3,2.0,0.5],[8.7,4.0,1.0]])
    def operatingcost_init(m,g,dl):
        return(op_cost[g-1,dl-1])
    
    model.OperatingCost = pyo.Param(model.G, model.DL, within=pyo.NonNegativeReals,
                                initialize = operatingcost_init)
    
    # Demand
    # demand_outcome = [900,1000,1100,1200]
    # demand_prob = [.15,.45,.25,.15]
    # demand_cumprob = np.cumsum(demand_prob)
    # assert(max(demand_cumprob) == 1.0)
    # def demand_init(m,dl):
    #     rd = random_array[2+dl]
    #     i = np.searchsorted(demand_cumprob,rd)
    #     return demand_outcome[i]
    
    model.Demand = pyo.Param(model.DL, within=pyo.NonNegativeReals,mutable=True)
    model.Demand[1] = scenario_info[2]
    model.Demand[2] = scenario_info[3]
    model.Demand[3] = scenario_info[4]

    model._mpisppy_probability=scenario_info[-1]
    
    # Cost of unserved demand
    unserved_cost =10.0
    model.CostUnservedDemand = pyo.Param(model.DL, within=pyo.NonNegativeReals,
                                     initialize=unserved_cost)
    
    #
    # Variables
    #
    
    # Capacity of generators
    model.CapacityGenerators = pyo.Var(model.G, domain=pyo.NonNegativeReals)
    
    # Operation level
    model.OperationLevel = pyo.Var(model.G, model.DL, domain=pyo.NonNegativeReals)
    
    # Unserved demand
    model.UnservedDemand = pyo.Var(model.DL, domain=pyo.NonNegativeReals)
    
    
    #
    # Constraints
    #
    
    # Minimum capacity
    def MinimumCapacity_rule(model, g):
        return model.CapacityGenerators[g] >= model.Cmin[g]
    
    model.MinimumCapacity = pyo.Constraint(model.G, rule=MinimumCapacity_rule)
    
    # Maximum operating level
    def MaximumOperating_rule(model, g):
        return sum(model.OperationLevel[g, dl] for dl in model.DL) <= model.Availability[g] * model.CapacityGenerators[g]
    
    model.MaximumOperating = pyo.Constraint(model.G, rule=MaximumOperating_rule)
    
    # Satisfy demand
    def SatisfyDemand_rule(model, dl):
        return sum(model.OperationLevel[g, dl] for g in model.G) + model.UnservedDemand[dl] >= model.Demand[dl]
    
    model.SatisfyDemand = pyo.Constraint(model.DL, rule=SatisfyDemand_rule)
    
    #
    # Stage-specific cost computations
    #
    
    def ComputeFirstStageCost_rule(model):
        return sum(model.Investment[g] * model.CapacityGenerators[g] for g in model.G)
    
    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)
    
    
    def ComputeSecondStageCost_rule(model):
        expr = sum(
            model.OperatingCost[g, dl] * model.OperationLevel[g, dl] for g in model.G for dl in model.DL) + sum(
            model.CostUnservedDemand[dl] * model.UnservedDemand[dl] for dl in model.DL)
        return expr

    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)
    
    
    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost
    
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return(model)

    #=========

def scenario_creator(sname, num_scens=None):
    scennum   = sputils.extract_num(sname)
    model = APL1P_exact_model_creator(scennum)
    
    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.Total_Cost_Objective,
            scen_name_list=None, # Deprecated?
            nonant_list=[model.CapacityGenerators], 
            scen_model=model,
        )
    ]
    
    return(model)

def scenario_names_creator(num_scens,start=None):
    # (only for Amalgomator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        

#=========
def inparser_adder(inparser):
    # (only for Amalgomator): add command options unique to apl1p
    pass


#=========
def kw_creator(options):
    # (only for Amalgomator): linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens" : options['num_scens'] if 'num_scens' in options else None,
              }
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass


#============================
def xhat_evaluator_apl1p_exact(xhats, solvername="gurobi_direct", solver_options=None):
    '''
    '''
    scenario_names = scenario_names_creator(1280)

    xhat_eval_options = {"iter0_solver_options": None,
                         "iterk_solver_options": None,
                         "display_timing": False,
                         "solvername": solvername,
                         "verbose": False,
                         "solver_options":solver_options}

    ev = xhat_eval.Xhat_Eval(xhat_eval_options,
                            scenario_names,
                            scenario_creator,
                            scenario_denouement,
                            scenario_creator_kwargs={'num_scens':1280})

    obj_vals=[]
    for xhat in xhats:
        obj_vals.append(ev.evaluate(xhat))

    return obj_vals


def make_exact_scenarios():
    # get an explicit list of all possible demands:
    avail_outcome = ([1.,0.9,0.5,0.1],[1.,0.9,0.7,0.1,0.0])
    avail_probability = ([0.2,0.3,0.4,0.1],[0.1,0.2,0.5,0.1,0.1])
    avail_cumprob = (np.cumsum(avail_probability[0]),np.cumsum(avail_probability[1]))

    demand_outcome = [900,1000,1100,1200]
    demand_prob = [.15,.45,.25,.15]

    possible_outcomes = []
    probs=[]

    for i in range(4):
        for j in range(5):
            for k in range(4):
                for m in range(4):
                    for n in range(4):
                        prob = (np.prod([[0.2,0.3,0.4,0.1][i], [0.1,0.2,0.5,0.1,0.1][j],
                                  [.15,.45,.25,.15][k], [.15,.45,.25,.15][m],[.15,.45,.25,.15][n]]))
                        possible_outcomes.append(
                            [[1.,0.9,0.5,0.1][i], [1.,0.9,0.7,0.1,0.0][j], [900,1000,1100,1200][k],
                            [900,1000,1100,1200][m],[900,1000,1100,1200][n], prob])

    np.savetxt('apl1p_data/apl1p_unique_scenarios.txt', np.array(possible_outcomes))
    


if __name__=='__main__':

    num_scens = 1280 # full suite of scenarios
    
    ama_options = { "EF-2stage": True,
                    "EF_solver_name": 'gurobi_direct',
                    "EF_solver_options": None,
                    "num_scens": num_scens
                    }
    #We use from_module to build easily an Amalgomator object
    ama = amalgomator.from_module("mpisppy.tests.examples.apl1p.apl1p_exact",
                                  ama_options,use_command_line=False)

    ama.run()
    print('best inner bound: ', ama.best_inner_bound)
    print('best outer bound: ', ama.best_outer_bound)

    xstar = sputils.nonant_cache_from_ef(ama.ef)
    print(xstar)

    # evaluate some other feasible nonants along with solution we just computed
    #res = xhat_evaluator_apl1p_exact([xstar, {'ROOT':[2000, 1000]}, {'ROOT':[2000, 1570]}])

    #print(res)
