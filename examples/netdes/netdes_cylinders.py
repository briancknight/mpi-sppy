# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import sys
import os
import copy
import netdes

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla
from mpisppy.extensions.cross_scen_extension import CrossScenarioExtension

write_solution = True

def _parse_args():
    parser = baseparsers.make_parser(num_scens_reqd=False)
    parser.add_argument("--instance-name",
                        help="netdes instance name (e.g., network-10-20-L-01)",
                        dest="instance_name",
                        type=str,
                        default=None)                
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagrangian_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser = baseparsers.slamup_args(parser)
    parser = baseparsers.cross_scenario_cuts_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    inst = args.instance_name
    num_scen = int(inst.split("-")[-3])
    if args.num_scens is not None and args.num_scens != num_scen:
        raise RuntimeError("Argument num-scens={} does not match the number "
                           "implied by instance name={} "
                           "\n(--num-scens is not needed for netdes)")

    with_fwph = args.with_fwph
    with_xhatlooper = args.with_xhatlooper
    with_xhatshuffle = args.with_xhatshuffle
    with_lagrangian = args.with_lagrangian
    with_slamup = args.with_slamup
    with_cross_scenario_cuts = args.with_cross_scenario_cuts

    if args.default_rho is None:
        raise RuntimeError("The --default-rho option must be specified")

    path = f"{netdes.__file__[:-10]}/data/{inst}.dat"
    scenario_creator = netdes.scenario_creator
    scenario_denouement = netdes.scenario_denouement    
    all_scenario_names = [f"Scen{i}" for i in range(num_scen)]
    scenario_creator_kwargs = {"path": path}

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    if with_cross_scenario_cuts:
        ph_ext = CrossScenarioExtension
    else:
        ph_ext = None

    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=ph_ext,
                              rho_setter = None)

    if with_cross_scenario_cuts:
        hub_dict["opt_kwargs"]["options"]["cross_scen_options"]\
            = {"check_bound_improve_iterations" : args.cross_scenario_iter_cnt}

    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if with_lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = None)
        
    # xhat looper bound spoke
    if with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # slam up bound spoke
    if with_slamup:
        slamup_spoke = vanilla.slamup_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # cross scenario cuts spoke
    if with_cross_scenario_cuts:
        cross_scenario_cuts_spoke = vanilla.cross_scenario_cuts_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    list_of_spoke_dict = list()
    if with_fwph:
        list_of_spoke_dict.append(fw_spoke)
    if with_lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if with_xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if with_slamup:
        list_of_spoke_dict.append(slamup_spoke)
    if with_cross_scenario_cuts:
        list_of_spoke_dict.append(cross_scenario_cuts_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if write_solution:
        wheel.write_first_stage_solution('netdes_build.csv')
        wheel.write_tree_solution('netdes_full_solution')


if __name__ == "__main__":
    main()
