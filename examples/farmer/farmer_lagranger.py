# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# demonstrate the special lagranger spoke

import farmer
import mpisppy.cylinders

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import baseparsers
from mpisppy.utils import vanilla


def _parse_args():
    parser = baseparsers.make_parser(num_scens_reqd=True)
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.xhatlooper_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser = baseparsers.lagranger_args(parser)
    parser = baseparsers.xhatshuffle_args(parser)
    parser.add_argument("--crops-mult",
                        help="There will be 3x this many crops (default 1)",
                        dest="crops_mult",
                        type=int,
                        default=1)                
    args = parser.parse_args()
    return args


def main():
    
    args = _parse_args()

    num_scen = args.num_scens
    crops_multiplier = args.crops_mult
    
    rho_setter = farmer._rho_setter if hasattr(farmer, '_rho_setter') else None
    if args.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so there must be --default-rho") 
    
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }
    scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)
    
    # Vanilla PH hub
    hub_dict = vanilla.ph_hub(*beans,
                              scenario_creator_kwargs=scenario_creator_kwargs,
                              ph_extensions=None,
                              rho_setter = rho_setter)

    # FWPH spoke
    if args.with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Special Lagranger bound spoke
    if args.with_lagranger:
        lagranger_spoke = vanilla.lagranger_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if args.with_xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if args.with_xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

        
    list_of_spoke_dict = list()
    if args.with_fwph:
        list_of_spoke_dict.append(fw_spoke)
    if args.with_lagranger:
        list_of_spoke_dict.append(lagranger_spoke)
    if args.with_xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if args.with_xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    mpisppy.cylinders.SPOKE_SLEEP_TIME = 0.1

    WheelSpinner(hub_dict, list_of_spoke_dict).spin()


if __name__ == "__main__":
    main()
