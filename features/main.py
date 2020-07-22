import argparse
from features.comparison import CompareConcept
import logging
from features.solver import SolverType
import features.solver as solver
from features.sample.sample import Instance, Sample
from features.grammar import Grammar
from features.feat import Features
from pathlib import Path
from features.model_util import write_symbols
import sys
from features.selection import solve_T_G, solve_T_G_subprocess

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pddl', nargs='+', help='Input instances in PDDL files')
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--proc',help="Runs clingo solver in separate process",
        action="store_const", dest="solver", const=SolverType.PROCESS, default=SolverType.SIMPLE)
    parser.add_argument('--out',required=True, help='Output file path')
    parser.add_argument('-load', default=None, help='Path for loading results')

    #Sample generation arguments
    sample_group = parser.add_argument_group('Sample')
    sample_group.add_argument('--symbol', action='store_true', help='Represent states in plain symbols instead of numericaly')
    sample_group.add_argument('--depth', type=int, default=None, help='Minimum expansion depth required')
    sample_group.add_argument('-s', dest='states', type=int, default=None, help='Minimum number of states required')
    sample_group.add_argument('-t', dest='transitions', type=int, default=None, help='Minimum number of transitions required')
    sample_group.add_argument('--complete', action='store_true', help='Expand all state space (could be too big!)')
    sample_group.add_argument('--goal', action='store_true', help='Ensure there is at least one goal per instance')
    
    #Concept generation arguments
    conc_group = parser.add_argument_group('Concepts')
    conc_group.add_argument('--exp',default=400, type=int, help='Max number of expressions in prunning set')
    conc_group.add_argument('--conc',default=250, type=int, help='Max number of concepts in file')
    group_compare = conc_group.add_mutually_exclusive_group(required = False)
    group_compare.add_argument('--fast', action='store_const', dest='compare', help='Prunning with cardinality',
        const=CompareConcept.FAST, default=CompareConcept.STANDARD)
    group_compare.add_argument('--std', action='store_const', dest='compare', help='Standard sound prunning',
        const=CompareConcept.STANDARD)
    group_compare.add_argument('--mix', action='store_const', dest='compare', help='Cardinality + standard prunning',
        const=CompareConcept.MIXED)

    #Feature generation arguments
    feat_group = parser.add_argument_group('Features')
    feat_group.add_argument('--cost', default=None, type=int, help='Feature cost required')
    feat_group.add_argument('-f', '--features', default=None, type=int, help='Minimum features required')
    feat_group.add_argument('--dist', action='store_true', help='Option to generate distance features')
    feat_group.add_argument('--batch',action='store',default=1, type=int, help='Concept files used simultaneaously in feature generation.')
    feat_group.add_argument('--atom',default=250, type=int, help='Max new features prunned together')
    feat_group.add_argument('--comp',default=250, type=int, help='Max number of known features used for prunning simultaneously')
    
    #Solver arguments
    parser.add_argument('--sat', action='store_true', help='Only apply sat')

    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    if not (args.features != None or args.cost != None):
        RuntimeError('At least one of arguments --cost and --features is required')

    logging.basicConfig(level=args.loglevel)
    solver.set_default(args.solver)
    print(args.pddl, args.out)
    out_path = Path(args.out)
    if not out_path.is_dir():
        try:
            out_path.mkdir()
        except (FileNotFoundError, FileExistsError) as e:
            print(repr(e))
            sys.exit()
    with open(str(out_path/'info.txt'), 'a') as fp:
        fp.write(str(sys.argv))
    if args.load != None:
        sample = Sample(load_path=str(Path(args.load)/'sample'))
    else:
        sample = Sample(instances=[Instance(pddl, numbered=(not args.symbol)) for pddl in args.pddl])
    grammar = Grammar(sample, str(out_path/'concepts'), comp_type=args.compare)
    if args.load:
        grammar.load_progress()
    features = Features(sample, grammar, str(out_path), distance=args.dist, load = args.load != None)
    print(args.depth)
    if not args.sat:
        sample.expand_states(
            depth=args.depth, states=args.states, transitions=args.transitions,
            goal_req=args.goal, complete=args.complete
        )
        sample.store(str(out_path/'sample'))
    if not args.sat:
        while (args.features != None and features.feature_count() < args.features) or \
                (args.cost != None and features.cost < args.cost):
            features.generate(
                batch=args.batch, max_f = args.features, max_pre=args.atom, feat_prune=args.comp,
                max_exp=args.exp, max_conc=args.conc
            )
    
    #solution, (time_g, mem_g), (time_s, mem_s) = solve_T_G(sample, features)
    sample.print_info()
    print('Total concepts: {}\nTotal features: {}'.format(grammar.total_concepts, features.total_features))
    
    #print('Grounding took {}s, min memory {} MB, max memory {} MB'.format(round(time_g, 3), round(min(mem_g)/1e6, 3), round(max(mem_g)/1e6, 3)))
    #logging.debug('Profiling samples: {}'.format(len(mem_s)))
    solution, t, mem = solve_T_G_subprocess(sample, features, args.out)
    logging.debug('Profiling samples: {}'.format(len(mem)))
    #print('Solving took {}s, min memory {} MB, max memory {} MB'.format(round(time_s, 3), round(min(mem_s)/1e6, 3), round(max(mem_s)/1e6, 3)))
    print('Solving took {}s, start memory {} MB, max memory {} MB'.format(round(t, 3), round(min(mem)/1e6, 3), round(max(mem)/1e6, 3)))
    print('Solutions found: {}'.format(len(solution)))
    print('Optimal solution: {}. Cost: {}.'.format(*solution[-1] if len(solution) > 0 else None))
    #print(round(t, 3), round(mem[0]/1e6, 3), round(max(mem)/1e6, 3))
    
    