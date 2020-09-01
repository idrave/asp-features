import argparse
import multiprocessing
from features.comparison import CompareConcept
import logging
from features.solver import SolverType
import features.solver as solver
from features.sample.sample import Instance, Sample, SampleView
from features.sample.problem import Problem
from features.grammar import Grammar
from features.feat import Features
from pathlib import Path
from features.model_util import write_symbols, symbol_to_str
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
    sample_group.add_argument('--loadsample', action='store_true', help='Load only sample')

    #Concept generation arguments
    conc_group = parser.add_argument_group('Concepts')
    conc_group.add_argument('--conc',default=100, type=int, help='Concept batch size')

    #Feature generation arguments
    feat_group = parser.add_argument_group('Features')
    feat_group.add_argument('--cost', default=None, type=int, help='Feature cost required')
    feat_group.add_argument('-f', '--features', default=None, type=int, help='Minimum features required')
    feat_group.add_argument('--dist', action='store_true', help='Option to generate distance features')
    feat_group.add_argument('--batch',action='store',default=1, type=int, help='Concept files used simultaneaously in feature generation.')
    feat_group.add_argument('--stop', action='store_true', help='Stop feature generation if desired number is reached')

    #Solver arguments
    parser.add_argument('--sat', action='store_true', help='Only apply sat')
    parser.add_argument('-threads', type=int, default=1, help='Number of threads to use in solver')

    return parser.parse_args()

import time

if __name__ == "__main__":
    args = get_args()
    if not (args.features != None or args.cost != None):
        RuntimeError('At least one of arguments --cost and --features is required')

    multiprocessing.set_start_method('spawn')
    solver.set_default(args.solver)
    
    print(args.pddl, args.out)
    out_path = Path(args.out)
    if not out_path.is_dir():
        try:
            out_path.mkdir()
        except (FileNotFoundError, FileExistsError) as e:
            print(repr(e))
            sys.exit()
    logging.basicConfig(level=args.loglevel,
                        handlers=[logging.FileHandler(str(out_path/'log.txt')),
                                    logging.StreamHandler()])
    with open(str(out_path/'info.txt'), 'a') as fp:
        fp.write(str(sys.argv))
    if args.load != None:
        st_sample = time.time()
        sample = Sample.load(str(Path(args.load)/'sample'))
    else:
        st_sample = time.time()
        sample = Sample(instances=[Instance(Problem(pddl), numbered=(not args.symbol)) for pddl in args.pddl])
        
    print(args.depth)
    if not args.sat:
        sample.expand_states(
            depth=args.depth, states=args.states, transitions=args.transitions,
            goal_req=args.goal, complete=args.complete
        )
    #sample_v = sample.get_view(
    #    depth=args.depth, states=args.states, transitions=args.transitions,
    #    goal_req=args.goal, complete=args.complete, optimal=True
    #)

    sample_v = SampleView(sample, min_t=args.transitions, min_depth=args.depth, optimal = True)
    logging.debug('SampleView done')
    print('Time expanding sample', time.time()-st_sample)
    if not args.sat: sample.store(str(out_path/'sample'))
    del sample
    if args.load == None or args.loadsample:
        grammar = Grammar(sample_v)
    else:
        grammar = Grammar.load(sample_v, str(Path(args.load)/'grammar'))
    if args.load == None or args.loadsample:
        features = Features(sample_v, grammar, distance=args.dist)
    else:
        features = Features.load(sample_v, grammar, str(Path(args.load)/'features'))

    print(type(features))
    if not args.sat:
        while (args.features != None and features.feature_count() < args.features) or \
                (args.cost != None and features.cost < args.cost):
            features.generate(
                max_f=(args.features if args.stop else None),
                batch=args.batch,
                batch_g=args.conc,
            )
    grammar.store(str(out_path/'grammar'))
    features.store(str(out_path/'features'))
    with open(str(out_path/'features.lp'), 'w') as fp:
        for f in features.get_features(num=args.features):
            fp.write(symbol_to_str(f.symbols.get_all_atoms()))
    #solution, (time_g, mem_g), (time_s, mem_s) = solve_T_G(sample, features)
    sample_v.print_info()
    print('Total concepts: {}\nTotal features: {}'.format(grammar.total_concepts, features.total_features))
    
    #print('Grounding took {}s, min memory {} MB, max memory {} MB'.format(round(time_g, 3), round(min(mem_g)/1e6, 3), round(max(mem_g)/1e6, 3)))
    #logging.debug('Profiling samples: {}'.format(len(mem_s)))
    solution, t, mem = solve_T_G_subprocess(sample_v, args.out, threads=args.threads)
    logging.debug('Profiling samples: {}'.format(len(mem)))
    logging.debug('Relevant: {}'.format(sample_v.get_relevant()))
    #print('Solving took {}s, min memory {} MB, max memory {} MB'.format(round(time_s, 3), round(min(mem_s)/1e6, 3), round(max(mem_s)/1e6, 3)))
    print('Solving took {}s, start memory {} MB, max memory {} MB'.format(round(t, 3), round(min(mem)/1e6, 3), round(max(mem)/1e6, 3)))
    print('Solutions found: {}'.format(len(solution)))
    print('Optimal solution: {}. Cost: {}.'.format(*solution[-1] if len(solution) > 0 else (None, None)))
    #print(round(t, 3), round(mem[0]/1e6, 3), round(max(mem)/1e6, 3))
    print(sample_v.get_relevant())
    