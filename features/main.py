import argparse
from features.comparison import CompareConcept
import logging
from features.solver import SolverType
import features.solver as solver
from features.sample.sample import Sample
from features.grammar import Grammar
from features.feat import Features
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pddl', required=True, nargs='+', help='Input instances in PDDL files')
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--proc',help="Runs clingo solver in separate process",
        action="store_const", dest="solver", const=SolverType.PROCESS, default=SolverType.SIMPLE)
    parser.add_argument('--out',required=True, help='Output file path')

    #Sample generation arguments
    sample_group = parser.add_argument_group('Sample')
    sample_group.add_argument('--symbol', action='store_true', help='Represent states in plain symbols instead of numericaly')
    sample_group.add_argument('--depth', default=6, type=int, help='Minimum expansion depth required')
    sample_group.add_argument('-s', dest='states', type=int, help='Minimum number of states required')
    sample_group.add_argument('-t', dest='transitions', type=int, help='Minimum number of transitions required')
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
    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel)
    solver.set_default(args.solver)
    print(args.pddl, args.out)
    out_path = Path(args.out)
    sample = Sample(args.pddl)
    grammar = Grammar(sample, str(out_path/'concepts'))
    features = Features(sample, grammar, str(out_path/'features.lp'))
    