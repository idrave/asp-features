from features.solver import ClingoSolver
from features.logic import Logic
from features.model_util import write_symbols
import argparse

def numbered_sample(sample_path, start=0):
    with ClingoSolver() as ctl:
        ctl.load([sample_path, Logic.pruneFile])
        ctl.ground([Logic.base, Logic.number_state(start)])
        symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
    return symbols[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input sample file')
    parser.add_argument('output', help='Output file for storing numbered sample')
    args = parser.parse_args()
    write_symbols(numbered_sample(args.input), args.output)