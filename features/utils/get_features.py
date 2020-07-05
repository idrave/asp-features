import clingo
import argparse
from features.model_util import write_symbols

clingo_prog = (
"#external show(X) : feature(X)."
"#show."
"#show featureId(F, I) : featureId(F, I), show(F)."
"#show feature(F) : feature(F), show(F)."
"#show bool(F) : bool(F), show(F)."
"#show num(F) : num(F), show(F)."
"#show cost(F, C) : cost(F, C), show(F)."
"#show qualValue(F, V, S) : qualValue(F, V, S), show(F)."
"#show delta(S, T, F, V) : delta(S, T, F, V), show(F)."
)

parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', required=True, help='Input feature file')
parser.add_argument('-out', '--output', required=True, help='Output feature file')
parser.add_argument('-f', '--features', required=True,nargs='+', type=int, help='Features to be included from input')
args = parser.parse_args()

ctl = clingo.Control()
ctl.add('base', [], clingo_prog)
ctl.load(args.input)
ctl.ground([('base', [])])
for feature in args.features:
    ctl.assign_external(clingo.Function('show', [feature]), True)
with ctl.solve(yield_=True) as models:
    model = next(models)
    write_symbols(model.symbols(shown=True), args.output)