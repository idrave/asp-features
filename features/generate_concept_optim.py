import clingo
from features.model_util import ModelUtil, to_model_list, filter_symbols
import sys
import time

def standard_prune(ctl: clingo.Control, depth, prune_round, cleanup=False, yield_=False):
    symbols = []
    ctl.ground([("standard_differ", [depth, prune_round])])
    if yield_:
        with ctl.solve(yield_=yield_) as models:
            for model in models:
                symbols.append(model.symbols(atoms=True))
    if cleanup: ctl.cleanup()
    return symbols

def optimal_prune(ctl: clingo.Control, depth, prune_round, cleanup=False, yield_=False):
    symbols = []
    ctl.ground([("optimal_differ_start", [depth, prune_round])])
    ctl.solve()
    if cleanup: ctl.cleanup()
    ctl.ground([("optimal_differ_end", [depth, prune_round])])
    if yield_:
        with ctl.solve(yield_=yield_) as models:
            for model in models:
                symbols.append(model.symbols(atoms=True))
    if cleanup: ctl.cleanup()
    return symbols

k = int(sys.argv[3])
output_path = sys.argv[2]
sample = sys.argv[1]
#prunning = sys.argv[4]

optim_prune = True
prune_rounds = ['prune_exp', 'prune_new', 'prune_pool', 'prune_candidate']
path = "/home/ivan/Documents/ai/features/features/"
#show = [('compare', 4), ('differ', 2), ('current', 1)]
show = []
print("Expand until: {}".format(k))
start = time.time()
symbols = []
for i in range(1, k+1):
    prg = clingo.Control()
    prg.load(sample)
    prg.load(path +"concept.lp")
    prg.load(path+"prune.lp")
    prg.load(path+"equiv.lp")
    prg.add("base", [], str(ModelUtil(symbols)))
    prg.add("base", [], "#const maxcost = {}.".format(k))

    prg.ground( [("base", []), ('step', [i])] )
    prg.solve()
    prg.cleanup()

    for p_round in prune_rounds:
        prg.ground([(p_round, [i])])
        prg.solve()
        yield_ = p_round == prune_rounds[-1]
        if optim_prune:
            symbols = optimal_prune(prg, i, p_round, yield_=yield_)
        else:
            symbols = standard_prune(prg, i, p_round, yield_=yield_)
            
    if len(symbols) == 0:
        raise RuntimeError("Not satisfiable!")
    elif len(symbols) > 1:
        raise RuntimeWarning("More than one model found!")
    symbols = symbols[0]
    model = ModelUtil(symbols)
    exp = model.count_symbol(('exp', 2))
    non_empty = model.count_symbol(('non_empty', 2))
    new_ = model.count_symbol(('new', 2))
    pool = model.count_symbol(('pool', 2))
    candidate = model.count_symbol(('candidate', 2))
    concepts = model.count_symbol(('conc', 2))
    print("Level {}. Exp: {}. New {}. Pool {}. Candidates {}. Concepts: {}".format(i,exp,new_,pool,candidate, concepts))
    if len(show): print(model.get_symbols(filter = show))
    prg.cleanup()
    symbols = filter_symbols(prg, program="relevant", prog_args=[i])[0]
    #print(str(ModelUtil(symbols)))
    del(prg)
    if i == k:
        ModelUtil(symbols).write(output_path)

print("Took {}s".format(time.time() - start))

