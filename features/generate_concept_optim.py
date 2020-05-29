import clingo
from features.model_util import ModelUtil, to_model_list, filter_symbols
import sys
import time
k = int(sys.argv[3])
output_path = sys.argv[2]
sample = sys.argv[1]
#prunning = sys.argv[4]

#pruneTypes = {"fast" : "fast_equiv"}

path = "/home/ivan/Documents/ai/features/features/"

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

    prg.ground( [("base", [])] )
    prg.ground( [("step", [i]),("optimal_equiv", [i]),("reduce", [i])] )

    with prg.solve(yield_=True) as models:
        models = to_model_list(models)
        if len(models) == 0:
            raise RuntimeError("Not satisfiable!")
        elif len(models) > 1:
            raise RuntimeWarning("More than one model found!")
        symbols = models[0].symbols(atoms=True)
        print("Level {}. Number of concepts: {}. Candidates: {}".format(i, ModelUtil(symbols).count_symbol(("conc",2)), ModelUtil(symbols).count_symbol(("candidate",2))))
    
    prg.cleanup()
    symbols = filter_symbols(prg, program="relevant", prog_args=[i])[0]
    #print(str(ModelUtil(symbols)))
    del(prg)
    if i == k:
        ModelUtil(symbols).write(output_path)

print("Took {}s".format(time.time() - start))

def optimal_prune(ctl: clingo.Control):
    