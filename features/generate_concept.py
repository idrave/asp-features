import clingo
from features.model_util import ModelUtil, to_model_list, filter_symbols
import sys

k = int(sys.argv[3])
output_path = sys.argv[2]
sample = sys.argv[1]

prg = clingo.Control()
path = "/home/ivan/Documents/ai/features/features/"
prg.load(sample)
prg.load(path +"concept.lp")
prg.load(path+"prune.lp")
prg.load(path+"equiv.lp")
prg.add("base", [], "#const maxcost = {}.".format(k))
prg.ground( [("base", [])] )
print("Expand until: {}".format(k))
for i in range(1, k+1):
    prg.ground( [("step", [i]),("standard_equiv", []),("reduce", [i])] )
    with prg.solve(yield_=True) as models:
        models = to_model_list(models)
        if len(models) == 0:
            raise RuntimeError("Not satisfiable!")
        elif len(models) > 1:
            raise RuntimeWarning("More than one model found!")
        #print(models[0])
        model = ModelUtil(models[0].symbols(atoms=True))
        print("Level {}. Number of concepts: {}. Candidates: {}".format(i, model.count_symbol(("conc",2)), model.count_symbol(("candidate",2))))
        if i == k:
            model.write(output_path)
    prg.cleanup()
