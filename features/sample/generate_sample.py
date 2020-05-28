
import clingo
import sys
from features.model_util import to_model_list, ModelUtil
from features.sample.get_encoding import apply_encoding

relevant_simple = ['has','variable','initialState','constant','goalState','state','holdsState','transition']        

relevant_plasp = [('has',2),('variable',1),('initialState',1),('constant',1),('goalState',1),('state',2),('holdsState',3),('transition',2)]        
relevant_clingo = [('pred',1),('arity',2),('const',1),('state',1),('hold',2),('hold',3),('transition',2),('relevant',2),('goal',1)]


def onmodel(m: clingo.Model):
    #print(m)
    symbols = m.symbols(atoms=True)
    count_state = 0
    count_transition = 0
    for s in symbols:
        if s.name == "state": count_state += 1
        if s.name == "transition": count_transition += 1

    print("States: {}, Transitions: {}".format(count_state, count_transition))

path = "/home/ivan/Documents/ai/features/features/sample"

problem = sys.argv[1]
output_path = sys.argv[2]
h = int(sys.argv[3])

prg = clingo.Control()
prg.load(path+"/expand_states.lp")
prg.load(problem)

prg.ground( [("base", [])] )

print("Expand until: {}".format(h))
plasp_symbols = None

for i in range(h+1):
    prg.ground([("expand", [i])])
    prg.solve()
    prg.cleanup()
    prg.ground([("prune", [i])])
    with prg.solve(yield_=True) as models:
        models = to_model_list(models)
        if len(models) == 0:
            raise RuntimeError("Not satisfiable!")
        elif len(models) > 1:
            raise RuntimeWarning("More than one model found!")
        if i == h:
            model = models[0]
            plasp_symbols = ModelUtil(model.symbols(atoms=True)).get_symbols(relevant_plasp)
    prg.cleanup()

final_symbols = ModelUtil(apply_encoding(plasp_symbols)).get_symbols(relevant_clingo)
ModelUtil(final_symbols).write(output_path)