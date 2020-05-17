
import clingo
import sys

relevant = ['has','variable','initialState','constant','goalState','state','holds','transition']        

def save_model(path, model: clingo.Model):
    model_str = ""
    for symbol in model.symbols(atoms=True):
        if symbol.name in relevant:
            model_str += str(symbol) + ".\n"
    with open(path, mode='w') as f:
        f.write(model_str)

def onmodel(m: clingo.Model):
    #print(m)
    symbols = m.symbols(atoms=True)
    count_state = 0
    count_transition = 0
    for s in symbols:
        if s.name == "state": count_state += 1
        if s.name == "transition": count_transition += 1

    print("States: {}, Transitions: {}".format(count_state, count_transition))

path = "/home/ivan/Documents/ai/features/src/sample"

problem = sys.argv[1]
output_path = sys.argv[2]
h = int(sys.argv[3])

prg = clingo.Control()
prg.load(path+"/expand_states.lp")
prg.load(problem)

prg.ground( [("base", [])] )

print("Expand until: {}".format(h))
for i in range(h+1):
    prg.ground([("expand", [i])])
    prg.solve()
    prg.cleanup()
    prg.ground([("prune", [i])])
    with prg.solve(yield_=True) as models:
        for model in models:
            onmodel(model)
            if i == h: save_model(output_path, model)
    prg.cleanup()
