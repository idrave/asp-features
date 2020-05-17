import clingo
import re
import sys
nulls = []

#variable(.*)

def on_model(m: clingo.Model):
    symbols = m.symbols(atoms=True)
    for s in symbols:
        if(s.name == "variable"):
            pred = s.arguments[0].arguments[0]
            if(pred.type == clingo.SymbolType.String):
                print(s)
                nulls.append(clingo.Function("arity", [pred, clingo.Number(0)]))
    m.extend(nulls)

path = sys.argv[1]

ctl = clingo.Control()

ctl.load(path)
ctl.ground([("base", [])])
ctl.solve(on_model=on_model)
print(nulls)