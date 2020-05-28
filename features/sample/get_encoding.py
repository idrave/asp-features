import clingo
from features.model_util import ModelUtil
import sys

def extract_null(m: clingo.Model):
    nulls = []
    symbols = m.symbols(atoms=True)
    for s in symbols:
        if(s.name == "variable"):
            pred = s.arguments[0].arguments[0]
            if(pred.type == clingo.SymbolType.String):
                print(s)
                nulls.append(pred)
    return nulls

def apply_encoding(atoms, display=False):
    ctl = clingo.Control()
    ctl.add("base", [], str(ModelUtil(atoms)))
    ctl.load("/home/ivan/Documents/ai/features/features/sample/encoding.lp")
    ctl.ground([("base", [])])
    nulls = []
    with ctl.solve(yield_=True) as models:
        for model in models:
            nulls = extract_null(model)
    for n in nulls:
        ctl.ground([("null_predicate", [n])])

    ctl.ground([("predicates", [])])
    ctl.solve()

    ctl.ground([("encode", [])])

    if display:
        ctl.ground([("print", [])])
    
    encoded_model = None
    with ctl.solve(yield_ = True) as models:
        for model in models:
            encoded_model = model.symbols(atoms=True)
    return encoded_model

