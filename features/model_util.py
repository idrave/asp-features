import clingo
from features.logic import Logic
import logging
import os

def to_model_list(solve_handle):
    return [model for model in solve_handle]

def symbol_to_str(symbols):
    model_str = ""
    for symbol in symbols:
        model_str += str(symbol)+'.\n'
    return model_str

def add_symbols(ctl: clingo.Control, symbols):
    with ctl.backend() as backend:
        for symbol in symbols:
            atom = backend.add_atom(symbol)
            backend.add_rule([atom], [], False)
    ctl.cleanup()

def load_symbols(file, program="base", prog_args=[]):
    ctl = clingo.Control()
    ctl.load(file)
    ctl.ground([(program, prog_args)])
    with ctl.solve(yield_ = True) as models:
        models = to_model_list(models)
        return models[0].symbols(atoms = True)

def filter_symbols(ctl, single=True):
    results = []
    with ctl.solve(yield_ = True) as models:
        models = list(models)
        if single:
            check_multiple(models)
        for model in models:
            keep = filter(lambda symbol: symbol.name == "keep__", model.symbols(atoms=True))
            results.append([symbol.arguments[0] for symbol in keep])
    if single: return results[0]
    return results

def get_symbols(symbols, name, arity):
    return list(filter(lambda s: s.match(name, arity), symbols))

def check_multiple(models):
    if len(models) == 0:
        raise RuntimeError("Not satisfiable!")
    elif len(models) > 1:
        raise RuntimeError("More than one model found!")

def write_symbols(symbols, filename, type_='w'):
    with open(filename, type_) as file:
        file.write(symbol_to_str(symbols))

def count_symbols(symbols, name, arity):
    return len(list(filter(lambda s: s.match(name, arity), symbols)))


class ModelUtil:
    def __init__(self, symbols : list):
        self.symbols = symbols

    def __str__(self):
        return symbol_to_str(self.symbols)

    def get_symbols(self, filter=None):
        if filter is None: return self.symbols
        model_str = []
        for symbol in self.symbols:
            if (symbol.name, len(symbol.arguments)) in filter:
                model_str.append(symbol)
        return model_str

    def write(self, filename, type_='w'):
        with open(filename, type_) as file:
            file.write(str(self))

    def count_symbol(self, symbol_compare):
        count = 0
        for symbol in self.symbols:
            if (symbol.name,len(symbol.arguments)) == symbol_compare:
                count += 1
        return count


