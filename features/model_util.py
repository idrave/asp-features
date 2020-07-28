import clingo
from features.logic import Logic
import logging
import os
from typing import List

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

class SymbolSet:
    def __init__(self, symbols: List[clingo.Symbol]):
        self.__numbers = []
        self.__strings = []
        self.__func = {}
        self.add_symbols(symbols)

    def add_symbols(self, symbols: List[clingo.Symbol]):
         for sym in symbols:
            assert(isinstance(sym, clingo.Symbol))
            if sym.type == clingo.SymbolType.Number:
                self.__numbers.append(sym)
            elif sym.type == clingo.SymbolType.String:
                self.__strings.append(sym)
            elif sym.type == clingo.SymbolType.Function:
                self.__add_func(sym)

    @property
    def numbers(self) -> List[clingo.Symbol]:
        return self.__numbers

    @property
    def strings(self) -> List[clingo.Symbol]:
        return self.__strings

    def __add_func(self, atom: clingo.Function):
        name = atom.name
        arity = len(atom.arguments)
        positive = atom.positive
        if name not in self.__func:
            self.__func[name] = {}
        if arity not in self.__func[name]:
            self.__func[name][arity] = ([], [])
        
        self.__func[name][arity][positive].append(atom.arguments)

    def has_atom(self, atom: clingo.Symbol) -> bool:
        if atom.type == clingo.SymbolType.Number:
            for a2 in self.numbers:
                if eq_symbol(atom, a2): return True
            return False
        if atom.type == clingo.SymbolType.String:
            for a2 in self.strings:
                if eq_symbol(atom, a2): return True
            return False
        if atom.type == clingo.SymbolType.Function:
            for a2 in self.get_atoms(atom.name, len(atom.arguments), positive=atom.positive):
                if eq_symbol(atom, a2): return True
            return False
        return False
            

    def get_atoms(self, name, arity, positive=True) -> List[clingo.Symbol]:
        if name not in self.__func or arity not in self.__func[name]:
            return []
        return [clingo.Function(name, args, positive=positive) for
                    args in self.__func[name][arity][positive]]

    def count_atoms(self, name, arity, positive=True) -> int:
        return len(self.get_atoms(name, arity, positive=positive))

    def get_all_atoms(self) -> List[clingo.Symbol]:
        result = []
        for n in self.__func:
            for a in self.__func[n]:
                result += self.get_atoms(n, a, positive=True)
                result += self.get_atoms(n, a, positive=False)
        return result

def eq_symbol(s1: clingo.Symbol, s2: clingo.Symbol):
    if s1.type != s2.type:
        return False
    if s1.type == clingo.SymbolType.Number:
        return s1.number == s2.number
    if s1.type == clingo.SymbolType.String:
        return s1.string == s2.string
    if s1.type == clingo.SymbolType.Function:
        if s1.name != s2.name or len(s1.arguments) != len(s2.arguments) or s1.positive != s2.positive:
            return False
        return all([eq_symbol(a1, a2) for a1, a2 in zip(s1.arguments, s2.arguments)])
    return True


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

class SymbolHash:
    def __init__(self, symbol: clingo.Symbol):
        self._symbol = symbol
    def __hash__(self):
        return hash(str(self._symbol))
    def __eq__(self, other):
        return eq_symbol(self.symbol, other.symbol)
    @property
    def symbol(self):
        return self._symbol