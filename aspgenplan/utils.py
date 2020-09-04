from aspgenplan.solver import create_solver
from aspgenplan.logic import Logic
from typing import List
import clingo
import logging
import os
import pickle
import codecs

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
        self._numbers = {}
        self._strings = {}
        self._str_list = []
        self._signature = {} #set of (name,arity,sign) of functions found
        self._sig_list = [] #list of (name,arity,sign) of functions found ordered by index
        self._functions = [] #list from signature id to dictionary of arguments that satisfy to id
        self._fun_list = [] #list from signature id to list of arguments that satisfy indexed by id
        self._num_true = []
        self._str_true = []
        self._fun_true = []
        self.add_symbols(symbols)

    def add_symbol(self, symbol: clingo.Symbol, fact=True):
        assert(isinstance(symbol, clingo.Symbol))
        if symbol.type == clingo.SymbolType.Number:
            return self.add_number(symbol, fact=fact)
        elif symbol.type == clingo.SymbolType.String:
            return self.add_string(symbol, fact=fact)
        elif symbol.type == clingo.SymbolType.Function:
            return self.add_function(symbol, fact=fact)

    def add_symbols(self, symbols: List[clingo.Symbol], fact=True):
         for sym in symbols:
            self.add_symbol(sym, fact=fact)
    
    def _get_sig_id(self, name, arity, positive):
        return self._signature.get((name, arity, positive), None)

    @staticmethod
    def signature(symbol: clingo.Symbol):
        return symbol.name, len(symbol.arguments), symbol.positive

    def _get_id(self, symbol: clingo.Symbol):
        if symbol.type == clingo.SymbolType.Number:
            if symbol.number not in self._numbers:
                return None
            else: return (0, symbol.number)
        elif symbol.type == clingo.SymbolType.String:
            result = self._strings.get(symbol.string, None)
            return None if result == None else (1, result)
        elif symbol.type == clingo.SymbolType.Function:
            sig = self._get_sig_id(*SymbolSet.signature(symbol))
            if sig == None: return None
            args = []
            for a in symbol.arguments:
                a_id = self._get_id(a)
                if a_id == None: return None
                args.append(a_id)
            result = self._functions[sig].get(tuple(args), None)
            if result == None: return None
            return (2, (sig, result))
        raise ValueError('Failed on symbol {}'.format(symbol))

    def _from_id(self, type, id: int) -> clingo.Symbol:
        #print('type {} and id {}'.format(type, id))
        if type == 0:
            return clingo.Number(id)
        if type == 1:
            return clingo.String(self._str_list[id])
        if type == 2:
            sig_id, sym_id = id
            name, arity, pos = self._sig_list[sig_id]
            args_id = self._fun_list[sig_id][sym_id]
            assert arity == len(args_id)
            args = [self._from_id(t,i) for t,i in args_id]
            return clingo.Function(name, args, pos)
        raise ValueError('Invalid type', type)

    def add_number(self, number: clingo.Symbol, fact=True):
        if number.number in self._numbers:
            return
        self._numbers[number.number] = True
        id = number.number
        if fact: self._num_true.append(id)
        return (0, id)

    def add_string(self, string: clingo.Symbol, fact = True):
        if self.has_symbol(string): return
        self._strings[string.string] = len(self._str_list)
        self._str_list.append(string.string)
        id = len(self._str_list) - 1
        if fact: self._str_true.append(id)
        return (1, id)

    def add_function(self, function: clingo.Symbol, fact=True):
        sig_id = self._get_sig_id(*SymbolSet.signature(function))
        if sig_id == None:
            sig_id = self.add_signature(SymbolSet.signature(function))
        args = []
        for a in function.arguments:
            a_id = self._get_id(a)
            #print(function, a_id)
            if a_id == None:
                a_id = self.add_symbol(a, fact=False)
            args.append(a_id)
        if tuple(args) in self._functions[sig_id]:
            return
        self._functions[sig_id][tuple(args)] = len(self._fun_list[sig_id])
        #print('adding on signature {} ({}) args {}'.format(sig_id, function, args))
        assert all([all([a != None for a in t]) for t in args])
        self._fun_list[sig_id].append(tuple(args))
        id = len(self._fun_list[sig_id]) - 1
        if fact: self._fun_true[sig_id].append(id)
        return (2, (sig_id, id))

    def add_signature(self, signature):
        if signature in self._signature: return
        self._signature[signature] = len(self._sig_list)
        self._sig_list.append(signature)
        self._functions.append({})
        self._fun_list.append([])
        self._fun_true.append([])
        return len(self._sig_list) - 1

    def has_symbol(self, symbol: clingo.Symbol):
        return self._get_id(symbol) != None

    @property
    def numbers(self) -> List[clingo.Symbol]:
        return [self._from_id(0, i) for i in self._num_true]

    @property
    def strings(self) -> List[clingo.Symbol]:
        return [self._from_id(1, i) for i in self._str_true]

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
        sig_id = self._get_sig_id(name, arity, positive)
        if sig_id == None: return []
        sym_id = self._fun_true[sig_id]
        #print('getting symbols of signature {} with id {}'.format(sig_id, sym_id))
        return [self._from_id(2, (sig_id, i)) for i in sym_id]

    def count_atoms(self, name, arity, positive=True) -> int:
        sig_id = self._get_sig_id(name, arity, positive)
        if sig_id == None: return 0
        return len(self._fun_true[sig_id])

    def get_all_atoms(self) -> List[clingo.Symbol]:
        result = self.numbers + self.strings
        for name, arity, pos in self._signature.keys():
            result += self.get_atoms(name, arity, positive=pos)
        return result

    def to_str(self):
        return symbol_to_str(self.get_all_atoms())

    @staticmethod
    def from_str(str_symbols: str):
        with create_solver() as ctl:
            ctl.add('base', [], str_symbols)
            ctl.ground([Logic.base])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return SymbolSet(sym)

    def to_json(self):
        info = {
            '_numbers' : self._numbers,
            '_strings' : self._strings,
            '_str_list' : self._str_list,
            '_signature' : self._signature, #json cannot have tuples as keys
            '_sig_list' : self._sig_list,
            '_functions' : self._functions, 
            '_fun_list' : self._fun_list,
            '_num_true' : self._num_true,
            '_str_true' : self._str_true,
            '_fun_true' : self._fun_true
        }
        return info

    def to_pickle(self):
        return codecs.encode(pickle.dumps(self), 'base64').decode()

    @staticmethod
    def from_pickle(bytesstr):
        return pickle.loads(codecs.decode(bytesstr.encode(), 'base64'))

    @staticmethod
    def load_dict(info):
        symset = SymbolSet([])
        def load(
            _numbers, _strings, _str_list, _signature, _sig_list,
            _functions, _fun_list, _num_true, _str_true, _fun_true):
            symset._numbers = _numbers
            symset._strings = _strings
            symset._str_list = _str_list
            symset._signature = _signature
            symset._sig_list = _sig_list
            symset._functions = _functions
            symset._fun_list = _fun_list
            symset._num_true = _num_true
            symset._str_true = _str_true
            symset._fun_true = _fun_true
        load(**info)        
        return symset

import json

class SymbolSetEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': [hint_tuples(i) for i in item]}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                d = {'__dict__': True, 'items': [[hint_tuples(key), hint_tuples(value)] for key, value in item.items()]}
                return d
            else:
                return item

        return super().encode(hint_tuples(obj))

def hinted_tuple_hook(obj):
    #print(obj)
    if not isinstance(obj, dict):
        return obj
    if '__tuple__' in obj:
        return tuple(obj['items'])
    elif '__dict__' in obj:
        return {hinted_tuple_hook(key) : hinted_tuple_hook(value) for key, value in obj['items']}
    else:
        return obj

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