import clingo
from model_util import add_symbols, ModelUtil, check_multiple
from features.logic import Logic
from pathlib import Path
import pickle
from multiprocessing import Process, Pipe, connection
import re
from enum import Enum
import logging
import os
from typing import List, Tuple, Union, Optional

def splitSymbols(symbols, start_id, first_size, id_prog, get_prog, max_conc=50):
    with ClingoSolver() as ctl:
        ctl.addSymbols(symbols)
        ctl.load(Logic.pruneFile)
        ctl.ground([Logic.base, id_prog(start_id, first_size, max_conc)])
        ctl.solve()
        
        group_n = None
        atom_n = None
        group_n = ctl.getAtoms('groups', 1)[0].arguments[0].number
        atom_n = ctl.getAtoms('count', 1)[0].arguments[0].number
        print(group_n, atom_n)
        #print(ctl.getAtoms('conceptId', 2))
        ctl.ground([get_prog])
        result = []
        for i in range(group_n):
            ctl.assign_external(clingo.Function('show', [clingo.Number(i)]), False)
        for i in range(group_n):
            ctl.assign_external(clingo.Function('show', [clingo.Number(i)]), True)
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            result.append(symbols[0])
            ctl.assign_external(clingo.Function('show', [clingo.Number(i)]), False)

    return atom_n, result



class ConceptFile:
    def __init__(self, name, file_, depth):
        self.name = name
        self.file = file_
        self.depth = depth

class ClingoSolver:

    def __init__(self, ctlkwargs: dict = {}):

        self.__ctl = clingo.Control(**ctlkwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            print(type, value, traceback)
        del self.__ctl

    def load(self, clingo_files: Union[str, List[str]]) -> None:
        if not isinstance(clingo_files, list):
            self.__ctl.load(clingo_files)
            return
        for f in clingo_files:
            self.__ctl.load(f)

    def addSymbols(self, symbols: List[clingo.Symbol]) -> None:
        
        with self.__ctl.backend() as backend:
            for symbol in symbols:
                atom = backend.add_atom(symbol)
                backend.add_rule([atom], [], False)
        self.__ctl.cleanup()

    def ground(self, programs: List[Tuple]) -> None:
        self.__ctl.ground(programs)

    def solve(self, solvekwargs: dict = dict(yield_=False), symbolkwargs: dict = dict(atoms=True)) -> Optional[List[List[clingo.Symbol]]]:
        if not solvekwargs['yield_']:
            self.__ctl.solve(**solvekwargs)
            return
        symbols = []
        with self.__ctl.solve(**solvekwargs) as models:
            for model in models:
                symbols.append(model.symbols(**symbolkwargs))
        return symbols

    def assign_external(self, symbol: clingo.Symbol, val: bool) -> None:
        self.__ctl.assign_external(symbol, val)

    def reset(self, start_symbols: List[clingo.Symbol] = []) -> None:
        del self.__ctl
        self.__ctl = clingo.Control
        self.addSymbols(start_symbols)

    def countAtoms(self, name, arity)-> int:
        return len(self.getAtoms(name, arity))

    def getAtoms(self, name, arity) -> List[clingo.Symbol]:
        return [a.symbol for a in self.__ctl.symbolic_atoms.by_signature(name, arity) if a.is_fact]

    def cleanup(self) -> None:
        self.__ctl.cleanup()

    def getExternals(self) -> List[clingo.Symbol]:
        return [a.symbol for a in self.__ctl.symbolic_atoms if a.is_external]

class ClingoOps(Enum):
    LOAD = 1
    ADD = 2
    GROUND = 3
    SOLVE = 4
    ASSIGN = 5
    RESET = 6
    COUNT = 7
    GET = 8
    CLEAN = 9
    EXTERNAL = 10
    END = 11
    @staticmethod
    def get_op(op, ctl: ClingoSolver):
        if op == ClingoOps.LOAD: return ctl.load
        if op == ClingoOps.ADD: return ctl.addSymbols
        if op == ClingoOps.GROUND: return ctl.ground
        if op == ClingoOps.SOLVE: return ctl.solve
        if op == ClingoOps.ASSIGN: return ctl.assign_external
        if op == ClingoOps.RESET: return ctl.reset
        if op == ClingoOps.COUNT: return ctl.countAtoms
        if op == ClingoOps.GET: return ctl.getAtoms
        if op == ClingoOps.CLEAN: return ctl.cleanup
        if op == ClingoOps.EXTERNAL: return ctl.getExternals
        raise ValueError('{} is not a valid ClingoOps')

class ClingoProcess:
    

    def __init__(self, ctlkwargs: dict = {}):
        #print('Hello')
        self.parent_conn, self.child_conn = Pipe(duplex=True)
        self.process = Process(target=self.__start, args=(ctlkwargs,))
        self.process.start()
        self.child_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.parent_conn.send((ClingoOps.END, [], {}))
        self.parent_conn.close()

    def __start(self, kwargs):
        self.parent_conn.close()
        with ClingoSolver(kwargs) as ctl:
            while True:
                op, args, kwargs = self.__child_recv()
                #print(op)
                if op == ClingoOps.END:
                    self.child_conn.close()
                    break
                func = ClingoOps.get_op(op, ctl)
                result = func(*args, **kwargs)
                self.__child_send(op, result)
                             

    def __child_send(self, op, values):
        if op in [ClingoOps.GET, ClingoOps.EXTERNAL]:
            values = list(map(symbol_pickle, values))
        elif op == ClingoOps.SOLVE and values is not None:
            values = list(map(lambda x: list(map(symbol_pickle, x)), values))
        self.child_conn.send(values)

    def __child_recv(self):
        op, args, kwargs = self.child_conn.recv()
        if op == ClingoOps.ADD:
            args[0] = list(map(lambda x: x.to_symbol(), args[0]))
        elif op == ClingoOps.RESET:
            kwargs['start_symbols'] = list(map(lambda x: x.to_symbol(), kwargs['start_symbols']))
        elif op == ClingoOps.ASSIGN:
            args[0] = args[0].to_symbol()
        return op, args, kwargs

    def __parent_send(self, op, args, kwargs):
        if op == ClingoOps.ADD:
            args[0] = list(map(symbol_pickle, args[0]))
        elif op == ClingoOps.RESET:
            kwargs['start_symbols'] = list(map(symbol_pickle, kwargs['start_symbols']))
        elif op == ClingoOps.ASSIGN:
            args[0] = symbol_pickle(args[0])
        self.parent_conn.send((op, args, kwargs))

    def __parent_recv(self, op):
        values = self.parent_conn.recv()
        if op in [ClingoOps.GET, ClingoOps.EXTERNAL]:
            values = list(map(lambda s: s.to_symbol(), values))
        elif op == ClingoOps.SOLVE and values is not None:
            values = list(map(lambda x: list(map(lambda s: s.to_symbol(), x)), values))
        return values

    def load(self, clingo_files: Union[str, List[str]]) -> None:
        self.__parent_send(ClingoOps.LOAD, [clingo_files], {})
        return self.__parent_recv(ClingoOps.LOAD)

    def addSymbols(self, symbols: List[clingo.Symbol]) -> None:
        self.__parent_send(ClingoOps.ADD, [symbols], {})
        return self.__parent_recv(ClingoOps.ADD)

    def ground(self, programs: List[Tuple]) -> None:
        self.__parent_send(ClingoOps.GROUND, [programs], {})
        return self.__parent_recv(ClingoOps.GROUND)

    def solve(self, solvekwargs: dict = dict(yield_=False), symbolkwargs: dict = dict(atoms=True)) -> Optional[List[List[clingo.Symbol]]]:
        self.__parent_send(ClingoOps.SOLVE, [], dict(solvekwargs=solvekwargs, symbolkwargs=symbolkwargs))
        return self.__parent_recv(ClingoOps.SOLVE)

    def assign_external(self, symbol: clingo.Symbol, val: bool) -> None:
        self.__parent_send(ClingoOps.ASSIGN, [symbol_pickle(symbol), val], {})
        return self.__parent_recv(ClingoOps.ASSIGN)

    def reset(self, start_symbols: List[clingo.Symbol] = []) -> None: #TODO reset child process when calling reset
        self.__parent_send(ClingoOps.RESET, [], dict(start_symbols=start_symbols))
        return self.__parent_recv(ClingoOps.RESET)

    def countAtoms(self, name, arity) -> int:
        self.__parent_send(ClingoOps.COUNT, [name, arity], {})
        return self.__parent_recv(ClingoOps.COUNT)

    def getAtoms(self, name, arity) -> List[clingo.Symbol]:
        self.__parent_send(ClingoOps.GET, [name, arity], {})
        return self.__parent_recv(ClingoOps.GET)

    def cleanup(self) -> None:
        self.__parent_send(ClingoOps.CLEAN, [], {})
        return self.__parent_recv(ClingoOps.CLEAN)

    def getExternals(self) -> List[clingo.Symbol]:
        self.__parent_send(ClingoOps.EXTERNAL, [], {})
        return self.__parent_recv(ClingoOps.EXTERNAL)

class Solver:
    SIMPLE = 1
    PROCESS = 2
    __dict = {
        SIMPLE : ClingoSolver,
        PROCESS : ClingoProcess
    }
    __default = SIMPLE
    @staticmethod
    def set_default(type_):
        if type_ in Solver.__dict:
            Solver.__default = type_
        else:
            raise KeyError('Invalid type {} for Solver'.format(type_))
    @staticmethod
    def get_default():
        return Solver.__default
    @staticmethod
    def open(args={}, type_=None):
        type_ = Solver.__default if type_ is None else type_
        if type_ not in Solver.__dict:
            raise KeyError('Invalid type {} for Solver'.format(type_))
        return Solver.__dict[type_](args)


def symbol_pickle(symbol: clingo.Symbol):
    if symbol.type == clingo.SymbolType.Number: return NumberPickle(symbol)
    if symbol.type == clingo.SymbolType.String: return StringPickle(symbol)
    if symbol.type == clingo.SymbolType.Function: return FunctionPickle(symbol)
    raise ValueError('Type {} not supported'.format(symbol.type))

class NumberPickle:
    def __init__(self, symbol: clingo.Number):
        self.number = symbol.number
    def to_symbol(self) -> clingo.Number:
        return clingo.Number(self.number)

class StringPickle:
    def __init__(self, symbol: clingo.String):
        self.string = symbol.string
    def to_symbol(self) -> clingo.String:
        return clingo.String(self.string)

class FunctionPickle:
    def __init__(self, symbol: clingo.Function):
        self.name = symbol.name
        self.arguments = list(map(symbol_pickle, symbol.arguments))
        self.positive = symbol.positive
    def to_symbol(self) -> clingo.Function:
        return clingo.Function(self.name, list(map(lambda x: x.to_symbol(), self.arguments)), self.positive)