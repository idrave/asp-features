import clingo
from multiprocessing import Process, Pipe
from enum import Enum, auto
from typing import List, Union, Tuple, Optional
from abc import ABC, abstractmethod
import psutil
import time

class Solver(ABC):
    def __init__(self, ctlkwargs: dict = {}):
        self.args = ctlkwargs

    @abstractmethod
    def open(self):
        pass
    
    @abstractmethod
    def close(self):
        pass

    def reset(self, start_symbols: List[clingo.Symbol] = [], **args) -> None:
        self.close()
        self.args = args
        self.open()
        self.addSymbols(start_symbols)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            print(type, value, traceback)
        self.close()

    @abstractmethod
    def load(self, clingo_files: Union[str, List[str]]) -> None:
        pass

    @abstractmethod
    def add(self, program, prog_args, rules) -> None:
        pass

    @abstractmethod
    def addSymbols(self, symbols: List[clingo.Symbol]) -> None:
        pass

    @abstractmethod
    def ground(self, programs: List[Tuple]) -> None:
        pass

    @abstractmethod
    def solve(self, solvekwargs: dict = dict(yield_=False), symbolkwargs: dict = dict(atoms=True)) -> Optional[List[List[clingo.Symbol]]]:
        pass

    @abstractmethod
    def assign_external(self, symbol: clingo.Symbol, val: bool) -> None:
        pass

    @abstractmethod
    def countAtoms(self, name, arity)-> int:
        pass

    @abstractmethod
    def getAtoms(self, name, arity) -> List[clingo.Symbol]:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    @abstractmethod
    def getExternals(self) -> List[clingo.Symbol]:
        pass

class ClingoSolver(Solver):
    def __init__(self, ctlkwargs: dict = {}):
        self.args = ctlkwargs

    def open(self):
        self.__ctl = clingo.Control(**self.args)

    def close(self):
        del self.__ctl

    def reset(self, start_symbols: List[clingo.Symbol] = [], **args) -> None:
        self.close()
        self.args = args
        self.open()
        self.addSymbols(start_symbols)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            print(type, value, traceback)
        self.close()

    def load(self, clingo_files: Union[str, List[str]]) -> None:
        if not isinstance(clingo_files, list):
            self.__ctl.load(clingo_files)
            return
        for f in clingo_files:
            self.__ctl.load(f)

    def add(self, program, prog_args, rules):
        self.__ctl.add(program, prog_args, rules)

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

    def countAtoms(self, name, arity)-> int:
        return len(self.getAtoms(name, arity))

    def getAtoms(self, name, arity) -> List[clingo.Symbol]:
        return [a.symbol for a in self.__ctl.symbolic_atoms.by_signature(name, arity) if a.is_fact]

    def cleanup(self) -> None:
        self.__ctl.cleanup()

    def getExternals(self) -> List[clingo.Symbol]:
        return [a.symbol for a in self.__ctl.symbolic_atoms if a.is_external]

class ClingoOps(Enum):
    LOAD = auto()
    ADD = auto()
    ADDSYM = auto()
    GROUND = auto()
    SOLVE = auto()
    ASSIGN = auto()
    RESET = auto()
    COUNT = auto()
    GET = auto()
    CLEAN = auto()
    EXTERNAL = auto()
    END = auto()
    @staticmethod
    def get_op(op, ctl: ClingoSolver):
        if op == ClingoOps.LOAD: return ctl.load
        if op == ClingoOps.ADD: return ctl.add
        if op == ClingoOps.ADDSYM: return ctl.addSymbols
        if op == ClingoOps.GROUND: return ctl.ground
        if op == ClingoOps.SOLVE: return ctl.solve
        if op == ClingoOps.ASSIGN: return ctl.assign_external
        if op == ClingoOps.RESET: return ctl.reset
        if op == ClingoOps.COUNT: return ctl.countAtoms
        if op == ClingoOps.GET: return ctl.getAtoms
        if op == ClingoOps.CLEAN: return ctl.cleanup
        if op == ClingoOps.EXTERNAL: return ctl.getExternals
        raise ValueError('{} is not a valid ClingoOps'.format(op))

class ClingoChild(Process):
    def __init__(self, args, pipe):
        self._args = args
        self._pipe = pipe

    def run(self):
        with ClingoSolver(self._args) as ctl:
            while True:
                op, args, kwargs = self._child_recv()
                if op == ClingoOps.END:
                    self._pipe.close()
                    break
                func = ClingoOps.get_op(op, ctl)
                result = func(*args, **kwargs)
                self._child_send(op, result)
                             
    def _child_send(self, op, values):
        if op in [ClingoOps.GET, ClingoOps.EXTERNAL]:
            values = list(map(symbol_pickle, values))
        elif op == ClingoOps.SOLVE and values is not None:
            values = list(map(lambda x: list(map(symbol_pickle, x)), values))
        self._pipe.send(values)

    def _child_recv(self):
        op, args, kwargs = self._pipe.recv()
        if op == ClingoOps.ADDSYM:
            args[0] = list(map(lambda x: x.to_symbol(), args[0]))
        elif op == ClingoOps.RESET:
            kwargs['start_symbols'] = list(map(lambda x: x.to_symbol(), kwargs['start_symbols']))
        elif op == ClingoOps.ASSIGN:
            args[0] = args[0].to_symbol()
        return op, args, kwargs

class ClingoProcess:
    
    def __init__(self, ctlkwargs: dict = {}):
        self.args = ctlkwargs
        
    def open(self):
        self._pipe, child_conn = Pipe(duplex=True)
        self.process = ClingoChild(self.args, child_conn)
        self.process.start()

    def close(self):
        self._pipe.send((ClingoOps.END, [], {}))
        self._pipe.close()
        self.process.join()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            print(type, value, traceback)
        self.close()        

    def _parent_send(self, op, args, kwargs):
        if op == ClingoOps.ADDSYM:
            args[0] = list(map(symbol_pickle, args[0]))
        elif op == ClingoOps.ASSIGN:
            args[0] = symbol_pickle(args[0])
        self._pipe.send((op, args, kwargs))

    def _parent_recv(self, op):
        values = self._pipe.recv()
        if op in [ClingoOps.GET, ClingoOps.EXTERNAL]:
            values = list(map(lambda s: s.to_symbol(), values))
        elif op == ClingoOps.SOLVE and values is not None:
            values = list(map(lambda x: list(map(lambda s: s.to_symbol(), x)), values))
        return values

    def load(self, clingo_files: Union[str, List[str]]) -> None:
        self._parent_send(ClingoOps.LOAD, [clingo_files], {})
        return self._parent_recv(ClingoOps.LOAD)

    def add(self, program, prog_args, rules):
        self._parent_send(ClingoOps.ADD, [program, prog_args, rules], {})
        return self._parent_recv(ClingoOps.ADD)

    def addSymbols(self, symbols: List[clingo.Symbol]) -> None:
        self._parent_send(ClingoOps.ADDSYM, [symbols], {})
        return self._parent_recv(ClingoOps.ADDSYM)

    def ground(self, programs: List[Tuple]) -> None:
        self._parent_send(ClingoOps.GROUND, [programs], {})
        return self._parent_recv(ClingoOps.GROUND)

    def solve(self, solvekwargs: dict = dict(yield_=False), symbolkwargs: dict = dict(atoms=True)) -> Optional[List[List[clingo.Symbol]]]:
        self._parent_send(ClingoOps.SOLVE, [], dict(solvekwargs=solvekwargs, symbolkwargs=symbolkwargs))
        return self._parent_recv(ClingoOps.SOLVE)

    def assign_external(self, symbol: clingo.Symbol, val: bool) -> None:
        self._parent_send(ClingoOps.ASSIGN, [symbol, val], {})
        return self._parent_recv(ClingoOps.ASSIGN)

    def reset(self, start_symbols: List[clingo.Symbol] = [], **kwargs) -> None: #TODO reset child process when calling reset
        self.close()
        self.args = kwargs
        self.open()
        self.addSymbols(start_symbols)

    def countAtoms(self, name, arity) -> int:
        self._parent_send(ClingoOps.COUNT, [name, arity], {})
        return self._parent_recv(ClingoOps.COUNT)

    def getAtoms(self, name, arity) -> List[clingo.Symbol]:
        self._parent_send(ClingoOps.GET, [name, arity], {})
        return self._parent_recv(ClingoOps.GET)

    def cleanup(self) -> None:
        self._parent_send(ClingoOps.CLEAN, [], {})
        return self._parent_recv(ClingoOps.CLEAN)

    def getExternals(self) -> List[clingo.Symbol]:
        self._parent_send(ClingoOps.EXTERNAL, [], {})
        return self._parent_recv(ClingoOps.EXTERNAL)

class ClingoProfiling(ClingoProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mem = None
        self._time = None
    
    def ground(self, programs: List[Tuple]) -> None:
        proc = psutil.Process(pid=self.process.pid)
        self._mem = []
        start = time.time()
        SLEEP_TIME=.001
        self._parent_send(ClingoOps.GROUND, [programs], {})
        while not self._pipe.poll():
            self._mem.append(proc.memory_info().rss)
            time.sleep(SLEEP_TIME)
        self._time = time.time() - start
        return self._parent_recv(ClingoOps.GROUND)

    def solve(self, solvekwargs: dict = dict(yield_=False), symbolkwargs: dict = dict(atoms=True)) -> Optional[List[List[clingo.Symbol]]]:
        proc = psutil.Process(pid=self.process.pid)
        self._mem = []
        start = time.time()
        SLEEP_TIME=.01
        self._parent_send(ClingoOps.SOLVE, [], dict(solvekwargs=solvekwargs, symbolkwargs=symbolkwargs))
        while not self._pipe.poll():
            self._mem.append(proc.memory_info().rss)
            time.sleep(SLEEP_TIME)
        self._time = time.time() - start
        return self._parent_recv(ClingoOps.SOLVE)

    def get_profiling(self):
        return self._time, self._mem

class SolverType(Enum):
    SIMPLE = 1
    PROCESS = 2
    PROFILE = 3

    @staticmethod
    def get_solver_class(type_):
        if not isinstance(type_, SolverType):
            raise ValueError('Invalid type {} for Solver'.format(type_))
        d = {
            SolverType.SIMPLE : ClingoSolver,
            SolverType.PROCESS : ClingoProcess,
            SolverType.PROFILE : ClingoProfiling
        }
        return d[type_]

DEFAULT_SOLVER_TYPE = SolverType.SIMPLE

def set_default(type_):
    if isinstance(type_, SolverType):
        global DEFAULT_SOLVER_TYPE
        DEFAULT_SOLVER_TYPE = type_
    else:
        raise ValueError('Invalid type {} for Solver'.format(type_))

def get_default():
    return DEFAULT_SOLVER_TYPE

def create_solver(args={}, type_=None) -> Solver:
    type_ = DEFAULT_SOLVER_TYPE if type_ is None else type_
    if not isinstance(type_, SolverType):
        raise ValueError('Invalid type {} for Solver'.format(type_))
    return SolverType.get_solver_class(type_)(ctlkwargs=args)

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