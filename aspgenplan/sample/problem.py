from aspgenplan.utils import SymbolSet, write_symbols, SymbolHash
from aspgenplan.solver import Solver, create_solver
from aspgenplan.logic import Logic
from typing import List, Optional
from pathlib import Path
import aspgenplan.logic
import subprocess
import clingo

def run_plasp(in_file):
    return subprocess.run([aspgenplan.logic.PLASP_PATH, 'translate', in_file], stdout=subprocess.PIPE).stdout.decode('utf-8')

def initial_state():
    return ('initial_state', [])

def successor():
    return ('successor', [])

def evaluate():
    return ('evaluate', [])

def show_expanded():
    return ('show_expanded', [])

def to_state(id):
    return ('to_state', [id])

def encode():
    return ('encode', [])

def hold():
    return ('hold', [])

import time
time_eq = 0

class Node:
    def __init__(self, symbols: SymbolSet, depth: int, parent, id=None, encoding=None):
        self._symbols = symbols
        self._id = id
        self._children = []
        self._transitions = []
        self._depth = depth
        if parent != None:
            if isinstance(parent, Node):
                self._parent = parent.id
            else:
                self._parent = parent
        else:
            self._parent = None
        self._goal = bool(symbols.count_atoms('goal', 1))
        self._hash = None
        self._sym_hash = {}
        self._calc_hash()
        self._encoding = encoding

    def store(self):
        info = {
            'symbols': self._symbols.to_pickle(),
            'id': self.id,
            'children': self.children,
            'depth': self.depth,
            'parent': self.parent,
            'encoding': self._encoding.to_pickle()
        }
        return info

    @staticmethod
    def load_state(info):
        def load(symbols, id, children, depth, parent, encoding):
            node = Node(SymbolSet.from_pickle(symbols), depth, parent, id=id, encoding=SymbolSet.from_pickle(encoding))
            node._children = children
            for child in node.children:
                node._transitions.append(clingo.Function('transition', [node.id, child]))
            return node
        return load(**info)

    @property
    def id(self) -> Optional[int]:
        return self._id

    @property
    def name(self) -> clingo.Symbol:
        return clingo.Number(self.id)

    @property
    def symbols(self) -> SymbolSet:
        return self._symbols

    @property
    def transitions(self) -> List[clingo.Symbol]:
        return self._transitions    

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def parent(self) -> Optional[int]:
        return self._parent

    @property
    def is_goal(self) -> bool:
        return self._goal

    @property
    def state(self):
        return self._symbols.get_atoms('state', 2) + self._symbols.get_atoms('stateId', 2) \
            + self._symbols.get_atoms('state', 1)

    @property
    def goal(self):
        return self._symbols.get_atoms('goal', 1)

    def add_child(self, child):
        self._children.append(child.id)
        self._transitions.append(clingo.Function('transition', [self.id, child.id]))

    @property
    def children(self):
        return self._children

    def _calc_hash(self):
        if self._hash != None: return
        self._hash = 0
        for var, val in self.description:
            self._hash += hash(str(var)+str(val))
            self._sym_hash[(SymbolHash(var), SymbolHash(val))] = True

    @property
    def description(self):
        return [s.arguments[0:2] for s in self._symbols.get_atoms('holds', 3)]

    def holds(self, variable:clingo.Symbol, value:clingo.Symbol):
        return self._sym_hash.get((SymbolHash(variable), SymbolHash(value)), False)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        global time_eq
        start = time.time()
        if not isinstance(other, Node):
            NotImplemented
        if hash(self) != hash(other):
            return False
        holds = self.description
        if len(holds) != len(other.description): return False
        for var, val in holds:
            if not other.holds(var, val):
                time_eq += time.time() - start
                return False
        time_eq += time.time() - start
        return True

    def make_state(self, id):
        assert(self.id == None)
        self._id = id
        with create_solver() as ctl:
            ctl.load(Logic.problem)
            ctl.load(Logic.sample_encoding)
            ctl.addSymbols(self.symbols.get_all_atoms())
            ctl.ground([Logic.base, to_state(id)])
            #print(ctl.getAtoms('expanded', 2))
            models = ctl.solve(solvekwargs=dict(yield_=True),symbolkwargs=dict(shown=True))
            assert(len(models) == 1)
            self._symbols = SymbolSet(models[0])
        #print(self.state)
            
    def encode(self, prob):
        assert(self._encoding == None)
        with create_solver() as ctl:
            ctl.load(Logic.problem)
            ctl.load(Logic.sample_encoding)
            ctl.addSymbols(self.symbols.get_all_atoms() + prob.predicates + prob.const)
            ctl.ground([Logic.base, hold(), encode()])
            models = ctl.solve(solvekwargs=dict(yield_=True),symbolkwargs=dict(shown=True))
            #print(ctl.getAtoms('hold', 3), len(models))
            assert(len(models) == 1)
            self._encoding = SymbolSet(models[0])

    @property
    def encoding(self) -> Optional[List[clingo.Symbol]]:
        return self._encoding.get_all_atoms()

class Problem:

    def __init__(self, pddl):
        self.pddl = str(Path(pddl).absolute())
        self.rules = run_plasp(self.pddl)
        self.__init_encoding()
    
    def store(self):
        return {'pddl': self.pddl}

    @staticmethod
    def load(info):
        return Problem(**info)

    def __init_encoding(self):
        def get_null_pred(variables):
            nulls = []
            for var in variables:
                pred = var.arguments[0].arguments[0]
                if(pred.type == clingo.SymbolType.String):
                    nulls.append(pred.string)
            return nulls

        with create_solver() as ctl:
            self._init_solver(ctl)
            ctl.ground([(Logic.base)])
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(atoms=True))
            assert(len(symbols) == 1)
            variables = SymbolSet(symbols[0]).get_atoms('variable', 1)
            #Add predicates of arity 0
            ctl.addSymbols([clingo.Function('arity', [n, 0]) for n in get_null_pred(variables)])

            ctl.ground([('predicates', []), ('const', [])])
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(symbols) == 1)
            symset = SymbolSet(symbols[0])
            self._pred = symset.get_atoms('pred', 1)
            self._arity = symset.get_atoms('arity', 2)
            self._const = symset.get_atoms('const', 1)

    @property
    def predicates(self):
        return self._pred + self._arity

    @property
    def const(self):
        return self._const

    def _init_solver(self, ctl: Solver):
        ctl.load(Logic.problem)
        ctl.load(Logic.sample_encoding)
        ctl.add('base', [], self.rules)

    def initial_state(self) -> Node:
        with create_solver() as ctl:
            self._init_solver(ctl)
            ctl.ground([Logic.base, initial_state(), evaluate()])
            ctl.solve()
            ctl.ground([show_expanded()])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(models) == 1)
            #print('Models', models)
            symbols = SymbolSet(models[0])
        return Node(symbols, 0, None)

    def get_successors(self, node: Node) -> List[Node]:
        #print('Generating successors of', node.id)
        write_symbols(node.symbols.get_all_atoms(), '/home/ivan/Documents/ai/features/res/samples/clear/hard_complete/node.lp')
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            self._init_solver(ctl)
            ctl.addSymbols(node.symbols.get_all_atoms())
            ctl.ground([Logic.base, successor()])
            ctl.solve()
            ctl.cleanup()
            #print(ctl.getAtoms('validAction', 3))
            ctl.ground([evaluate()])
            ctl.solve()
            ctl.cleanup()
            ctl.ground([show_expanded()])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            ctl.cleanup()
        new_nodes = []
        for m in models:
            new_nodes.append(Node(SymbolSet(m), node.depth + 1, node))
        #print('Child nodes', [node.symbols.get_atoms('expanded', 2) for node in new_nodes])
        return new_nodes
            