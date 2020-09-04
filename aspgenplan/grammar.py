from aspgenplan.utils import SymbolSet, SymbolHash, symbol_to_str
from aspgenplan.solver import SolverType, create_solver, set_default_solver
from aspgenplan.sample import Sample, Node
from aspgenplan.logic import Logic
from pathlib import Path
from typing import List, Tuple, Union
import clingo
import sys
import math
import logging
import re
import argparse
import os
import json

class Concept:
    cardinality = ('cardinality', [])
    compareExp = ('compare_exp', [])
    keepExp = ('keep_exp', [])
    pruneExp = ('prune_exp', [])
    compareExpConc = ('compare_exp_conc', [])
    prune_file = str(Logic.logicPath/'prune.lp')

    @staticmethod
    def numberConc(start, first, gsize):
        return ('number_conc', [start, first, gsize])

    classify = ('classify', [])
    #classifyExp = ('classify_exp', [])

    @staticmethod
    def primitive(depth):
        return ('primitive', [depth])
    
    @staticmethod
    def negation(depth):
        return ('negation', [depth])

    @staticmethod
    def equalRole(depth):
        return ('equal_role', [depth])

    @staticmethod
    def conjunction(depth, in1, in2):
        return ('conjunction', [depth, in1, in2])

    @staticmethod
    def uni(depth):
        return ('uni', [depth])
        
    @staticmethod
    def exi(depth):
        return ('exi', [depth])

    @staticmethod
    def init_sets(max_exp, max_conc):
        return ('split_exp_conc', [max_exp, max_conc])

    @staticmethod
    def show_set(set):
        return ('show_exp_set', [set])

class Constant:
    def __init__(self, symbol: clingo.Symbol):
        assert isinstance(symbol, clingo.Symbol) and symbol.match('const', 1)
        self._symbol = symbol

    @property
    def name(self):
        return self._symbol.arguments[0]

    @property
    def symbol(self):
        return self._symbol

class StateSet:
    def __init__(self, symbols: SymbolSet):
        self._symbols = symbols
        self._hash = None
        self._st_hash = {}
        self._calc_hash()

    @property
    def symbols(self):
        return self._symbols

    def belong(self, state: Union[clingo.Symbol, SymbolHash, Node] = None, const: Union[clingo.Symbol, SymbolHash] = None):
        if state is not None:
            if isinstance(state, Node):
                state = state.name
            if not isinstance(state, clingo.Symbol) and not isinstance(state, SymbolHash):
                raise TypeError('Unexpected type {}'.format(type(state)))
            if not isinstance(state, SymbolHash):
                state = SymbolHash(state)
        if const is not None:
            if not isinstance(const, clingo.Symbol) and not isinstance(state, SymbolHash):
                raise TypeError('Unexpected type {}'.format(type(const)))
            if not isinstance(const, SymbolHash):
                const = SymbolHash(const)
        if state is not None:
            b_state = self._st_hash.get(state, {})
            if const is not None:
                return b_state.get(const, False)
            else:
                return b_state
        if const is not None:
            return dict([(st, True) for st in self._st_hash if st[const]])   
        return self._st_hash

    def _calc_hash(self):
        assert(self._hash == None)
        self._hash = 0
        for atom in self.symbols.get_atoms('belong', 3):
            st = atom.arguments[2]
            const = atom.arguments[0]
            self._hash += hash(str(st)+str(const))
            st = SymbolHash(st)
            if st not in self._st_hash:
                self._st_hash[st] = {}
            self._st_hash[st][SymbolHash(const)] = True
    
    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, ConceptObj):
            NotImplemented
        if hash(self) != hash(other): return False
        belong = self.belong()
        if len(belong) != len(other.belong()): return False
        for st in belong:
            if len(belong[st]) != len(other.belong(state=st)): return False
            for c in belong[st]:
                if not other.belong(state=st, const=c): return False
        return True

class Expression(StateSet):
    def __init__(self, symbols: SymbolSet, cost: int):
        super().__init__(symbols)
        self._cost = cost

    @property
    def cost(self):
        return self._cost

    def belong(self, state: Union[clingo.Symbol, Node] = None, const: Union[clingo.Symbol, Constant] = None):
        if const is not None:
            if isinstance(const, Constant):
                const = const.name
        return super().belong(state=state, const=const)

    def as_concept(self, id):
        with create_solver() as ctl:
            ctl.load(Logic.grammarFile)
            ctl.addSymbols(self.symbols.get_all_atoms())
            ctl.ground([Logic.base, ('to_concept', [id])])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            symbols = SymbolSet(sym[0])
        return ConceptObj(id, self.cost, symbols)

class ConceptObj(Expression):
    def __init__(self, id, cost, symbols):
        super().__init__(symbols, cost)
        self._id = id

    @property
    def id(self):
        return self._id

    @property
    def concept(self):
        return self._symbols.get_atoms('conc', 2)

    @property
    def conceptId(self):
        return self._symbols.get_atoms('conceptId', 2)

    def store(self):
        info = {
            'id': self.id,
            'cost': self.cost,
            'symbols': self.symbols.to_pickle()
        }
        return info
    
    @staticmethod
    def load(info):
        info['symbols'] = SymbolSet.from_pickle(info['symbols'])
        return ConceptObj(**info)

class PreRole(StateSet):
    def __init__(self, symbols: SymbolSet, cost: int):
        super().__init__(symbols)
        self._cost = cost

    @property
    def cost(self):
        return self._cost

    def belong(self, state: Union[clingo.Symbol, Node] = None, const: Union[clingo.Symbol, Tuple[Constant, Constant]] = None):
        if const is not None:
            if isinstance(const, tuple):
                assert len(const) == 2
                const = clingo.Function('', [c.name for c in const])
        return super().belong(state=state, const=const)

    def as_role(self, id):
        with create_solver() as ctl:
            ctl.load(Logic.grammarFile)
            ctl.addSymbols(self.symbols.get_all_atoms())
            ctl.ground([Logic.base, ('to_role', [id])])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            symbols = SymbolSet(sym[0])
        return Role(id, self.cost, symbols)

class Role(PreRole):
    def __init__(self, id, cost, symbols):
        super().__init__(symbols, cost)
        self._id = id

    @property
    def id(self):
        return self._id

    @property
    def role(self):
        return self._symbols.get_atoms('role', 2)

    @property
    def roleId(self):
        return self._symbols.get_atoms('roleId', 2)

    def store(self):
        info = {
            'id': self.id,
            'cost': self.cost,
            'symbols': self.symbols.to_pickle()
        }
        return info
    
    @staticmethod
    def load(info):
        info['symbols'] = SymbolSet.from_pickle(info['symbols'])
        return Role(**info)

class Primitive:
    def __init__(self, sample: Union[Sample]):
        self.sample = sample

    def __call__(self) -> List[Expression]:
        logging.debug('Calling Primitive')
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.grammarFile])
            ctl.addSymbols(self.sample.get_sample())
            ctl.ground([Logic.base, Concept.primitive(1), ('cardinality', []), ('show_exp', [])])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [Expression(SymbolSet(model), 1) for model in models]

class Negation:
    def __init__(self, sample: Union[Sample], primitive: List[ConceptObj]):
        self.sample = sample
        self.primitive = primitive
    def __call__(self) -> List[Expression]:
        logging.debug('Calling Negation({})'.format([c.id for c in self.primitive]))
        sym = []
        for c in self.primitive:
            sym += c.symbols.get_all_atoms()
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.grammarFile])
            ctl.addSymbols(sym)
            ctl.addSymbols(self.sample.get_const() + self.sample.get_states())
            ctl.ground([Logic.base, Concept.negation(2), ('cardinality', []), ('show_exp', [])])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))

        return [Expression(SymbolSet(model), 2) for model in models]

class EqualRole:
    def __init__(self, sample, roles: List[Role]):
        self.sample = sample
        self.roles = roles

    def __call__(self) -> List[Expression]:
        logging.debug('Calling EqualRole')
        sym = []
        for r in self.roles:
            sym += r.symbols.get_all_atoms()
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.grammarFile])
            ctl.addSymbols(self.sample.get_const() + self.sample.get_states())
            ctl.addSymbols(sym)
            ctl.ground([Logic.base, Concept.equalRole(3), ('cardinality', []), ('show_exp', [])])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [Expression(SymbolSet(model), 3) for model in models]

class Conjunction:
    def __init__(self, sample, concept1: List[ConceptObj], concept2: List[ConceptObj]):
        self.sample = sample
        self.concept1 = concept1
        self.concept2 = concept2

    def __call__(self) -> List[Expression]:
        logging.debug('Calling Conjunction({},{})'.format([c.id for c in self.concept1], [c.id for c in self.concept2]))
        sym = []
        for c in self.concept1:
            sym += c.symbols.get_all_atoms()
        for c in self.concept2:
            sym += c.symbols.get_all_atoms()
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.grammarFile])
            ctl.addSymbols(sym)
            ctl.addSymbols(self.sample.get_const() + self.sample.get_states())
            cost = self.concept1[0].cost + self.concept2[0].cost + 1
            ctl.ground([Logic.base, Concept.conjunction(cost, self.concept1[0].cost, self.concept2[0].cost), ('cardinality', []), ('show_exp', [])])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [Expression(SymbolSet(model), cost) for model in models]

class Uni:
    def __init__(self, sample, concept: List[ConceptObj], roles):
        self.sample = sample
        self.concept = concept
        self.roles = roles

    def __call__(self) -> List[Expression]:
        logging.debug('Calling Uni({})'.format([c.id for c in self.concept]))
        sym = []
        for c in self.concept:
            sym += c.symbols.get_all_atoms()
        for r in self.roles:
            sym += r.symbols.get_all_atoms()
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.grammarFile])
            ctl.addSymbols(sym)
            ctl.addSymbols(self.sample.get_const() + self.sample.get_states())
            cost = self.concept[0].cost + 2
            ctl.ground([Logic.base, Concept.uni(cost), ('cardinality', []), ('show_exp', [])])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [Expression(SymbolSet(model), cost) for model in models]

class Exi:
    def __init__(self, sample, concept: List[ConceptObj], roles: List[Role]):
        self.sample = sample
        self.concept = concept
        self.roles = roles
    def __call__(self) -> List[Expression]:
        logging.debug('Calling Exi({})'.format([c.id for c in self.concept]))
        sym = []
        for c in self.concept:
            sym += c.symbols.get_all_atoms()
        for r in self.roles:
            sym += r.symbols.get_all_atoms()
        with create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.grammarFile])
            ctl.addSymbols(sym)
            ctl.addSymbols(self.sample.get_const() + self.sample.get_states())
            cost = self.concept[0].cost + 2
            ctl.ground([Logic.base, Concept.exi(cost), ('cardinality', []), ('show_exp', [])])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [Expression(SymbolSet(model), cost) for model in models]

def roles(sample):
    with create_solver(args=dict(arguments=['-n 0'])) as ctl:
        ctl.load([Logic.grammarFile])
        ctl.addSymbols(sample.get_sample())
        ctl.ground([Logic.base, Logic.roles])
        ctl.solve()
        ctl.ground([('show_role', [])])
        models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
    return [PreRole(SymbolSet(model), 1) for model in models]

class Grammar:
    def __init__(self, sample: Sample):
        self.sample = sample
        self.concepts = {}
        self.conceptNum = {}
        self._conc_set = {}
        self._role_set = {}
        self.cost = 0
        self._roles = None
        self.total_concepts = 0
        self._last_role = -1
        self._last_conc = -1

    @staticmethod
    def load(sample, path):
        grammar = Grammar(sample)
        path = Path(path)
        with open(str(path/'grammar.json'), 'r') as fp:
            info = json.load(fp)
        def load(cost, roles, concepts):
            grammar.cost = cost
            for r in roles:
                role = Role.load(r)
                grammar.add_role(role)
            for c in concepts:
                conc = ConceptObj.load(c)
                grammar.add_concept(conc)
        load(**info)
        return grammar
    
    def store(self, path):
        path = Path(path)
        if not path.is_dir():
            try:
                path.mkdir()
            except (FileNotFoundError, FileExistsError) as e:
                print(repr(e))
                sys.exit()
        info = {
            'cost' : self.cost,
            'roles' : [r.store() for r in self.get_roles()] if self.get_roles() != None else None,
            'concepts' : [c.store() for c in self.get_concepts()]
        }
        with open(str(path/'grammar.json'), 'w') as fp:
            json.dump(info, fp)
        with open(str(path/'concepts.lp'), 'w') as fp:
            for c in self.get_concepts():
                fp.write(symbol_to_str(c.symbols.get_all_atoms()))
        with open(str(path/'roles.lp'), 'w') as fp:
            for r in self.get_roles():
                fp.write(symbol_to_str(r.symbols.get_all_atoms()))

    def add_roles(self):
        preroles = roles(self.sample)
        if self.get_roles() == None:
            self._roles = []
        self._role_set = {}
        for r in preroles:
            if r not in self._role_set:
                nr = r.as_role(self._last_role+1)
                self.add_role(nr)

    def add_role(self, role: Role):
        logging.debug('Adding role {}'.format(role.roleId))
        if self.get_roles() == None:
            self._roles = []
        self._roles.append(role)
        self._role_set[role] = True
        self._last_role = max(self._last_role, role.id)

    def get_roles(self):
        return self._roles

    def batch_cost(self, cost, batch):
        for i in range(0, len(self.concepts[cost]), batch):
            yield self.concepts[cost][i:i+batch]

    def _concepts_depth(self, depth, batch=1):
        variables = []
        if depth == 1:
            return [Primitive(self.sample)]
        elif depth == 2:
            for conc in self.batch_cost(depth-1, batch):
                variables.append(Negation(self.sample, conc))
            return variables
        else:
            if depth == 3:
                variables.append(EqualRole(self.sample, self.get_roles()))
            b_conj = int(math.sqrt(batch))
            b_crole = batch // len(self.get_roles())
            b_conj = b_conj if b_conj > 0 else 1
            b_crole = b_crole if b_crole > 0 else 1
            for i in range(1, (depth + 1)//2):
                for conc1 in self.batch_cost(i, b_conj):
                    for conc2 in self.batch_cost(depth - i - 1, b_conj):
                        variables.append(Conjunction(self.sample, conc1, conc2))
            for conc in self.batch_cost(depth - 2, b_crole):
                variables.append(Uni(self.sample, conc, self.get_roles()))
                variables.append(Exi(self.sample, conc, self.get_roles()))
            return variables

    def get_concepts(self) -> List[ConceptObj]:
        depths = list(self.concepts.keys())
        depths.sort()
        concepts = []
        for d in depths:
            concepts += self.concepts[d]
        return concepts

    def conceptIterator(self):
        depths = list(self.concepts.keys())
        depths.sort()

        for d in depths:
            for conc in self.concepts[d]:
                yield conc

    def batchIterator(self, batch=1):
        depths = list(self.concepts.keys())
        depths.sort()
        l = []
        for d in depths:
            for conc in self.concepts[d]:
                l.append(conc)
                if len(l) == batch:
                    yield l
                    l = []
        if len(l):
            yield l

    def expand_grammar(self, max_depth, batch=1):
        logging.debug("Starting {}. Ending {}".format(self.cost, max_depth))
        
        if self.get_roles() is None: self.add_roles()
        
        for depth in range(self.cost+1, max_depth+1):
            print('Depth {}:'.format(depth))
            self.concepts[depth] = []
            self.conceptNum[depth] = 0
            expressions = self._concepts_depth(depth, batch=batch)
            logging.debug('Number of concept groups: {}'.format(len(expressions)))
            for i, exp in enumerate(expressions):
                if (i+1)%5==0 or i+1==len(expressions):
                    print('Concepts {}/{}'.format(i+1, len(expressions)))
                new_concepts = exp()
                for conc in new_concepts:
                    if conc not in self._conc_set:
                        nconc = conc.as_concept(self._last_conc+1)
                        self.add_concept(nconc)
            print("Total {}: {} concepts.\n".format(depth, self.conceptNum[depth]))
        self.cost = max_depth

    def add_concept(self, concept: ConceptObj):
        logging.debug('Adding {} of cost {}'.format(concept.conceptId, concept.cost))
        if concept.cost not in self.concepts:
            self.concepts[concept.cost] = []
        self.concepts[concept.cost].append(concept)
        if concept.cost not in self.conceptNum:
            self.conceptNum[concept.cost] = 0
        self.conceptNum[concept.cost] += 1
        self.total_concepts += 1
        self._conc_set[concept] = True
        self._last_conc = max(self._last_conc, concept.id)

    def get_cost(self, cost):
        if cost > self.cost or cost <= 0:
            raise ValueError('Concepts of cost {} have not been generated.'.format(cost))
        return self.concepts[cost]

    def is_generated(self, cost):
        return cost in self.concepts

def countFile(file_name):
    count = 0
    with open(str(file_name), 'r') as f:
        for line in f:
            if re.match(r'conc\(.*?,\d+?\)\.', line):
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='Sample file path')
    parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('max_depth', type=int, help='Maximum concept depth')
    parser.add_argument('-load', help='Path to existing concepts')
    parser.add_argument('-batch', type=int, default=1, help='Max batch size used for generating concepts')
    
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--proc',help="Runs clingo solver in separate process",
        action="store_const", dest="solver", const=SolverType.PROCESS, default=SolverType.SIMPLE)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    set_default_solver(args.solver)
    print(args.solver)
    sample = Sample.load(args.sample)
    if args.load != None:
        grammar = Grammar.load(sample, args.load)
    else:
        grammar = Grammar(sample)
    import time
    start = time.time()
    print(Logic.logicPath, Logic.grammarFile)
    grammar.expand_grammar(args.max_depth, batch=args.batch)
    print('Total number of concepts: {}'.format(grammar.total_concepts))
    print("Took {}s.".format(round(time.time()-start, 2)))
    grammar.store(args.out_dir)