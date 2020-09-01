from features.knowledge import ConceptFile, number_symbols
from features.comparison import Comparison
import features.solver as solver
from features.solver import SolverType
import logging
from features.model_util import write_symbols, count_symbols, get_symbols, SymbolSet, SymbolHash, symbol_to_str
from features.logic import Logic
from features.grammar import Grammar, Concept, ConceptObj
from features.prune import Pruneable
from features.comparison import CompareFeature
from features.sample.sample import Sample, SampleFile
from typing import List, Union
from pathlib import Path
import clingo
import sys
from argparse import ArgumentParser
from pathlib import Path
import os
import json
from abc import ABC, abstractmethod

class Feature(Pruneable):
    prune_file = str(Logic.logicPath/'prune.lp')
    processFeat = ('feature', [])
    comparePreFeature = ('compare_prefeature', [])
    compareFeature = ('compare_feature', [])
    pruneFeature = ('prune_feature', [])
    @staticmethod
    def numberFeat(start):
        return ('number_feat', [start])
    @staticmethod
    def divide_feat(start, first, gsize):
        return ('divide_feat', [start, first, gsize])
    classifyFeat = ('classify_feat', [])
    primitiveFeature = ('primitiveFeature', [])
    conceptFeature = ('conceptFeature', [])
    @staticmethod
    def distFeature(k):
        return ('distFeature', [k])

    @staticmethod
    def init_sets(max_pre, max_feat):
        return ('split_features', [max_pre, max_feat])

    @staticmethod
    def show_set(set):
        return ('show_features', [set])


class PreFeature:
    def __init__(self, symbols: SymbolSet):
        self._symbols = symbols
        self._hash = None
        self._val_hash = {}
        self._del_hash = {}
        self._calc_hash()

    def to_feature(self, id):
        with solver.create_solver() as ctl:
            ctl.load(Logic.featureFile)
            ctl.addSymbols(self._symbols.get_all_atoms())
            ctl.ground([Logic.base, ('to_feature', [id])])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
            symbols = SymbolSet(sym)
        return FeatureObj(id, symbols)

    @property
    def symbols(self) -> SymbolSet:
        return self._symbols

    @property
    def value(self):
        return [(s.arguments[2], s.arguments[1]) for s in self.symbols.get_atoms('qualValue', 3)]
    
    @property
    def delta(self):
        return [(tuple(s.arguments[0:2]), s.arguments[3]) for s in self.symbols.get_atoms('delta', 4)]

    def value_state(self, state: clingo.Symbol):
        return self._val_hash.get(SymbolHash(state), None)

    def delta_states(self, state1: clingo.Symbol, state2: clingo.Symbol):
        return self._del_hash.get((SymbolHash(state1), SymbolHash(state2)), None)

    def _calc_hash(self):
        assert(self._hash == None)
        self._hash = 0
        for st, val in self.value:
            self._hash += hash(str(st)+str(val))
            assert(SymbolHash(st) not in self._val_hash)
            self._val_hash[SymbolHash(st)] = SymbolHash(val)
        for (st1, st2), d in self.delta:
            self._hash += hash(str(st1)+str(st2)+str(d))
            assert((SymbolHash(st1), SymbolHash(st2)) not in self._del_hash)
            self._del_hash[(SymbolHash(st1), SymbolHash(st2))] = SymbolHash(d)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, PreFeature):
            raise NotImplementedError
        if hash(self) != hash(other):
            return False
        value = self.value
        delta = self.delta
        if len(value) != len(other.value): return False
        if len(delta) != len(other.delta): return False
        for state in value:
            if self.value_state(state) != other.value_state(state):
                return False
        for st1, st2 in delta:
            if self.delta_states(st1, st2) != other.delta_states(st1, st2):
                return False
        return True


class FeatureObj(PreFeature):

    def __init__(self, id, symbols):
        super().__init__(symbols)
        self._id = id

    @property
    def id(self):
        return self._id

    @property
    def feature(self):
        return self.symbols.get_atoms('feature', 1)

    @property
    def featureId(self):
        return self.symbols.get_atoms('featureId', 2)

    def to_json(self):
        data = {
            'id': self.id,
            'symbols': self.symbols.to_pickle()
        }
        return data

    @staticmethod
    def from_json(jsondict):
        def load(id, symbols):
            return FeatureObj(id, SymbolSet.from_pickle(symbols))
        return load(**jsondict)

class Nullary:
    def __init__(self, sample: Union[Sample, SampleFile]):
        self.sample = sample

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling Nullary')
        with solver.create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.featureFile])
            ctl.addSymbols(self.sample.get_sample())
            ctl.ground([Logic.base, Feature.primitiveFeature, Feature.processFeat])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [PreFeature(SymbolSet(model)) for model in models]

class ConceptFeat:
    def __init__(self, sample: Union[Sample, SampleFile], concepts: List[ConceptObj]):
        self.sample = sample
        self.concepts = concepts

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling ConceptFeat({})'.format([c.id for c in self.concepts]))
        sym = []
        for c in self.concepts:
            sym += c.symbols.get_all_atoms()
        with solver.create_solver(args=dict(arguments=['-n 0'])) as ctl:
            ctl.load([Logic.featureFile])
            ctl.addSymbols(sym)
            ctl.addSymbols(self.sample.get_states())
            ctl.addSymbols(self.sample.get_transitions())
            ctl.addSymbols(self.sample.get_const())
            #ctl.addSymbols(self.sample.get_sample())
            ctl.ground([Logic.base, Feature.conceptFeature, Feature.processFeat])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return [PreFeature(SymbolSet(model)) for model in models]

class Distance:
    def __init__(self, sample: Sample, conc1: List[ConceptObj], roles, conc:List[ConceptObj], conc2:List[ConceptObj], max_cost=8):
        self.sample = sample
        self.roles = roles
        self.conc1 = conc1
        self.conc = conc
        self.conc2 = conc2
        self.max_cost = max_cost

    def __call__(self) -> List[FeatureObj]:
        logging.debug('Calling Distance({})'.format(tuple(([c.id for c in co] for co in (self.conc1, self.conc, self.conc2)))))
        with solver.create_solver(args=dict(arguments=['-n 0'])) as ctl:
            possym = []
            symbols = []
            for i, cs in enumerate((self.conc1, self.conc, self.conc2)):
                s = []
                for c in cs:
                    s += c.concept
                    symbols += c.symbols.get_all_atoms()
                ctl.load([Logic.featureFile])
                ctl.addSymbols(s)
                ctl.ground([Logic.base, ('dist_pos', [i])])
                models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
                assert(len(models) == 1)
                possym += models[0]
                ctl.reset(arguments=['-n 0'])
            #print(possym)
            ctl.load([Logic.featureFile, self.roles])
            ctl.addSymbols(self.sample.get_states())
            ctl.addSymbols(self.sample.get_transitions())
            ctl.addSymbols(self.sample.get_const())
            ctl.addSymbols(symbols + possym)
            ctl.ground([Logic.base, Feature.distFeature(self.max_cost)])
            count = 1
            while True:
                ctl.ground([('dist_const', [count-1])])
                models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
                assert(len(models) == 1)
                model = models[0]
                #print(model)
                if len(model) != count:
                    break
                count += 1
            ctl.ground([('value_dist', []), Feature.processFeat])
            models = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            logging.debug('Found {} features'.format(len(models)))
        return [PreFeature(SymbolSet(model)) for model in models]

class Features:
    def __init__(self, sample: Union[Sample, SampleFile], grammar, distance=False):
        self.sample = sample
        self.concepts = grammar
        self.features = []
        self._feat_set = {}
        self.cost = 0
        self.total_features = 0
        self._distance = distance

    def get_features(self, num=None) -> List[FeatureObj]:
        num = num if num != None else self.feature_count()
        print(num)
        return self.features[0:num]

    def feature_count(self):
        return self.total_features

    def generate(self, max_cost=None, max_f=None, batch=1, batch_g=1):
        max_cost = self.cost+1 if max_cost is None else max_cost
        print('Features with max cost {}'.format(max_cost))

        features = []
        if self.cost == 0 and max_cost > 0:
            features.append(Nullary(self.sample))

        if self.concepts.cost < max_cost:
            self.concepts.expand_grammar(max_cost, batch=batch_g)
        conc = []
        for i in range(self.cost+1, max_cost+1):
            conc += self.concepts.get_cost(i)
        for i in range(0,len(conc),batch):
            features.append(ConceptFeat(self.sample, conc[i:i+batch]))
        b_dist = int((batch // len(self.concepts.get_roles())) ** (1.0/3.0))
        b_dist = b_dist if b_dist > 0 else 1
        if self._distance:
            for cost in range(self.cost+1, max_cost+1):
                for i in range(1, cost-2):
                    for j in range(1, cost-1-i):
                        for k in range(1, cost-i-j):
                            for conc1 in self.concepts.batch_cost(i, b_dist):
                                for conc in self.concepts.batch_cost(j, b_dist):
                                    for conc2 in self.concepts.batch_cost(k, b_dist):
                                        features.append(
                                            Distance(
                                                self.sample, conc1,
                                                self.concepts.get_roles(), conc,
                                                conc2, max_cost=max_cost))
        for i, feat in enumerate(features):
            if max_f != None and self.feature_count >= max_f: break
            print('Features {}/{}'.format(i+1, len(features)))
            fs = feat()
            #logging.debug('Initial features: {}'.format(len(fs)))
            #logging.debug('Initial bool: {}'.format(count_symbols(symbols, 'bool', 1)))
            #logging.debug('Initial num: {}'.format(count_symbols(symbols, 'num', 1)))
            for f in fs:
                if f not in self._feat_set:
                    self.add_feature(f.to_feature(self.total_features))
                if max_f != None and self.feature_count >= max_f: break

        print('Generated {} features'.format(self.total_features))
        self.cost = max_cost

    def add_feature(self, feat: PreFeature):
        logging.debug('Adding {}'.format(feat.featureId))
        self.features.append(feat)
        self.total_features += 1
        self._feat_set[feat] = True

    def store(self, path):
        path = Path(path)
        if not path.is_dir():
            try:
                path.mkdir()
            except (FileNotFoundError, FileExistsError) as e:
                print(repr(e))
                sys.exit()
        with open(str(path/'features.json'), 'w') as fp:
            json.dump(self.to_json(), fp)
        with open(str(path/'features.lp'), 'w') as fp:
            for f in self.features:
                fp.write(symbol_to_str(f.symbols.get_all_atoms()))

    @staticmethod
    def load(sample, grammar, path):
        with open(str(Path(path)/'features.json'), 'r') as fp:
            return Features.from_json(sample, grammar, json.load(fp))

    def to_json(self):
        data = {
            'features': [f.to_json() for f in self.features],
            'cost': self.cost,
            '_distance': self._distance
        }
        return data

    @staticmethod
    def from_json(sample, grammar, jsondict):
        def load(features, cost, _distance):
            feat = Features(sample, grammar, _distance)
            feat.cost = cost
            for f in features:
                feat.add_feature(FeatureObj.from_json(f))
            return feat
        return load(**jsondict)


import time

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('sample', type=str, help='Sample file path')
    parser.add_argument('concepts', type=str, help='Input concepts/roles directory')
    parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('max_cost', type=int, help='Maximum feature cost')
    parser.add_argument('--dist', action='store_true', help='Option to generate distance features')
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--proc',help="Runs clingo solver in separate process",
        action="store_const", dest="solver", const=SolverType.PROCESS, default=SolverType.SIMPLE)
    parser.add_argument('--batch',action='store',default=1, type=int, help='Concept files used simultaneaously in feature generation.')
    parser.add_argument('--atom',default=50, type=int, help='Max new features prunned together')
    parser.add_argument('--comp',default=50, type=int, help='Max number of known features used for prunning simultaneously')
    parser.add_argument('--load',action='store_true', help='Whether to load features from given path')

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    sample = Sample.load(args.sample)
    grammar = Grammar(sample)
    grammar.expand_grammar(args.max_cost, batch=args.batch)
    #grammar.load_progress(args.max_cost)
    features = Features(sample, grammar, distance=args.dist)
    start = time.time()
    features.generate(max_cost=args.max_cost, batch=args.batch)
    print('Took {}s'.format(time.time() - start))
    features.store(args.out_dir)