from features.knowledge import ConceptFile, number_symbols
from features.comparison import Comparison
import features.solver as solver
from features.solver import SolverType
import logging
from features.model_util import write_symbols, count_symbols, get_symbols
from features.logic import Logic
from features.grammar import Grammar, Concept
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

class Nullary:
    def __init__(self, sample: Union[Sample, SampleFile]):
        self.sample = sample

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling Nullary')
        with solver.create_solver() as ctl:
            ctl.load([Logic.featureFile])
            ctl.addSymbols(self.sample.get_sample())
            ctl.ground([Logic.base, Feature.primitiveFeature, Feature.processFeat])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class ConceptFeat:
    def __init__(self, sample: Union[Sample, SampleFile], concepts: Union[ConceptFile, List[ConceptFile]]):
        self.sample = sample
        self.concepts = [concepts] if isinstance(concepts, ConceptFile) else concepts

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling ConceptFeat({})'.format([c.name for c in self.concepts]))
        with solver.create_solver() as ctl:
            ctl.load([Logic.featureFile] + [conc.file for conc in self.concepts])
            ctl.addSymbols(self.sample.get_states())
            ctl.addSymbols(self.sample.get_transitions())
            ctl.addSymbols(self.sample.get_const())
            #ctl.addSymbols(self.sample.get_sample())
            ctl.ground([Logic.base, Feature.conceptFeature, Feature.processFeat])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Distance:
    def __init__(self, sample, roles, concepts: Union[ConceptFile, List[ConceptFile]], max_cost=8):
        self.sample = sample
        self.roles = roles
        self.concepts = [concepts] if isinstance(concepts, ConceptFile) else concepts
        self.max_cost = max_cost

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling ConceptFeat({})'.format(self.concepts.name))
        with solver.create_solver() as ctl:
            ctl.load([Logic.featureFile, self.roles] + [conc.file for conc in self.concepts])
            ctl.addSymbols(self.sample.get_states())
            ctl.addSymbols(self.sample.get_transitions())
            ctl.addSymbols(self.sample.get_const())
            ctl.ground([Logic.base, Feature.distFeature(self.max_cost), Feature.processFeat])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Features:
    def __init__(self, sample: Union[Sample, SampleFile], grammar, output, distance=False, load=False):
        self.sample = sample
        self.concepts = grammar
        path = Path(output)
        self.features = str(path / 'features.lp')
        self.info = str(path / 'features.json')
        self.left = []
        if not path.is_dir():
            path.mkdir()

        if load:
            self.load_info()
            if self.__distance != distance:
                raise ValueError('Invalid distance argument {}. Loaded features have distance {}'.format(distance, self.__distance))
        else:
            with open(self.features, 'w'): pass
            self.total_features = 0
            self.cost = 0
            self.__comp_type = CompareFeature.STANDARD
            self.compare = CompareFeature(comp_type=self.__comp_type)
            self.__distance = distance

    def prune(self, features, max_pre, max_feat):
        return Feature.prune_symbols(
                    features, self.features, self.compare,
                    max_atoms=max_pre, max_comp=max_feat
                )

    def addFeatures(self, symbols):
        feat_n, symbols = number_symbols(
            symbols,
            self.total_features,
            Feature.numberFeat
        )
        logging.info('{} features.'.format(feat_n))
        write_symbols(symbols, self.features, type_='a')
        del symbols[:]
        self.total_features += feat_n

    def feature_count(self):
        return self.total_features

    def generate(self, max_cost=None, batch=1, max_f=None, max_pre=50, feat_prune=50, **kwargs):
        max_cost = self.cost+1 if max_cost is None else max_cost
        logging.debug('Features with max cost {}'.format(max_cost))

        features = self.left
        self.left = []
        if self.cost == 0 and max_cost > 0:
            features.append(Nullary(self.sample))

        if self.concepts.cost < max_cost:
            self.concepts.expand_grammar(max_cost, **kwargs)
        conc = []
        for i in range(self.cost+1, max_cost+1):
            conc += self.concepts.get_cost(i)
        for i in range(0,len(conc),batch):
            features.append(ConceptFeat(self.sample, conc[i:i+batch]))
        if self.__distance:
            pass #TODO

        for feat in features:
            if max_f == None or self.total_features < max_f:
                symbols = feat()
                logging.debug('Initial features: {}'.format(count_symbols(symbols, 'prefeature', 1)))
                logging.debug('Initial bool: {}'.format(count_symbols(symbols, 'bool', 1)))
                logging.debug('Initial num: {}'.format(count_symbols(symbols, 'num', 1)))
                symbols = self.prune(symbols, max_pre, feat_prune)
                self.addFeatures(symbols)
                del symbols[:]
            else:
                self.left.append(feat)
        print('Generated {} features'.format(self.total_features))
        logging.debug('Left: {}'.format(self.left))
        self.cost = max_cost
        self.update_info()

    def update_info(self):
        data = {
            'total_features': self.total_features,
            'cost': self.cost,
            'compare': self.__comp_type,
            'distance': self.__distance
        }
        with open(self.info, 'w') as fp:
            json.dump(data, fp)

    def load_info(self):
        with open(self.info, 'r') as fp:
            data = json.load(fp)
        self.total_features = data['total_features']
        self.cost = data['cost']
        self.__comp_type = data['compare']
        self.compare = CompareFeature(comp_type=self.__comp_type)
        self.__distance = data['distance']

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
    sample = Sample(load_path=args.sample)
    grammar = Grammar(sample, args.concepts)
    grammar.load_progress(args.max_cost)
    features = Features(sample, grammar, args.out_dir, distance=args.dist, load=args.load)
    features.generate(max_cost=args.max_cost, batch=args.batch, max_pre=args.atom, max_feat=args.comp)