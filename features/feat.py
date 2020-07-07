from features.knowledge import Solver, ConceptFile, prune_symbols, Comparison, number_symbols
import logging
from features.model_util import write_symbols, count_symbols, get_symbols
from features.logic import Logic
from features.grammar import Grammar, Concept
from features.prune import Pruneable
from features.comparison import CompareFeature
from typing import List, Union
from pathlib import Path
import clingo
from argparse import ArgumentParser
import os

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
    def __init__(self, sample, transitions):
        self.sample = sample
        self.transitions = transitions

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling Nullary')
        with Solver.open() as ctl:
            ctl.load([Logic.featureFile, self.sample, self.transitions])
            ctl.ground([Logic.base, Feature.primitiveFeature, Feature.processFeat])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class ConceptFeat:
    def __init__(self, sample, concepts: Union[ConceptFile, List[ConceptFile]], transitions):
        self.sample = sample
        self.concepts = [concepts] if isinstance(concepts, ConceptFile) else concepts
        self.transitions = transitions

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling ConceptFeat({})'.format([c.name for c in self.concepts]))
        with Solver.open() as ctl:
            ctl.load([Logic.featureFile, self.sample, self.transitions] + [conc.file for conc in self.concepts])
            ctl.ground([Logic.base, Feature.conceptFeature, Feature.processFeat])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Distance:
    def __init__(self, sample, roles, concepts: Union[ConceptFile, List[ConceptFile]], transitions, max_cost=8):
        self.sample = sample
        self.roles = roles
        self.concepts = [concepts] if isinstance(concepts, ConceptFile) else concepts
        self.transitions = transitions
        self.max_cost = max_cost

    def __call__(self) -> List[clingo.Symbol]:
        logging.debug('Calling ConceptFeat({})'.format(self.concepts.name))
        with Solver.open() as ctl:
            ctl.load([Logic.featureFile, self.sample, self.transitions, self.roles] + [conc.file for conc in self.concepts])
            ctl.ground([Logic.base, Feature.distFeature(self.max_cost), Feature.processFeat])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

def get_transitions(sample):
    with Solver.open() as ctl:
        ctl.load([sample, Logic.pruneFile])
        ctl.ground([Logic.base, Logic.transitions])
        result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
    return result

class Features:
    def __init__(self, sample, concept_path, output):
        self.sample_path = sample
        self.concepts = Grammar(sample, concept_path)
        self.path = Path(output)
        self.features = str(self.path/'features.lp')
        with open(self.features, 'w'): pass
        self.total_features = 0
        self.compare = CompareFeature(comp_type=CompareFeature.STANDARD)

    def list_features(self, max_cost, batch=1, distance=False):
        ans = []
        if max_cost > 0:
            ans.append(Nullary(self.sample_path, self.transitions))
        self.concepts.loadProgress(max_cost)
        for conc in self.concepts.batchIterator(batch=batch):
            ans.append(ConceptFeat(self.concepts.simple, conc, self.transitions))
        if distance:
            pass #TODO
        return ans

    def difference(self, symbols, batch=10):
        logging.debug('Prune with existing features')
        symbols = prune_symbols(
            symbols,
            Logic.pruneFile,
            Feature.compareFeature,
            self.compare,
            Feature.pruneFeature,
            files=[self.features])
        return symbols

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

    def add_transitions(self):
        symbols = get_transitions(self.sample_path)
        self.transitions = str(self.path/'transitions.lp')
        write_symbols(symbols, self.transitions)

    def generate(self, max_cost=8, batch=1, max_pre=50, max_feat=50, distance=False):
        logging.debug('Features with max cost {}'.format(max_cost))
        self.add_transitions()

        features = self.list_features(max_cost, batch=batch, distance=distance)
        for feat in features:
            symbols = feat()
            logging.debug('Initial features: {}'.format(count_symbols(symbols, 'prefeature', 1)))
            logging.debug('Initial bool: {}'.format(count_symbols(symbols, 'bool', 1)))
            logging.debug('Initial num: {}'.format(count_symbols(symbols, 'num', 1)))
            symbols = self.prune(symbols, max_pre, max_feat)
            self.addFeatures(symbols)
            del symbols[:]
        print('Generated {} features'.format(self.total_features))

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
        action="store_const", dest="solver", const=Solver.PROCESS, default=Solver.SIMPLE)
    parser.add_argument('--batch',action='store',default=1, type=int, help='Concept files used simultaneaously in feature generation.')
    parser.add_argument('--atom',default=50, type=int, help='Max new features prunned together')
    parser.add_argument('--comp',default=50, type=int, help='Max number of known features used for prunning simultaneously')

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    features = Features(args.sample, args.concepts, args.out_dir)
    features.generate(max_cost=args.max_cost, batch=args.batch, max_pre=args.atom, max_feat=args.comp, distance=args.dist)