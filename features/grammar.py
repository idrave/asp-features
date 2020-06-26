
import clingo
import sys
import logging
import pathlib
import re
import argparse
import os
from pathlib import Path
from model_util import check_multiple
from model_util import to_model_list, ModelUtil, filter_symbols, check_multiple, add_symbols, write_symbols
from features.logic import Logic, Concept
from features.knowledge import ClingoSolver, ConceptFile, splitSymbols
from typing import List, Tuple

class Comparison:
    def __init__(self, comp_type='standard'):
        self.file = str(Logic.logicPath/'differ.lp')
        types = {
            'standard' : self.__standardCompare,
            'fast' : self.__fastCompare,
            'mixed' : self.__mixedCompare,
            'feature' : self.__featureCompare
        }
        self.type = types.get(comp_type, None)
        if self.type is None:
            raise RuntimeError("Invalid comparison type.")

    def __call__(self, ctl):
        self.type(ctl)

    def __standardCompare(self, ctl: ClingoSolver):
        ctl.load(str(self.file))
        ctl.ground([('standard_differ', [])])
        ctl.solve()

    def __fastCompare(self, ctl: ClingoSolver):
        ctl.load(str(self.file))
        ctl.ground([('fast_differ', [])])
        ctl.solve()
    
    def __mixedCompare(self, ctl: ClingoSolver):
        ctl.load(str(self.file))
        ctl.ground([('optimal_differ_start', [])])
        ctl.solve()
        ctl.ground([('optimal_differ_end', [])])
        ctl.solve()

    def __featureCompare(self, ctl: ClingoSolver):
        ctl.load(str(self.file))
        ctl.ground([('feature_differ', [])])
        ctl.solve()

def prune_symbols(symbols: List[clingo.Symbol], prune_file: str,
                  compare_prog: Tuple, compare: Comparison, prune_prog: Tuple,
                  files: List =[]):
    with ClingoSolver() as ctl:
        ctl.load(prune_file)
        ctl.load(files)
        ctl.addSymbols(symbols)
        ctl.ground([Logic.base, compare_prog])
        compare(ctl)
        #print(files, ctl.countAtoms('exp', 2), ctl.countAtoms('conc', 2), ctl.countAtoms('compare', 2), compare_prog, prune_prog)
        ctl.ground([prune_prog])
        result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        #print(ctl.getAtoms('belong', 3))
        #logging.debug('Result: {}, {}'.format(ctl.countAtoms('keep', 1), len(result)))
    return result

class Primitive:
    def __init__(self, sample):
        self.sample = sample

    def __call__(self):
        logging.debug('Calling Primitive')
        with ClingoSolver() as ctl:
            ctl.load([Logic.grammarFile, self.sample])
            ctl.ground([Logic.base, Concept.primitive(1), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Negation:
    def __init__(self, sample, primitive: ConceptFile):
        self.sample = sample
        self.primitive = primitive
    def __call__(self):
        logging.debug('Calling Negation({})'.format(self.primitive.name))
        with ClingoSolver() as ctl:
            ctl.load([Logic.grammarFile, self.primitive.file, self.sample])
            ctl.ground([Logic.base, Concept.negation(2), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]

        return result

class EqualRole:
    def __init__(self, sample, roles):
        self.sample = sample
        self.roles = roles

    def __call__(self):
        logging.debug('Calling EqualRole')
        with ClingoSolver() as ctl:
            ctl.load([Logic.grammarFile, self.roles, self.sample])
            ctl.ground([Logic.base, Concept.equalRole(3), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Conjunction:
    def __init__(self, sample, concept1: ConceptFile, concept2: ConceptFile):
        self.sample = sample
        self.concept1 = concept1
        self.concept2 = concept2

    def __call__(self):
        logging.debug('Calling Conjunction({},{})'.format(self.concept1.name, self.concept2.name))
        with ClingoSolver() as ctl:
            ctl.load([Logic.grammarFile, self.concept1.file, self.concept2.file, self.sample])
            depth = self.concept1.depth + self.concept2.depth + 1
            ctl.ground([Logic.base, Concept.conjunction(depth, self.concept1.depth, self.concept2.depth), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Uni:
    def __init__(self, sample, concept: ConceptFile, roles):
        self.sample = sample
        self.concept = concept
        self.roles = roles

    def __call__(self):
        logging.debug('Calling Uni({})'.format(self.concept.name))
        with ClingoSolver() as ctl:
            ctl.load([Logic.grammarFile, self.concept.file, self.roles, self.sample])
            depth = self.concept.depth + 2
            ctl.ground([Logic.base, Concept.uni(depth), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

class Exi:
    def __init__(self, sample, concept: ConceptFile, roles):
        self.sample = sample
        self.concept = concept
        self.roles = roles
    def __call__(self):
        logging.debug('Calling Exi({})'.format(self.concept.name))
        with ClingoSolver() as ctl:
            ctl.load([Logic.grammarFile, self.concept.file, self.roles, self.sample])
            depth = self.concept.depth + 2
            ctl.ground([Logic.base, Concept.exi(depth), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

def roles(sample):
    with ClingoSolver() as ctl:
        ctl.load([sample, Logic.grammarFile])
        ctl.ground([Logic.base, Logic.roles, Logic.keepRoles])
        result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
    return result

def const_state(sample):
    with ClingoSolver() as ctl:
        ctl.load([sample, Logic.grammarFile])
        ctl.ground([Logic.base, Logic.simplifySample])
        result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
    return result


class Grammar:
    def __init__(self, sample, path, comp_type='standard'):
        self.samplePath = sample
        self.path = Path(path)
        self.concepts = {}
        self.conceptNum = {}
        self.compare = Comparison(comp_type=comp_type)
        self.total_concepts = 0
    
    def createDir(self):
        print(self.path.absolute())
        if not self.path.is_dir():
            try:
                self.path.mkdir()
            except (FileNotFoundError, FileExistsError) as e:
                print(repr(e))
                sys.exit()


    
    def loadProgress(self, depth):
        if depth < 1: return
        directory = os.listdir(self.path)
        self.roles = str(self.path/'roles.lp')
        self.simple = str(self.path/'simple.lp')
        for elem in directory:
            match = re.match(r'depth_(\d*)_(\d*)', elem)
            if match:
                dep = int(match.group(1))
                if dep <= depth:
                    logging.debug('Found file: {}'.format(match.group(0)))
                    if dep not in self.concepts:
                        self.concepts[dep] = []
                        self.conceptNum[dep] = 0
                    filename =  str(self.path/elem)
                    self.conceptNum[dep] += countFile(filename)
                    self.concepts[dep].append(ConceptFile(match.group(0), filename, dep))
        for depth in self.concepts:
            self.total_concepts += self.conceptNum[depth]
            self.concepts[depth].sort(key=lambda x: x.name)
    
    def addRoles(self):
        symbols = roles(self.samplePath)
        with ClingoSolver() as ctl:
            ctl.load(Logic.pruneFile)
            ctl.addSymbols(symbols)
            ctl.ground([Logic.base, Logic.index_role(0)])
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        self.roles = str(self.path/'roles.lp')
        write_symbols(symbols, self.roles)
        del symbols[:]

    def addConstAndState(self):
        symbols = const_state(self.samplePath)
        self.simple = str(self.path/'simple.lp')
        write_symbols(symbols, self.simple)
        del symbols[:]

    def getDepth(self, depth):
        if depth == 1:
            return [Primitive(self.samplePath)]
        elif depth == 2:
            return [Negation(self.simple, prim) for prim in self.concepts[1]]
        else:
            variables = []
            if depth == 3:
                variables.append(EqualRole(self.simple, self.roles))
            
            #for i in range(1, (depth + 1)//2):
            #    for conc1 in self.concepts[i]:
            #        for conc2 in self.concepts[depth - i - 1]:
            #            variables.append(Conjunction(self.simple, conc1, conc2))
            for conc in self.concepts[depth - 2]:
                variables.append(Uni(self.simple, conc, self.roles))
                variables.append(Exi(self.simple, conc, self.roles))
            return variables

    def conceptIterator(self):
        depths = list(self.concepts.keys())
        depths.sort()
        for d in depths:
            for conc in self.concepts[d]:
                yield conc

    def difference(self, expression_set: List[clingo.Symbol]) -> List[clingo.Symbol]:
        for conc in self.conceptIterator():
            logging.debug('Prune with {}'.format(conc.name))
            expression_set = prune_symbols(
                expression_set,
                Logic.pruneFile,
                Concept.compareExpConc,
                self.compare,
                Concept.pruneExp,
                files=[conc.file])
        return expression_set
                        
    def addConcepts(self, depth, symbols, max_size=50):
        startSize = (max_size - (self.conceptNum[depth] % max_size)) % max_size
        logging.debug('Start size {}'.format(startSize))
        concept_n, groups = splitSymbols(
            symbols,
            self.total_concepts,
            startSize,
            Concept.numberConc,
            Concept.classify,
            max_conc=max_size)
        
        logging.info('{} concepts.'.format(concept_n))
        for i, group in enumerate(groups):
            if i == 0:
                if startSize > 0:
                    write_symbols(group, self.concepts[depth][-1].file, type_='a')
                    logging.debug('Appending to existing file')
            else:
                name = 'depth_{}_{}'.format(depth, len(self.concepts[depth]))
                newset = ConceptFile(name, str(self.path/'{}.lp'.format(name)), depth)
                self.concepts[depth].append(newset)
                write_symbols(group, newset.file)
                logging.debug('Created new file {}'.format(newset.file))
                del group[:]
        self.conceptNum[depth] += concept_n
        self.total_concepts += concept_n

    def expandGrammar(self, start_depth, max_depth, logg=False, max_conc=50, type_='clingo'):
        logging.info("Starting {}. Ending {}".format(start_depth, max_depth))
        
        if start_depth <= 1:
            self.createDir()
            self.addRoles()
            self.addConstAndState()
        #else:
        #    self.loadProgress(start_depth-1)
        
        for depth in range(start_depth, max_depth+1):
            if logg: print('Depth {}:'.format(depth))
            self.concepts[depth] = []
            self.conceptNum[depth] = 0
            expressions = self.getDepth(depth)
            logging.debug('Number of concept groups: {}'.format(len(expressions)))
            for exp in expressions:
                symbols = exp()
                symbols = prune_symbols(
                    symbols,
                    Logic.pruneFile,
                    Concept.compareExp,
                    self.compare,
                    Concept.pruneExp
                )
                symbols = self.difference(symbols)

                self.addConcepts(depth, symbols, max_size=max_conc)
                del symbols[:]
                    
            print("Total {}: {} concepts.\n".format(depth, self.conceptNum[depth]))
        self.joinOutput(self.path/'concept_summary.lp')

    def joinOutput(self, out_file):
        with open(str(out_file), 'w') as outfile:
            for conc in self.conceptIterator():
                with open(conc.file) as infile:
                    outfile.write(infile.read())
    '''
    def getConceptBatches(self, batch_size):
        result = []
        count = 0
        for dep in self.depth:
            for conc in self.depth[dep]:
                if count == 0: result.append(list())
                result[-1].append(conc)
                count = (count + 1) % batch_size
        return result


    def generateFeatures(self, concept_batch=1, feature_max=50):
        self.transitions = GrammarKnowledge('transition', self.path/'transition.lp', Logic.base, [self.sample])
        self.transitions.generate() #TODO any generate will fail if the file loaded has #show. and no other #show statements
        self.transitions.filter(Logic.pruneFile, Logic.transitions)
        self.transitions.write()

        self.features = []
        self.feature_n = 0
        feature_compare = Comparison(Logic.logicPath, comp_type='feature')
        print('#### Starting Feature Generation ####')
        name = 'features_{}'.format(self.feature_n)
        feature = FeatureSet(name, '{}.lp'.format(name), Logic.primitiveFeature, [self.transitions, self.sample])
        feature.generate(feature_compare)
        feature.toFinal()
        symbols = feature.getSymbols()
        logging.debug('Generating features from null predicates')

        self.feature_n = self.addResults(
                'features', self.feature_n, self.features, symbols,
                Logic.enumerateFeat,Logic.classifyFeat, max_size=feature_max)
        feature.clean()
        
        batches = self.getConceptBatches(concept_batch)
        for batch in batches:
            logging.debug('Generating features from {}'.format([ks.name for ks in batch]))
            name = 'features_{}'.format(self.feature_n)
            feature = FeatureSet(name, '{}.lp'.format(name), Logic.conceptFeature, [self.transitions]+batch)
            feature.generate(feature_compare)
            for feat in self.features:
                logging.debug('Prunning with {}'.format(feat.name))
                feature.removeRedundant(feat,feature_compare)
            feature.toFinal()
            symbols = feature.getSymbols()
            self.feature_n = self.addResults(
                    'features', self.feature_n, self.features, symbols,
                    Logic.enumerateFeat,Logic.classifyFeat, max_size=feature_max)
            feature.clean()
        print('Features Generated: {}'.format(self.feature_n))
    
    def addCardinality(self):
        for dep in self.depth:
            for conc in self.depth[dep]:
                logging.debug('Generating cardinality for {}'.format(conc.name))
                ctl = clingo.Control()
                conc.load(ctl)
                self.simpleSample.load(ctl)
                ctl.load(Logic.pruneFile)
                ctl.ground([Logic.base, ('cardinality', [])])
                symbols = []
                with ctl.solve(yield_=True) as models:
                    models = to_model_list(models)
                    symbols = models[0].symbols(shown=True)
                ModelUtil(symbols).write(conc.file)'''


    @staticmethod
    def countSymbol(symbols, name):
        count = 0
        for symbol in symbols:
            if symbol.name == name:
                count += 1

        return count

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
    parser.add_argument('out_dir', type=str, help='Output folder')
    parser.add_argument('max_depth', type=int, help='Maximum concept depth')
    parser.add_argument('-s', '--start',action='store',default=1, type=int, help='Starting depth')
    parser.add_argument('-c', '--conc',action='store',default=50, type=int, help='Max number of concepts in file')
    parser.add_argument('-f', '--feat',action='store',default=50, type=int, help='Max number of features in file')
    parser.add_argument('--fast', action='store_true', help='Prunning with cardinality')
    parser.add_argument('--std', action='store_true', help='Standard sound prunning')
    parser.add_argument('--mix', action='store_true', help='Cardinality + standard prunning')
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--batch',action='store',default=1, type=int, help='Concept files used simultaneaously in feature generation.')
    args = parser.parse_args()
    

    logging.basicConfig(level=args.loglevel)
    if sum((int(b) for b in (args.fast, args.std, args.mix))) > 1:
        RuntimeError('More than one prunning type specified')

    comp_type = 'fast' if args.fast else None
    comp_type = 'standard' if args.std else comp_type
    comp_type = 'mixed' if args.mix else comp_type
    comp_type = 'standard' if comp_type is None else comp_type    
    print(comp_type)
    grammar = Grammar(args.sample,args.out_dir, comp_type=comp_type)
    import time
    start = time.time()
    grammar.loadProgress(args.start-1)
    #grammar.addCardinality()
    grammar.expandGrammar(args.start, args.max_depth, logg=True, max_conc=args.conc)
    #grammar.generateFeatures(concept_batch=args.batch, feature_max=args.feat)
    print("Took {}s.".format(round(time.time()-start, 2)))
