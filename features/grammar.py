
import clingo
import sys
import logging
import pathlib
import re
import argparse
import os
from pathlib import Path
from model_util import to_model_list, ModelUtil, filter_symbols, check_multiple, add_symbols, write_symbols, count_symbols, get_symbols
from features.logic import Logic
from knowledge import ConceptFile, splitSymbols
import features.solver as solver
from features.solver import SolverType
from comparison import CompareConcept
from typing import List, Tuple
from prune import Pruneable

class Concept(Pruneable):
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

class Primitive:
    def __init__(self, sample):
        self.sample = sample

    def __call__(self):
        logging.debug('Calling Primitive')
        with solver.create_solver() as ctl:
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
        with solver.create_solver() as ctl:
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
        with solver.create_solver() as ctl:
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
        with solver.create_solver() as ctl:
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
        with solver.create_solver() as ctl:
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
        with solver.create_solver() as ctl:
            ctl.load([Logic.grammarFile, self.concept.file, self.roles, self.sample])
            depth = self.concept.depth + 2
            ctl.ground([Logic.base, Concept.exi(depth), Concept.keepExp])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result

def roles(sample):
    with solver.create_solver() as ctl:
        ctl.load([sample, Logic.grammarFile])
        ctl.ground([Logic.base, Logic.roles, Logic.keepRoles])
        result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
    return result

def const_state(sample):
    with solver.create_solver() as ctl:
        ctl.load([sample, Logic.pruneFile])
        ctl.ground([Logic.base, Logic.simplifySample])
        result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
    return result


class Grammar:
    def __init__(self, sample, path, comp_type=CompareConcept.STANDARD):
        self.samplePath = sample
        self.path = Path(path)
        self.concepts = {}
        self.conceptNum = {}
        self.compare = CompareConcept(comp_type=comp_type)
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
        with solver.create_solver() as ctl:
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
            
            for i in range(1, (depth + 1)//2):
                for conc1 in self.concepts[i]:
                    for conc2 in self.concepts[depth - i - 1]:
                        variables.append(Conjunction(self.simple, conc1, conc2))
            for conc in self.concepts[depth - 2]:
                variables.append(Uni(self.simple, conc, self.roles))
                variables.append(Exi(self.simple, conc, self.roles))
            return variables

    def get_concepts(self):
        depths = list(self.concepts.keys())
        depths.sort()
        concepts = []
        for d in depths:
            for conc in self.concepts[d]:
                concepts.append(conc)
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

    def prune(self, expressions, max_exp, max_conc):
        concept_files = [conc.file for conc in self.get_concepts()]
        return Concept.prune_symbols(
                    expressions, concept_files, self.compare,
                    max_atoms=max_exp, max_comp=max_conc
                )

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

    def expandGrammar(self, start_depth, max_depth, logg=False, max_exp=50, max_conc=50):
        logging.info("Starting {}. Ending {}".format(start_depth, max_depth))
        
        if start_depth <= 1:
            self.createDir()
            self.addRoles()
            self.addConstAndState()
        else:
            self.loadProgress(start_depth-1)
        
        for depth in range(start_depth, max_depth+1):
            if logg: print('Depth {}:'.format(depth))
            self.concepts[depth] = []
            self.conceptNum[depth] = 0
            expressions = self.getDepth(depth)
            logging.debug('Number of concept groups: {}'.format(len(expressions)))
            for exp in expressions:
                symbols = exp()
                logging.debug('Expressions {}'.format(count_symbols(symbols, 'exp', 2)))
                symbols = self.prune(symbols, max_exp=max_exp, max_conc=max_conc)
                self.addConcepts(depth, symbols, max_size=max_conc)
                del symbols[:]
                    
            print("Total {}: {} concepts.\n".format(depth, self.conceptNum[depth]))
        self.joinOutput(self.path/'concept_summary.lp')

    def joinOutput(self, out_file):
        with open(str(out_file), 'w') as outfile:
            for conc in self.conceptIterator():
                with open(conc.file) as infile:
                    outfile.write(infile.read())

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
    parser.add_argument('-s', '--start',action='store',default=1, type=int, help='Starting depth')
    parser.add_argument('--exp',default=50, type=int, help='Max number of expressions in prunning set')
    parser.add_argument('--conc',default=50, type=int, help='Max number of concepts in file')
    
    group_compare = parser.add_mutually_exclusive_group(required = False)
    group_compare.add_argument('--fast', action='store_const', dest='compare', help='Prunning with cardinality',
        const=CompareConcept.FAST, default=CompareConcept.STANDARD)
    group_compare.add_argument('--std', action='store_const', dest='compare', help='Standard sound prunning',
        const=CompareConcept.STANDARD)
    group_compare.add_argument('--mix', action='store_const', dest='compare', help='Cardinality + standard prunning',
        const=CompareConcept.MIXED)
    
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('--proc',help="Runs clingo solver in separate process",
        action="store_const", dest="solver", const=SolverType.PROCESS, default=SolverType.SIMPLE)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    print(args.compare)
    solver.set_default(args.solver)
    print(args.solver)
    grammar = Grammar(args.sample,args.out_dir, comp_type=args.compare)
    import time
    start = time.time()
    print(Logic.logicPath, Logic.grammarFile)
    grammar.expandGrammar(args.start, args.max_depth, logg=True, max_exp=args.exp, max_conc=args.conc)
    print("Took {}s.".format(round(time.time()-start, 2)))
