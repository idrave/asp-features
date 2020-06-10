
import clingo
import sys
import logging
import pathlib
import re
import argparse
import os
from pathlib import Path
from model_util import check_multiple
from model_util import to_model_list, ModelUtil, filter_symbols, check_multiple, add_symbols
from features.logic import Logic
from features.knowledge import GrammarKnowledge, KnowledgeSet, ConceptSet, ClingoFile, PickleFile, FileCreator

class Comparison:
    def __init__(self, path, comp_type='standard'):
        self.file = path/'differ.lp'
        types = {
            'standard' : self.standardCompare,
            'fast' : self.fastCompare,
            'mixed' : self.mixedCompare
        }
        self.compare = types.get(comp_type, None)
        if self.compare is None:
            raise RuntimeError("Invalid comparison type.")

    def standardCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('standard_differ', [])])
        ctl.solve()

    def fastCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('fast_differ', [])])
        ctl.solve()
    
    def mixedCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('optimal_differ_start', [])])
        ctl.solve()
        ctl.ground([('optimal_differ_end', [])])
        ctl.solve()


class Grammar:
    def __init__(self, sample, path, comp_type='standard'):
        self.samplePath = Path(sample)
        self.path = Path(path)
        self.depth = {}
        self.conceptNum = {}
        self.concepts = []
        self.sample = KnowledgeSet('sample', self.samplePath)
        self.compare = Comparison(Logic.logicPath, comp_type=comp_type)
    
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
        self.roles = KnowledgeSet('roles', self.path/'roles.lp')
        self.simpleSample = KnowledgeSet('simple_sample', self.path/'simple_sample.lp')
        for elem in directory:
            match = re.match(r'depth_(\d*)_(\d*)', elem)
            if match:
                dep = int(match.group(1))
                if dep <= depth:
                    logging.debug('Found file: {}'.format(match.group(0)))
                    if dep not in self.depth:
                        self.depth[dep] = []
                    kn_file =  self.path/elem
                    self.depth[dep].append(KnowledgeSet(match.group(0), kn_file))

    def getDepth(self, depth, simple=False, logg=False, type_='clingo'):
        concept_list = []
        if depth == 1:
            primitive_file =  self.path/'primitive.lp'
            concept = ConceptSet('primitive', primitive_file, Logic.primitive(depth),[self.sample])
            concept_list.append(concept)
            return concept_list

        if depth == 2:
            negation_file =  self.path/'negation.lp'
            concept = ConceptSet(
                'negation', negation_file, Logic.negation(depth),
                [self.simpleSample] + self.depth[1]
            )
            concept_list.append(concept)
            return concept_list

        if depth == 3:
            equalrole_file =  self.path/'equal_role.lp'
            concept = ConceptSet(
                'equal_role', equalrole_file, Logic.equalRole(depth),
                [self.simpleSample, self.roles]
            )
            concept_list.append(concept)

        for i in range(1,(depth+1)//2):
            for concept1 in self.depth[i]:
                for concept2 in self.depth[depth-i-1]:
                    name = 'conjunction_{}_{}_{}'.format(depth, concept1.name, concept2.name)
                    conjunction_file =  self.path/'{}.lp'.format(name)
                    concept = ConceptSet(
                        name, conjunction_file, Logic.conjunction(depth, i, depth-i-1),
                        [self.simpleSample, concept1, concept2]
                    )
                    concept_list.append(concept)

        for conc in self.depth[depth-2]:
            name = 'exi_{}_{}'.format(depth, conc.name)
            exi_file =  self.path/'{}.lp'.format(name)
            concept = ConceptSet(
                name, exi_file, Logic.exi(depth),
                [self.simpleSample, self.roles, conc]
            )
            concept_list.append(concept)
            name = 'uni_{}_{}'.format(depth, conc.name)
            uni_file =  self.path/'{}.lp'.format(name)
            concept = ConceptSet(
                name, uni_file, Logic.uni(depth),
                [self.simpleSample, self.roles, conc]
            )
            concept_list.append(concept)
        return concept_list

    def addResults(self, depth, symbols, max_conc=50):
        ctl = clingo.Control(['-n 0'])
        add_symbols(ctl, symbols)
        ctl.load(Logic.pruneFile)
        startSize = (max_conc - (self.conceptNum[depth] % max_conc)) % max_conc
        logging.debug('Start size {}'.format(startSize))
        
        ctl.ground([Logic.base, Logic.enumerate(startSize, max_conc)])
        group_n = None

        with ctl.solve(yield_=True) as models:
            models = to_model_list(models)
            check_multiple(models)
            for model in models:
                symbols = model.symbols(atoms=True)
                group_n = model.symbols(shown=True)[0].arguments[0].number
                print('SHOWN', model.symbols(shown=True))
        del(ctl)

        ctl = clingo.Control(['-n 0'])
        add_symbols(ctl, symbols)
        ctl.load(Logic.pruneFile)
        ctl.ground([Logic.base, Logic.classify])
        concept_n = None
        for i in range(group_n+1):
            ctl.assign_external(clingo.Function('model', [clingo.Number(i)]), False)
        for i in range(group_n+1):
            ctl.assign_external(clingo.Function('model', [clingo.Number(i)]), True)
            logging.debug('{}th group'.format(i))
            with ctl.solve(yield_=True) as models:
                models = to_model_list(models)
                check_multiple(models)
                model = models[0]
                symbols = model.symbols(shown=True)
                
                if i == 0:
                    concept_n = symbols[0].arguments[0].number
                    logging.info('{} concepts.'.format(concept_n))
                elif i == 1:
                    if startSize > 0:
                        ModelUtil(symbols).write(self.depth[depth][-1].file, type_='a')
                        logging.debug('Appending to existing file')
                else:
                    name = 'depth_{}_{}'.format(depth, len(self.depth[depth]))
                    newset = KnowledgeSet(name, self.path/'{}.lp'.format(name))
                    self.depth[depth].append(newset)
                    ModelUtil(symbols).write(newset.file)
                    logging.debug('Created new file {}'.format(newset.file))
                ctl.assign_external(clingo.Function('model', [clingo.Number(i)]), False)
        self.conceptNum[depth] += concept_n

    def expandGrammar(self, start_depth, max_depth, logg=False, max_conc=50, type_='clingo'):
        logging.info("Starting {}. Ending {}".format(start_depth, max_depth))
        
        if start_depth <= 1:
            self.createDir()
            self.roles = GrammarKnowledge('roles',  self.path/'roles.lp', Logic.roles, [self.sample])
            self.simpleSample = GrammarKnowledge('simple_sample',  self.path/'simple_sample.lp', Logic.base, [self.sample])
            self.simpleSample.generate()
            self.simpleSample.filter(Logic.simplifySample)
            self.simpleSample.write()
            self.roles.generate()
            self.roles.filter(Logic.keepRoles)
            self.roles.write()
        else:
            self.loadProgress(start_depth-1)
        
        for depth in range(start_depth, max_depth+1):
            if logg: print('Depth {}:'.format(depth))
            self.depth[depth] = []
            self.conceptNum[depth] = 0
            expressions = self.getDepth(depth, logg=logg)

            print([exp.name for exp in expressions])
            for exp in expressions:
                logging.debug("Generating expression {}".format(exp.name))
                exp.generate(self.compare)
                #logging.debug('Expression # {}'.format(Grammar.countSymbol(exp.getSymbols(), 'exp')))
                for depth in self.depth:
                    for concept in self.depth[depth]:
                        logging.debug('Prunning with {}'.format(concept.name))
                        exp.removeRedundant(concept, comparison=self.compare)
                        #logging.debug('Expression # {}'.format(Grammar.countSymbol(exp.getSymbols(), 'exp')))
                exp.toConcept()
                symbols = exp.getSymbols()
                self.addResults(depth, symbols, max_conc=max_conc)
                exp.clean()
            
            if logg:
                count = 0
                for conc in self.depth[depth]:
                    aux = countFile(conc.file)
                    count += aux
                print("Total {}: {} concepts.\n".format(depth, count))

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
    parser.add_argument('--fast', action='store_true', help='Prunning with cardinality')
    parser.add_argument('--std', action='store_true', help='Standard sound prunning')
    parser.add_argument('--mix', action='store_true', help='Cardinality + standard prunning')
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
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
    grammar.expandGrammar(args.start, args.max_depth, logg=True)
    print("Took {}s.".format(round(time.time()-start, 2)))
