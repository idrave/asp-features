
import clingo
import sys
import logging
import pathlib
import re
import argparse
import os
from pathlib import Path
from model_util import check_multiple
from model_util import to_model_list, ModelUtil, filter_symbols, check_multiple, add_symbols, splitSymbols
from features.logic import Logic
from features.knowledge import GrammarKnowledge, KnowledgeSet, ConceptSet, FeatureSet

class Comparison:
    def __init__(self, path, comp_type='standard'):
        self.file = path/'differ.lp'
        types = {
            'standard' : self.standardCompare,
            'fast' : self.fastCompare,
            'mixed' : self.mixedCompare,
            'feature' : self.featureCompare
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

    def featureCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('feature_differ', [])])
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
            concept = ConceptSet('primitive', self.path, Logic.primitive(depth),[self.sample])
            concept_list.append(concept)
            return concept_list

        if depth == 2:
            concept = ConceptSet(
                'negation', self.path, Logic.negation(depth),
                [self.simpleSample] + self.depth[1]
            )
            concept_list.append(concept)
            return concept_list

        if depth == 3:
            concept = ConceptSet(
                'equal_role', self.path, Logic.equalRole(depth),
                [self.simpleSample, self.roles]
            )
            concept_list.append(concept)

        for i in range(1,(depth+1)//2):
            for concept1 in self.depth[i]:
                for concept2 in self.depth[depth-i-1]:
                    name = 'conjunction_{}_{}_{}'.format(depth, concept1.name, concept2.name)
                    concept = ConceptSet(
                        name, self.path, Logic.conjunction(depth, i, depth-i-1),
                        [self.simpleSample, concept1, concept2]
                    )
                    concept_list.append(concept)

        for conc in self.depth[depth-2]:
            name = 'exi_{}_{}'.format(depth, conc.name)
            concept = ConceptSet(
                name, self.path, Logic.exi(depth),
                [self.simpleSample, self.roles, conc]
            )
            concept_list.append(concept)
            name = 'uni_{}_{}'.format(depth, conc.name)
            concept = ConceptSet(
                name, self.path, Logic.uni(depth),
                [self.simpleSample, self.roles, conc]
            )
            concept_list.append(concept)
        return concept_list

    def addResults(self, name, count, list_res, symbols, enumer, classify, max_size=50):
        startSize = (max_size - (count % max_size)) % max_size
        logging.debug('Start size {}'.format(startSize))
        result_n, groups = splitSymbols(symbols, startSize, enumer, classify, max_conc=max_size)
        
        logging.info('{} concepts.'.format(result_n))
        for i, group in enumerate(groups):
            if i == 0:
                if startSize > 0:
                    ModelUtil(group).write(list_res[-1].file, type_='a')
                    logging.debug('Appending to existing file')
            else:
                name = '{}_{}'.format(name, len(list_res))
                newset = KnowledgeSet(name, self.path/'{}.lp'.format(name))
                list_res.append(newset)
                ModelUtil(group).write(newset.file)
                logging.debug('Created new file {}'.format(newset.file))
        return count + result_n

    def expandGrammar(self, start_depth, max_depth, logg=False, max_conc=50, type_='clingo'):
        logging.info("Starting {}. Ending {}".format(start_depth, max_depth))
        
        if start_depth <= 1:
            self.createDir()
            self.roles = GrammarKnowledge('roles',  self.path/'roles.lp', Logic.roles, [self.sample])
            self.simpleSample = GrammarKnowledge('simple_sample',  self.path/'simple_sample.lp', Logic.base, [self.sample])
            self.simpleSample.generate()
            self.simpleSample.filter(Logic.pruneFile, Logic.simplifySample)
            self.simpleSample.write()
            self.roles.generate()
            self.roles.filter(Logic.pruneFile, Logic.keepRoles)
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
                exp.toFinal()
                symbols = exp.getSymbols()
                self.conceptNum[depth] = self.addResults(
                    'depth_{}'.format(depth), self.conceptNum[depth], self.depth[depth], symbols,
                    Logic.enumerate,Logic.classify, max_size=max_conc)
                exp.clean()
                #exp.pop()
                    
            print("Total {}: {} concepts.\n".format(depth, self.conceptNum[depth]))
        self.joinOutput(self.path/'concept_summary.lp')

    def joinOutput(self, out_file):
        with open(str(out_file), 'w') as outfile:
            for dep in self.depth:
                for conc in self.depth[dep]:
                    with open(conc.file) as infile:
                        outfile.write(infile.read())

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
                ModelUtil(symbols).write(conc.file)


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
    grammar.loadProgress(args.start)
    #grammar.addCardinality()
    #grammar.expandGrammar(args.start, args.max_depth, logg=True, max_conc=args.conc)
    grammar.generateFeatures(concept_batch=args.batch, feature_max=args.feat)
    print("Took {}s.".format(round(time.time()-start, 2)))
