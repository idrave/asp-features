
import clingo
import sys
import logging
import pathlib
import re
import argparse
from pathlib import Path
from model_util import check_multiple
from model_util import to_model_list, ModelUtil, filter_symbols, check_multiple, add_symbols

class Logic:
    #logicPath = Path(__file__).parent.absolute()
    logicPath = Path('/home/ivan/Documents/ai/features/features')
    grammarFile = str(logicPath/'concept.lp')
    base = ('base', [])
    @staticmethod
    def primitive(depth):
        return ('primitive', [depth])
    
    roles = ('roles', [])

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
    def enumerate(start, gsize):
        return ('enumerate', [start, gsize])

    classify = ('classify', [])
    pruneFile = str(logicPath/'prune.lp')
    keepExp = ('keep_exp', []) 
    compareExp = ('compare_exp', []) 
    pruneExp = ('prune_exp', [])
    compareExpConc = ('compare_exp_conc', []) 
    toConcept = ('exp2conc', [])
    keepRoles = ('get_roles', [])
    simplifySample = ('simplify', [])
    divide = ('divide', [])

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

class KnowledgeSet:
    def __init__(self, name, file):
        self.name = name
        self.file = str(file)

    def load(self, ctl):
        ctl.load(self.file)

class GrammarKnowledge(KnowledgeSet):
    def __init__(self, name, file, program, requirements):
        super(GrammarKnowledge, self).__init__(name, file)
        
        self.program = program
        self.requirements = requirements
        self.isGenerated = False
        self.isStored = False
        self._symbols = None

    def _solveControl(self, ctl):
        with ctl.solve(yield_=True) as models:
            models = list(models)
            check_multiple(models)
            self._symbols = models[0].symbols(shown=True)

    def _solveProgram(self):
        if self.isGenerated:
            raise RuntimeError("Set {} already generated.".format(self.name))
        ctl = clingo.Control()
        for req in self.requirements:
            req.load(ctl)
        
        ctl.load(Logic.grammarFile)
        ctl.ground([('base', []), self.program])
        self._solveControl(ctl)
        del(ctl)

    def generate(self):          
        self._solveProgram()
        self.isGenerated = True
        
    def filter(self, prune):
        if not self.isGenerated:
            raise RuntimeError("Set {} has not been generated. Cannot filter".format(self.name))
        if self.isStored:
            raise RuntimeError("Set {} has been stored in a file. Cannot filter".format(self.name))
        ctl = clingo.Control()
        ctl.load(Logic.pruneFile)
        ctl.add('base', [], str(ModelUtil(self._symbols)))
        ctl.ground([('base', []), prune])
        #self._symbols = filter_symbols(ctl, single=True)
        self._solveControl(ctl)

        del(ctl)

    def write(self):
        if not self.isGenerated:
            raise RuntimeError("Set {} has not been generated. Cannot write to file".format(self.name))
        if self.isStored:
            raise RuntimeError("Set {} has been stored in a file. Cannot overwrite".format(self.name))
        with open(self.file, 'w') as file:
            file.write(str(ModelUtil(self._symbols)))
        self.isStored = True
        del(self._symbols)


class ConceptSet(GrammarKnowledge):
    def __init__(self, name, file, program, requirements):
        super(ConceptSet, self).__init__(name, file, program, requirements)
        self.isConcept = False

    def generate(self, comparison: Comparison):
        super(ConceptSet, self)._solveProgram()
        ctl = clingo.Control()
        ctl.add('base', [], str(ModelUtil(self._symbols)))
        ctl.load(Logic.pruneFile)
        ctl.ground([('base', []), Logic.compareExp])
        comparison.compare(ctl)
        ctl.ground([Logic.pruneExp])
        #self._symbols = filter_symbols(ctl, single=True)
        self._solveControl(ctl)
        del(ctl)
        self.isGenerated = True

    def removeRedundant(self, concepts, comparison: Comparison):
        ctl = clingo.Control()
        ctl.add('base', [], str(ModelUtil(self._symbols)))
        ctl.load(Logic.pruneFile)
        concepts.load(ctl)
        ctl.ground([('base', []), Logic.compareExpConc])
        comparison.compare(ctl)
        ctl.ground([Logic.pruneExp])
        #self._symbols = filter_symbols(ctl, single=True)
        self._solveControl(ctl)
        del(ctl)

    def toConcept(self):
        if not self.isGenerated:
            raise RuntimeError("Set {} has not been generated. Cannot write to file".format(self.name))
        if self.isStored:
            raise RuntimeError("Set {} has been stored in a file. Cannot overwrite".format(self.name))
        ctl = clingo.Control()
        ctl.load(Logic.pruneFile)
        ctl.add('base', [], str(ModelUtil(self._symbols)))
        ctl.ground([('base', []), Logic.toConcept])
        #self._symbols = filter_symbols(ctl, single=True)
        self._solveControl(ctl)
        del(ctl)

    def getSymbols(self):
        return self._symbols

    def clean(self):
        del(self._symbols)

    def write(self):
        raise RuntimeError("OHNO")


class Grammar:
    def __init__(self, sample, path, comp_type='standard'):
        self.samplePath = Path(sample)
        self.path = Path(path)
        self.sample = KnowledgeSet('sample', self.samplePath)
        self.roles = GrammarKnowledge('roles', self.path/'roles.lp', Logic.roles, [self.sample])
        self.simpleSample = GrammarKnowledge('simple_sample', self.path/'simple_sample.lp', Logic.base, [self.sample])
        self.depth = {}
        self.conceptNum = {}
        self.concepts = []
        
        self.compare = Comparison(Logic.logicPath, comp_type=comp_type)
    
    def createDir(self):
        print(self.path.absolute())
        if not self.path.is_dir():
            try:
                self.path.mkdir()
            except (FileNotFoundError, FileExistsError) as e:
                print(repr(e))
                sys.exit()

    def getDepth(self, depth, simple=False, logg=False):
        concept_list = []
        if depth == 1:
            primitive_file = self.path/'primitive.lp'
            print(str(primitive_file))
            concept = ConceptSet('primitive', primitive_file, Logic.primitive(depth),[self.sample])
            concept_list.append(concept)
            return concept_list

        if depth == 2:
            negation_file = self.path/'negation.lp'
            concept = ConceptSet(
                'negation', negation_file, Logic.negation(depth),
                [self.simpleSample] + self.depth[1]
            )
            concept_list.append(concept)
            return concept_list

        if depth == 3:
            equalrole_file = self.path/'equal_role.lp'
            concept = ConceptSet(
                'equal_role', equalrole_file, Logic.equalRole(depth),
                [self.simpleSample, self.roles]
            )
            concept_list.append(concept)

        for i in range(1,(depth+1)//2):
            if simple:
                name = 'conjunction_{}_{}'.format(depth, i)
                conjunction_file = self.path/'{}.lp'.format(name)
                concept = ConceptSet(
                    name, conjunction_file, Logic.conjunction(depth, i, depth-i-1),
                    [self.simpleSample] + self.depth[i] + self.depth[depth-i-1]
                )
                concept_list.append(concept)
            else:
                for concept1 in self.depth[i]:
                    for concept2 in self.depth[depth-i-1]:
                        name = 'conjunction_{}_{}_{}'.format(depth, concept1.name, concept2.name)
                        conjunction_file = self.path/'{}.lp'.format(name)
                        concept = ConceptSet(
                            name, conjunction_file, Logic.conjunction(depth, i, depth-i-1),
                            [self.simpleSample, concept1, concept2]
                        )
                        concept_list.append(concept)

        if simple:
            name = 'exi_{}'.format(depth)
            exi_file = self.path/'{}.lp'.format(name)
            concept = ConceptSet(
                name, exi_file, Logic.exi(depth),
                [self.simpleSample, self.roles] + self.depth[depth-2]
            )
            concept_list.append(concept)
            name = 'uni_{}'.format(depth)
            uni_file = self.path/'{}.lp'.format(name)
            concept = ConceptSet(
                name, uni_file, Logic.uni(depth),
                [self.simpleSample, self.roles] + self.depth[depth-2]
            )
            concept_list.append(concept)
        else:
            for conc in self.depth[depth-2]:
                name = 'exi_{}_{}'.format(depth, conc.name)
                exi_file = self.path/'{}.lp'.format(name)
                concept = ConceptSet(
                    name, exi_file, Logic.exi(depth),
                    [self.simpleSample, self.roles, conc]
                )
                concept_list.append(concept)
                name = 'uni_{}_{}'.format(depth, conc.name)
                uni_file = self.path/'{}.lp'.format(name)
                concept = ConceptSet(
                    name, uni_file, Logic.uni(depth),
                    [self.simpleSample, self.roles, conc]
                )
                concept_list.append(concept)
        return concept_list

    def addResults(self, depth, symbols, max_conc=50):
        ctl = clingo.Control(['-n 0'])
        #ctl.add('base', [], str(ModelUtil(symbols)))
        add_symbols(ctl, symbols)
        ctl.load(str(self.path/'aux.lp'))
        ctl.load(Logic.pruneFile)
        startSize = (max_conc - (self.conceptNum[depth] % max_conc)) % max_conc
        logging.debug('Start size {}'.format(startSize))
        
        ctl.ground([Logic.base, Logic.enumerate(startSize, max_conc)])
        group_n = None
        with ctl.solve(yield_=True) as models:
            for model in models:
                symbols = model.symbols(atoms=True)
                group_n = model.symbols(shown=True)[0].arguments[0].number
                print('SHOWN', model.symbols(shown=True))
        del(ctl)
        ctl = clingo.Control(['-n 0'])
        #ctl.add('base', [], str(ModelUtil(symbols)))
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
                    logging.debug('Created new fil {}'.format(newset.file))
                ctl.assign_external(clingo.Function('model', [clingo.Number(i)]), False)
        self.conceptNum[depth] += concept_n

    @staticmethod
    def getNumConcepts(symbols, num):
        concepts = []
        count = 0
        for symbol in symbols:
            if symbol.name == 'conc':
                if count < num:
                    concepts.append(clingo.Function('keep__', [symbol, 0]))
                    count += 1
                else:
                    concepts.append(clingo.Function('keep__', [symbol, 1]))
        ctl = clingo.Control()
        ctl.load(Logic.pruneFile)
        ctl.add('base', [], str(ModelUtil(concepts + symbols)))
        ctl.ground([('base', []), Logic.divide])
        result = Grammar.getSymbolGroups(ctl, 2)
        del(ctl)
        return tuple(result)

    @staticmethod
    def splitConcepts(symbols, group_size):
        concepts = []
        groups = 0
        count = 0
        for symbol in symbols:
            if symbol.name == 'conc':
                concepts.append(clingo.Function('keep__', [symbol, groups]))
                #print(str(clingo.Function('keep__', [symbol, groups])))
                count += 1
                if count == group_size:
                    count = 0
                    groups += 1
        ctl = clingo.Control()
        ctl.load(Logic.pruneFile)
        ctl.add('base', [], str(ModelUtil(concepts + symbols)))
        ctl.ground([('base', []), Logic.divide])
        result = Grammar.getSymbolGroups(ctl, groups+1)
        del(ctl)
        return result

    @staticmethod
    def getSymbolGroups(ctl, groups):
        result = [[] for i in range(groups)]
        with ctl.solve(yield_=True) as models:
            models = list(models)
            check_multiple(models)
            symbols = models[0].symbols(atoms=True)
            #print(len(symbols))
            for symbol in symbols:
                if symbol.name == 'keep__':
                    gid = symbol.arguments[1]
                    #print(gid)
                    result[gid.number].append(symbol.arguments[0])
        return result


    def expandGrammar(self, start_depth, max_depth, logg=False, max_conc=50):
        logging.info("Starting {}. Ending {}".format(start_depth, max_depth))
        if start_depth <= 1:
            self.createDir()
            self.simpleSample.generate()
            self.simpleSample.filter(Logic.simplifySample)
            self.simpleSample.write()
            self.roles.generate()
            self.roles.filter(Logic.keepRoles)
            self.roles.write()
        
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
                ModelUtil(symbols).write(str(self.path/'aux.lp'))
                '''
                concept_n = Grammar.countSymbol(symbols, 'conc')
                print('{}: {} concepts'.format(exp.name, concept_n))
                if residue > 0:
                    filler = None
                    if concept_n <= max_conc-residue:
                        filler = symbols
                        symbols = []
                        residue = (residue + concept_n) % max_conc
                        concept_n = 0
                    else:
                        filler, symbols = Grammar.getNumConcepts(symbols, max_conc-residue)
                        concept_n = concept_n - (max_conc-residue)
                        residue = concept_n % max_conc
                    ModelUtil(filler).write(self.depth[depth][-1].file, type_='a')
                else:
                    residue = concept_n % max_conc

                if concept_n == 0: symbol_groups = []
                elif concept_n <= max_conc: symbol_groups = [symbols]
                else: symbol_groups = Grammar.splitConcepts(symbols, group_size=max_conc)

                for group in symbol_groups:
                    name = 'depth_{}_{}'.format(depth, group_n)
                    newset = KnowledgeSet(name, self.path/'{}.lp'.format(name))
                    self.depth[depth].append(newset)
                    ModelUtil(group).write(newset.file)
                    group_n += 1
                logging.debug('Left residue of {}'.format(residue))'''
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
    #grammar.loadGrammar(args.start, logg=True)
    grammar.expandGrammar(args.start, args.max_depth, logg=True)
    print("Took {}s.".format(round(time.time()-start, 2)))
