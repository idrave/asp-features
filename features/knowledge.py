import clingo
from model_util import add_symbols, ModelUtil, check_multiple
from features.logic import Logic
from pathlib import Path
import pickle
import re
from enum import Enum
import logging
import os
from typing import List, Tuple, Union

def splitSymbols(symbols, start_id, first_size, id_prog, get_prog, max_conc=50):
    with ClingoSolver() as ctl:
        ctl.addSymbols(symbols)
        ctl.load(Logic.pruneFile)
        ctl.ground([Logic.base, id_prog(start_id, first_size, max_conc)])
        ctl.solve()
        
        group_n = None
        atom_n = None
        group_n = ctl.getAtoms('groups', 1)[0].arguments[0].number
        atom_n = ctl.getAtoms('count', 1)[0].arguments[0].number
        print(group_n, atom_n)
        #print(ctl.getAtoms('conceptId', 2))
        ctl.ground([get_prog])
        result = []
        for i in range(group_n):
            ctl.assign_external(clingo.Function('show', [clingo.Number(i)]), False)
        for i in range(group_n):
            ctl.assign_external(clingo.Function('show', [clingo.Number(i)]), True)
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            result.append(symbols[0])
            ctl.assign_external(clingo.Function('show', [clingo.Number(i)]), False)

    return atom_n, result

class KnowledgeSet:
    def __init__(self, name, file):
        self.name = name
        self.file = str(file)

    def load(self, ctl):
        ctl.load(self.file)
'''
class ClingoFile:
    def __init__(self, filen, programs=[]):
        self.file = filen
        self.programs=programs
    def addProgram(self, program):
        self.programs.append(program)
    def hasProgram(self, program):
        return (program in self.programs)
    def load(self, ctl):
        ctl.load(self.file)

class ClingoProgram:
    def __init__(self, name, arg_num):
        self.name = name
        self.arg_num = arg_num

    def 
'''

class ConceptFile:
    def __init__(self, name, file_, depth):
        self.name = name
        self.file = file_
        self.depth = depth

class ClingoSolver:

    def __init__(self, ctlkwargs: dict = {}):

        self.__ctl = clingo.Control(**ctlkwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            print(type, value, traceback)
        del self.__ctl

    def load(self, clingo_files: Union[str, List[str]]) -> None:
        if not isinstance(clingo_files, list):
            self.__ctl.load(clingo_files)
            return
        for f in clingo_files:
            self.__ctl.load(f)

    def addSymbols(self, symbols: List[clingo.Symbol]):
        
        with self.__ctl.backend() as backend:
            for symbol in symbols:
                atom = backend.add_atom(symbol)
                backend.add_rule([atom], [], False)
        self.__ctl.cleanup()

    def ground(self, programs: List[Tuple]):
        self.__ctl.ground(programs)

    def solve(self, solvekwargs: dict = dict(yield_=False), symbolkwargs: dict = dict(atoms=True)):
        if not solvekwargs['yield_']:
            self.__ctl.solve(**solvekwargs)
            return
        symbols = []
        with self.__ctl.solve(**solvekwargs) as models:
            for model in models:
                symbols.append(model.symbols(**symbolkwargs))
        return symbols

    def assign_external(self, symbol: clingo.Symbol, val: bool):
        self.__ctl.assign_external(symbol, val)

    def reset(self, start_symbols: List[clingo.Symbol]):
        del self.__ctl
        self.__ctl = clingo.Control
        self.addSymbols(start_symbols)

    def countAtoms(self, name, arity):
        return len(self.getAtoms(name, arity))

    def getAtoms(self, name, arity):
        return [a.symbol for a in self.__ctl.symbolic_atoms.by_signature(name, arity) if a.is_fact]

    def cleanup(self):
        self.__ctl.cleanup()

    def getExternals(self):
        return [a.symbol for a in self.__ctl.symbolic_atoms if a.is_external]

'''
class GrammarKnowledge:
    def __init__(self, name, program, requirements, gen_file = Logic.grammarFile):
        self.name = name
        self._loaded = {}
        self._ctl = False
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
        
        ctl.load(self.genFile)
        ctl.ground([('base', []), self.program])
        self._solveControl(ctl)
        del(ctl)

    def generate(self):          
        self._solveProgram()
        self.isGenerated = True

    def filter(self, clingo_file, prune):
        ctl = clingo.Control()
        ctl.load(clingo_file)
        add_symbols(ctl, self._symbols)
        ctl.ground([('base', []), prune])
        self._solveControl(ctl)

        del(ctl)

    def getSymbols(self):
        return self._symbols

    def clean(self):
        del self._symbols[:]

    def write(self, option = 'w'):
        if not self.isGenerated:
            raise RuntimeError("Set {} has not been generated. Cannot write to file".format(self.name))
        if self.isStored:
            raise RuntimeError("Set {} has been stored in a file. Cannot overwrite".format(self.name))
        ModelUtil(self._symbols).write(self.file, type_=option)
        self.isStored = True
        self.clean()

class PruneableSet(GrammarKnowledge):
    def __init__(self, name, file, program, requirements, keep_f, keep_p, prune, final, cin, cout, gen_file=Logic.grammarFile):
        super(PruneableSet, self).__init__(name, file, program, requirements, gen_file=gen_file)
        self.isConcept = False
        self.expCount = 0
        self.keepFile = keep_f
        self.keepProg = keep_p
        self.pruneProg = prune
        self.finalProg = final
        self.compareIn = cin
        self.compareOut = cout

    def generate(self, comparison):
        super(PruneableSet, self)._solveProgram()

        self.filter(self.keepFile, self.keepProg)

        ctl = clingo.Control()
        add_symbols(ctl, self._symbols)
        ctl.load(Logic.pruneFile)
        ctl.ground([('base', []), self.compareIn])
        comparison.compare(ctl)
        ctl.ground([self.pruneProg])
        self._solveControl(ctl)

        del(ctl)
        self.isGenerated = True

    def removeRedundant(self, concepts, comparison):
        ctl = clingo.Control()
        add_symbols(ctl, self._symbols)
        ctl.load(Logic.pruneFile)
        concepts.load(ctl)
        ctl.ground([('base', []), self.compareOut])
        comparison.compare(ctl)
        ctl.ground([self.pruneProg])
        self._solveControl(ctl)
        del(ctl)

    def toFinal(self):
        if not self.isGenerated:
            raise RuntimeError("Set {} has not been generated. Cannot write to file".format(self.name))
        if self.isStored:
            raise RuntimeError("Set {} has been stored in a file. Cannot overwrite".format(self.name))
        ctl = clingo.Control()
        ctl.load(Logic.pruneFile)
        add_symbols(ctl, self._symbols)
        ctl.ground([('base', []), self.finalProg])
        self._solveControl(ctl)
        del(ctl)
    
    def write(self):
        raise RuntimeError("OHNO")

class ConceptSet(PruneableSet):
    def __init__(self, name, file, program, requirements):
        super(ConceptSet, self).__init__(
            name, file, program, requirements, Logic.pruneFile, Logic.keepExp, Logic.pruneExp,
            Logic.toConcept, Logic.compareExp, Logic.compareExpConc)

class FeatureSet(PruneableSet):
    def __init__(self, name, file, program, requirements):
        super(FeatureSet, self).__init__(
            name, file, program, requirements, Logic.featureFile, Logic.processFeat, Logic.pruneFeature,
            Logic.toFeature, Logic.comparePreFeature, Logic.compareFeature, gen_file=Logic.featureFile)
        

'''