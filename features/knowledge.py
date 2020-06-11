import clingo
from model_util import add_symbols, ModelUtil, check_multiple, splitSymbols
from features.logic import Logic
from pathlib import Path
import pickle
import re
from enum import Enum
import logging
import os

class ClingoFile:
    def __init__(self, filename):
        self.file = str(filename)
    def load(self, ctl):
        ctl.load(self.file)
    def write(self, symbols, option='w'):
        ModelUtil(symbols).write(self.file, type_=option)

class PickleFile:
    def __init__(self, filename):
        self.file = str(filename)
    def load(self, ctl):
        symbols = pickle.load(self.file)
        add_symbols(ctl, symbols)
    def write(self, symbols, option='w'):
        if option == 'w':
            with open(self.file, 'wb') as fileh:
                pickle.dump(symbols, fileh)
        elif option == 'a':
            saved = None
            with open(self.file, 'rb') as fileh:
                saved = pickle.load(fileh)
            with open(self.file, 'wb') as fileh:
                pickle.dump(saved + symbols,fileh)
        else:
            raise RuntimeError('PickleFile write type {} not defined.'.format(option))


class FileCreator:
    ext = {'clingo':'lp', 'pickle':'pickle'}
    types = {'clingo':ClingoFile, 'pickle':PickleFile}

    @staticmethod
    def checkType(filename):
        extension = re.search(r'\.(.*?)$', filename)
        print(extension.group(1))
        if extension is None: return None
        for type_ in FileCreator.ext:
            if FileCreator.ext[type_] == extension.group(1):
                return type_

    @staticmethod
    def create(filename):
        filetype = FileCreator.checkType(str(filename))
        if filetype is None:
            raise RuntimeError('File type not specified for {}'.format(filename))
        return FileCreator.types[filetype](filename)
    @staticmethod
    def createType(filename, type):
        pass

class KnowledgeSet:
    def __init__(self, name, file):
        self.name = name
        self.file = str(file)

    def load(self, ctl):
        ctl.load(self.file)

class GrammarKnowledge(KnowledgeSet):
    def __init__(self, name, file, program, requirements, gen_file = Logic.grammarFile):
        super(GrammarKnowledge, self).__init__(name, file)
        
        self.genFile = gen_file
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
        

