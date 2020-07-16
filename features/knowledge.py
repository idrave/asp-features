import clingo
from features.logic import Logic
import features.solver as solver
import logging
from typing import List, Tuple, Union, Optional

def splitSymbols(symbols, start_id, first_size, id_prog, get_prog, max_conc=50):
    with solver.create_solver() as ctl:
        ctl.addSymbols(symbols)
        ctl.load(Logic.pruneFile)
        ctl.ground([Logic.base, id_prog(start_id, first_size, max_conc)])
        ctl.solve()
        group_n = None
        atom_n = None
        group_n = ctl.getAtoms('groups', 1)[0].arguments[0].number
        atom_n = ctl.getAtoms('count', 1)[0].arguments[0].number
        logging.debug('Split groups: {}. Atoms: {}'.format(group_n, atom_n))
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

def number_symbols(symbols, start_id, id_prog):
    with solver.create_solver() as ctl:
        ctl.addSymbols(symbols)
        ctl.load(Logic.pruneFile)
        ctl.ground([Logic.base, id_prog(start_id)])
        symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        
        atom_n = None
        atom_n = ctl.getAtoms('count', 1)[0].arguments[0].number

    return atom_n, symbols


class ConceptFile:
    def __init__(self, name, file_, depth):
        self.name = name
        self.file = file_
        self.depth = depth