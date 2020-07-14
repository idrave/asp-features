import clingo
import subprocess
import os
import features.solver as solver
import features.logic
from features.logic import Logic
from features.model_util import SymbolSet, write_symbols
from typing import List, Optional
import argparse

def run_plasp(in_file):
    return subprocess.run([features.logic.PLASP_PATH, 'translate', in_file], stdout=subprocess.PIPE).stdout.decode('utf-8')

class Instance:
    def __init__(self, pddl, numbered=True):
        self.pddl = pddl
        self.rules = run_plasp(self.pddl)
        self.numbered = numbered
        self.__depth = None
        self.__complete = False
        self.__changed = False
        self.__hold = None
        self.symbols = SymbolSet([])
        self.__init_encoding()
        
    def __init_encoding(self):
        def get_null_pred(variables):
            nulls = []
            for var in variables:
                pred = var.arguments[0].arguments[0]
                if(pred.type == clingo.SymbolType.String):
                    nulls.append(pred.string)
            return nulls

        with solver.create_solver() as ctl:
            self.init_solver(ctl)
            ctl.ground([(Logic.base)])
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(atoms=True))
            assert(len(symbols) == 1)
            variables = SymbolSet(symbols[0]).get_atoms('variable', 1)
            #Add predicates of arity 0
            ctl.addSymbols([clingo.Function('arity', [n, 0]) for n in get_null_pred(variables)])

            ctl.ground([('predicates', []), ('const', [])])
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(symbols) == 1)
            symset = SymbolSet(symbols[0])
            self.__pred = symset.get_atoms('pred', 1)
            self.__arity = symset.get_atoms('arity', 2)
            self.__const = symset.get_atoms('const', 1)

    def init_solver(self, ctl):
        ctl.load(Logic.sample_file)
        ctl.load(Logic.sample_encoding)
        ctl.add('base', [], self.rules)
        if self.symbols is not None:
            ctl.addSymbols(self.symbols.get_all_atoms())

    def get_rules(self):
        return self.rules

    def is_complete(self):
        return self.__complete

    def next_depth(self):
        return 0 if self.depth is None else self.depth + 1

    def expand(self):
        with solver.create_solver() as ctl:
            self.init_solver(ctl)
            ctl.ground([('base', []), ("expand", [self.next_depth()])])
            ctl.solve()
            ctl.cleanup()
            ctl.ground([("prune", [self.next_depth()])])
            ctl.solve()
            show_prog = 'show_numbered' if self.numbered else 'show_default' 
            ctl.ground([(show_prog, [self.next_depth()])])
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(symbols) == 1)
            symbols = symbols[0]
            ctl.cleanup()
            
            old_states = self.symbols.count_atoms('state', 1)
            self.symbols.add_symbols(symbols)
            new_states = self.symbols.count_atoms('state', 1)
            if new_states == old_states:
                self.__complete = True
            else:
                self.__depth = self.next_depth()
                self.__changed = True

    def state_number(self):
        if self.symbols is None: return 0
        return self.symbols.count_atoms('state', 1)

    def transition_number(self):
        if self.symbols is None: return 0
        return self.symbols.count_atoms('transition', 2)

    @property
    def depth(self):
        return self.__depth

    def is_goal(self):
        if self.symbols is None: return False
        return self.symbols.count_atoms('goal', 1) > 0

    def get_states(self):
        if self.symbols is None: return []
        return self.symbols.get_atoms('state', 1) + self.symbols.get_atoms('stateId', 2)

    def get_predicates(self):
        return self.__pred + self.__arity

    def get_const(self):
        return self.__const

    def get_transitions(self):
        if self.symbols is None: return []
        return self.symbols.get_atoms('transition', 2)

    def get_goal(self):
        if self.symbols is None: return []
        return self.symbols.get_atoms('goal', 1)

    def get_encoding(self):
        if self.__changed:
            with solver.create_solver() as ctl:
                self.init_solver(ctl)
                ctl.addSymbols(self.get_predicates() + self.get_const())
                ctl.ground([('base', []), ('hold', [])])            
                symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
                assert(len(symbols) == 1)
                self.__hold = symbols[0]
        return self.__hold + self.get_predicates() + self.get_const() + \
               self.get_states() + self.get_transitions() + self.get_goal()


class Sample:
    def __init__(self, instances: List[Instance]):
        self.instances = instances

    def expand_states(self, depth=None, states=None, transitions=None, goal_req=False, complete=False):
        d = -1
        s_n = 0
        t_n = 0
        stop = False

        while not stop:
            print('Expanding depth {}'.format(d+1))
            stop = True
            for instance in self.instances:
                if (depth != None and d < depth) or (states != None and s_n < states) \
                        or (transitions != None and t_n < transitions) \
                        or (goal_req and not instance.is_goal()) or (complete and not instance.is_complete()):
                    instance.expand()
                    stop = False
            if not stop:
                d += 1
                s_n = 0
                t_n = 0
                for instance in self.instances:
                    s_n += instance.state_number()
                    t_n += instance.transition_number()
                print('States {}. Transitions {}.'.format(s_n, t_n))

        self.depth = depth
        self.state_count = s_n
        self.transition_count = t_n

    def is_goal(self):
        return all([inst.is_goal() for inst in self.instances])

    def is_complete(self):
        return all(inst.is_complete() for inst in self.instances)

    def get_states(self) -> List[clingo.Symbol]:
        states = []
        for instance in self.instances:
            states += instance.get_states()
        return states

    def get_const(self) -> List[clingo.Symbol]:
        const = {}
        for instance in self.instances:
            for c in instance.get_const():
                const[c] = True
        return list(const.keys())

    def get_transitions(self) -> List[clingo.Symbol]:
        transitions = []
        for instance in self.instances:
            transitions += instance.get_transitions()
        return transitions

    def get_sample(self) -> List[clingo.Symbol]:
        result = []
        for instance in self.instances:
            result += instance.get_encoding()
        return result

class SampleFile:
    def __init__(self, s_file):
        self.file = s_file
        with solver.create_solver() as ctl:
            ctl.load(self.file)
            ctl.ground([Logic.base])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(sym) == 1)
            self.symset = SymbolSet(sym[0])

    def is_goal(self):
        return self.symset.count_atoms('goal', 1) > 0

    def is_complete(self):
        raise NotImplementedError #TODO add complete atom to encoding

    def get_states(self) -> List[clingo.Symbol]:
        return self.symset.get_atoms('state', 1)

    def get_const(self) -> List[clingo.Symbol]:
        return self.symset.get_atoms('const', 1)

    def get_transitions(self) -> List[clingo.Symbol]:
        return self.symset.get_atoms('transitions', 2)

    def get_sample(self) -> List[clingo.Symbol]:
        return self.symset.get_all_atoms()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pddl', nargs='+', required=True, help='Input instances in PDDL files')
    parser.add_argument('--symbol', action='store_true', help='Represent states in plain symbols instead of numericaly')
    parser.add_argument('-d', '--depth', default=6, type=int, help='Minimum expansion depth required')
    parser.add_argument('-s', dest='states', type=int, help='Minimum number of states required')
    parser.add_argument('-t', dest='transitions', type=int, help='Minimum number of transitions required')
    parser.add_argument('--complete', action='store_true', help='Expand all state space (could be too big!)')
    parser.add_argument('--goal', action='store_true', help='Ensure there is at least one goal per instance')
    parser.add_argument('--out', required=True, help='Output file path')
    args = parser.parse_args()
    instances = [Instance(p, numbered=not args.symbol) for p in args.pddl]
    sample = Sample(instances)
    sample.expand_states(
        depth=args.depth,
        states=args.states,
        transitions=args.transitions,
        goal_req=args.goal,
        complete=args.complete
    )
    write_symbols(sample.get_sample(), args.out)