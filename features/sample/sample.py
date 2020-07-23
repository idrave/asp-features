import clingo
import subprocess
import os
import features.solver as solver
import features.logic
from features.logic import Logic
from features.model_util import SymbolSet, write_symbols
from typing import List, Optional
import argparse
from pathlib import Path
import sys
import json

def run_plasp(in_file):
    return subprocess.run([features.logic.PLASP_PATH, 'translate', in_file], stdout=subprocess.PIPE).stdout.decode('utf-8')

#TODO: conditional effects are not properly translated by plasp.

class Instance:
    def __init__(self, pddl=None, numbered=True, load_info=None):
        if load_info != None:
            self.load(**load_info)
        else:
            self.pddl = str(Path(pddl).absolute())
            self.rules = run_plasp(self.pddl)
            self.numbered = numbered
            self.__depth = None
            self.__complete = False
            self._relevant = None
            self.symbols = SymbolSet([])
            self.__init_encoding()
            self._state_n = []      #Number of states in each depth
            self._transition_n = [] #Number of transitions starting in each depth-1
        
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

    def expand(self, index_start=None):
        if self.is_complete(): return
        if index_start == None:
            index_start = self.state_number()
        with solver.create_solver() as ctl:
            self.init_solver(ctl)
            ctl.ground([('base', []), ("expand", [self.next_depth()])])
            ctl.solve()
            ctl.cleanup()
            ctl.ground([("prune", [self.next_depth()])])
            ctl.solve()
            if self.numbered:
                ctl.ground([('show_numbered', [self.next_depth(), index_start])])
            else:
                ctl.ground([('show_default', [self.next_depth()])]) 
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(symbols) == 1)
            symbols = symbols[0]
            
            ctl.cleanup()
            old_states = self.state_number()
            self.symbols.add_symbols(symbols)
            new_states = self.state_number()
            self._state_n.append(new_states)
            self._transition_n.append(self.transition_number())
            if new_states == old_states:
                self.__complete = True
            else:
                self.__depth = self.next_depth()

    def state_number(self, depth=None):
        """
        Number of states in the whole instance. If depth is specified,
        number of states with such depth
        """
        assert(depth == None or depth >= 0)
        if self.symbols is None: return 0
        if depth == None: return self.symbols.count_atoms('state', 1)
        if depth > self.depth: return 0
        return self._state_n[depth]
         

    def transition_number(self, depth=None):
        """
        Number of transitions in the whole instance. If depth is specified,
        number of transitions starting in a node with such depth-1
        """
        assert(depth == None or depth >= 0)
        if self.symbols is None: return 0
        if depth == None: return self.symbols.count_atoms('transition', 2)
        if depth > self.depth: return 0
        return self._transition_n[depth]

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

    def get_encoding(self, max_depth=None, optimal=False):
        max_depth = max_depth if max_depth != None else self.depth
        assert(not optimal or self.is_goal())
        optimal = int(optimal)
        
        with solver.create_solver() as ctl:
            self.init_solver(ctl)
            ctl.addSymbols(self.get_predicates() + self.get_const())
            if optimal:
                ctl.addSymbols(self.get_relevant())
            ctl.ground([('base', []), ('hold', []), ('get_encoding', [max_depth, optimal])])            
            symbols = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(symbols) == 1)
            symbols = symbols[0]
        return symbols + self.get_predicates() + self.get_const()

    def load(self, pddl, numbered, depth, complete, file_n, state_n, transition_n):
        self.pddl = pddl
        self.rules = run_plasp(self.pddl)
        self.numbered = numbered
        self.__depth = depth
        self.__complete = complete
        with solver.create_solver() as ctl:
            ctl.load(file_n)
            ctl.ground([Logic.base])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            assert(len(sym) == 1)
            self.symbols = SymbolSet(sym[0])
        self.__init_encoding()
        self._state_n = state_n
        self._transition_n = self._transition_n

    def store(self, path):
        write_symbols(self.symbols.get_all_atoms(), path)
        info = {
            'pddl': self.pddl,
            'numbered': self.numbered,
            'depth': self.__depth,
            'complete': self.__complete,
            'file_n': path,
            'state_n': self._state_n,
            'transition_n': self._transition_n
        }
        return info

    def get_relevant(self):
        assert(self.is_goal())
        if self._relevant is None:
            with solver.create_solver() as ctl:
                ctl.load(Logic.sample_marking)
                ctl.addSymbols(self.get_goal() + self.symbols.get_atoms('state', 2) + self.get_transitions())
                ctl.ground([Logic.base, ('relevant', [])])
                sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
                assert(len(sym) == 1)
                self._relevant = sym[0]
        return self._relevant


class Sample:
    def __init__(self, instances: List[Instance]=None, load_path = None):
        if load_path == None:
            self.instances = instances
            self.depth = None
            self.state_count = 0
            self.transition_count = 0
        else:
            self.load(load_path)

    def expand_states(self, depth=None, states=None, transitions=None, goal_req=False, complete=False):
        d = self.depth if self.depth != None else -1
        s_n = self.state_count
        t_n = self.transition_count
        stop = False

        while not stop:
            stop = True
            s_aux = s_n
            for instance in self.instances:
                s_old = instance.state_number()
                if (depth != None and d < depth) or (states != None and s_n < states) \
                        or (transitions != None and t_n < transitions) \
                        or (goal_req and not instance.is_goal()) or (complete and not instance.is_complete()):
                    instance.expand(index_start=s_aux)
                    s_aux += instance.state_number() - s_old
                    stop = False
            
            if not stop:
                print('Expanded depth {}'.format(d+1))
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

    def get_instances(self):
        return self.instances

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

    def load(self, path):
        path = Path(path)
        with open(str(path/'sample.json'), 'r') as fp:
            info = json.load(fp)
        self._load(**info)
        
    def _load(self, depth, states, transitions, instances):
        self.depth = depth
        self.state_count = states
        self.transition_count = transitions
        self.instances = []
        for inst in instances:
            self.instances.append(Instance(load_info=inst))

    def store(self, path):
        path = Path(path)
        if not path.is_dir():
            try:
                path.mkdir()
            except (FileNotFoundError, FileExistsError) as e:
                print(repr(e))
                sys.exit()
        info = {
            'depth': self.depth,
            'states': self.state_count,
            'transitions': self.transition_count,
            'instances': []
        }
        for i, instance in enumerate(self.instances):
            info['instances'].append(instance.store(str(path/'instance_{}.lp'.format(i))))
        with open(str(path/'sample.json'), 'w') as fp:
            json.dump(info, fp)

    def get_relevant(self):
        result = []
        for inst in self.instances:
            result += inst.get_relevant()
        return result
    
    def get_view(self, depth=None, states=None, transitions=None,
                 goal_req=False, complete=False, optimal=False):
        goal_req = goal_req or optimal
        self.expand_states(
            depth=depth, states=states, transitions=transitions,
            goal_req=goal_req, complete=complete
        )
        if complete: return SampleView(self)

        s_n = 0
        t_n = 0
        for d in range(self.depth+1):
            for inst in self.instances:
                s_n += inst.state_number(depth=d)
                t_n += inst.transition_number(depth=d)
            if (states == None or s_n >= states) and (transitions == None or t_n >= transitions):
                break

        depth = max(depth, d)
        assert(
            self.is_complete() or 
            ((states == None or s_n >= states) and (transitions == None or t_n >= transitions)))
        return SampleView(self, max_depth=depth, optimal=optimal)

    def print_info(self):
        print(('Number of states: {}\nNumber of transitions: {}\n'
            'Includes goals: {}\nComplete sample: {}\n')
            .format(self.state_count,self.transition_count,self.is_goal(),self.is_complete()))


class SampleView:

    def __init__(self, sample: Sample, max_depth=None, optimal=None):
        inst = sample.get_instances()
        sym = []
        for i in inst:
            sym += i.get_encoding(max_depth= max_depth, optimal= optimal)
        self.symbols = SymbolSet(sym)
        self._complete = sample.depth <= max_depth

    def is_goal(self):
        return self.symbols.count_atoms('goal', 1) > 0

    def is_complete(self):
        return self._complete

    def get_states(self) -> List[clingo.Symbol]:
        return self.symbols.get_atoms('state', 1) + self.symbols.get_atoms('stateId', 2)

    def get_const(self) -> List[clingo.Symbol]:
        const = {}
        for c in self.symbols.get_atoms('const', 1):
            const[c] = True
        return list(const.keys())

    def get_transitions(self) -> List[clingo.Symbol]:
        return self.symbols.get_atoms('transition', 2)

    def get_sample(self) -> List[clingo.Symbol]:
        return self.symbols.get_all_atoms()

    def get_relevant(self):
        return self.symbols.get_atoms('relevant', 2)

    def print_info(self):
        print(('Number of states: {}\nNumber of transitions: {}\n'
                'Includes goals: {}\nComplete sample: {}\n')
                .format(
                    self.symbols.count_atoms('state', 1),
                    self.symbols.count_atoms('transition', 2),
                    self.is_goal(),self.is_complete()))

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
        return self.symset.get_atoms('transition', 2)

    def get_sample(self) -> List[clingo.Symbol]:
        return self.symset.get_all_atoms()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pddl', nargs='+', required=True, help='Input instances in PDDL files')
    parser.add_argument('--symbol', action='store_true', help='Represent states in plain symbols instead of numericaly')
    parser.add_argument('-d', '--depth', default=None, type=int, help='Minimum expansion depth required')
    parser.add_argument('-s', dest='states', type=int, help='Minimum number of states required')
    parser.add_argument('-t', dest='transitions', type=int, help='Minimum number of transitions required')
    parser.add_argument('--complete', action='store_true', help='Expand all state space (could be too big!)')
    parser.add_argument('--goal', action='store_true', help='Ensure there is at least one goal per instance')
    parser.add_argument('--relevant', action='store_true', help='Print relevant transitions of the sample')
    parser.add_argument('--out', required=True, help='Output file path')
    parser.add_argument('-load', default=None, help='Load from stored sample')
    args = parser.parse_args()
    if args.load == None:
        instances = [Instance(p, numbered=not args.symbol) for p in args.pddl]
        sample = Sample(instances=instances)
    else:
        sample = Sample(load_path=args.load)
    sample.expand_states(
        depth=args.depth,
        states=args.states,
        transitions=args.transitions,
        goal_req=args.goal,
        complete=args.complete
    )
    if args.relevant:
        print(sample.get_relevant())
    sample.store(args.out)