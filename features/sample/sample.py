import clingo
import subprocess
import os
import features.solver as solver
import features.logic
from features.logic import Logic
from features.model_util import SymbolSet, write_symbols
from features.sample.problem import Problem, Node
from features.sample.problem import time_eq
from typing import List, Optional
import argparse
from pathlib import Path
import logging
import sys
import json
import time

def run_plasp(in_file):
    return subprocess.run([features.logic.PLASP_PATH, 'translate', in_file], stdout=subprocess.PIPE).stdout.decode('utf-8')

#TODO: conditional effects are not properly translated by plasp.

class Instance:
    def __init__(self, prob: Problem, numbered=True):
        self._problem = prob
        self.numbered = numbered
        self._depth = None
        self._complete = False
        self._goals = []
        self._optimal = None
        self._relevant = None
        self._st_id = {} #States by id
        self._states = [] #States by depth
        self._st_set = {} #States by hash
        self._next_id = 0
        #self.__init_encoding()
        self._state_n = 0      #Number of states
        self._transition_n = 0 #Number of transitions
        
    @staticmethod
    def load(problem, numbered, depth, complete, states, optimal):
        problem = Problem.load(problem)
        instance = Instance(problem, numbered=numbered)
        instance._depth = depth
        instance._complete = complete
        assert(instance.depth+1 == len(states) )
        instance._states = []
        for d in range(instance.depth+1):
            instance._states.append([])
            for s in states[d]:
                node = Node.load_state(s)
                instance._states[-1].append(node)
                instance._st_set[node] = node
                instance._st_id[node.id] = node
                instance._state_n += 1
                instance._transition_n += len(node.transitions)
                instance._next_id = max(instance._next_id, node.id + 1)
                if node.is_goal:
                    instance._goals.append(node)
                    if node.id == optimal:
                        instance._optimal = node
        return instance

    #TODO fix store/load
    def store(self):
        info = {
            'problem': self.problem.store(),
            'numbered': self.numbered,
            'depth': self.depth,
            'complete': self.is_complete,
            'states': [[st.store() for st in self._states[d]] for d in range(self.depth+1)],
            'optimal': self._optimal.id if self.is_goal else None
        }
        return info
    
    @property
    def is_complete(self):
        return self._complete

    def next_depth(self):
        return 0 if self.depth is None else self.depth + 1

    def find_state(self, node: Node) -> Optional[Node]:
        return self._st_set.get(node, None)
        
    def expand(self, start_id=None):
        print('Call to expand on depth', self.depth)
        assert(start_id == None or start_id >= self._next_id)
        if start_id != None: self._next_id = start_id
        if self.is_complete: return
        self._states.append([])

        def add_state(node: Node):
            node.make_state(self._next_id)
            node.encode(self.problem)
            #print('Adding state {}, hash {}'.format(node.id, hash(node)))
            #print([str(s) for s in node.symbols.get_all_atoms()])
            self._st_id[node.id] = node
            self._st_set[node] = node
            self._states[self.next_depth()].append(node)
            self._state_n += 1
            self._next_id += 1
            if node.is_goal:
                if not self.is_goal:
                    self._optimal = node
                self._goals.append(node)
            
        if self.next_depth() == 0:
            add_state(self.problem.initial_state())
        else:
            for state in self._states[self.depth]:
                nodes = self.problem.get_successors(state)
                for n in nodes:
                    st = self.find_state(n)
                    if st == None:
                        add_state(n)
                        st = n
                    state.add_child(st)
                    self._transition_n += 1
        #print('Next Depth', self.next_depth())
        self._depth = self.next_depth()  
        print('New States', self.count_states(depth=self.depth))
        if self.count_states(depth=self.depth) == 0:
            print('COMPLETE')
            self._complete = True 
             
    def count_states(self, depth=None):
        """
        Number of states in the whole instance. If depth is specified,
        number of states with such depth
        """
        assert(depth == None or depth >= 0)
        if depth == None: return self._state_n
        if self.depth == None or depth > self.depth: return 0
        return len(self._states[depth])

    def count_transitions(self, depth=None):
        """
        Number of transitions in the whole instance. If depth is specified,
        number of transitions starting in a node with such depth
        """
        assert(depth == None or depth >= 0)
        if depth == None: return self._transition_n
        if depth > self.depth: return 0
        return sum([len(st.transitions) for st in self._states[depth]])

    @property
    def depth(self) -> Optional[int]:
        return self._depth

    @property
    def is_goal(self):
        return len(self._goals) > 0

    @property
    def states(self):
        if self.depth == None: return []
        states = []
        for d in range(self.depth+1):
            for st in self._states[d]:
                states += st.state
        return states

    @property
    def transitions(self):
        if self.depth == None: return []
        transitions = []
        for d in range(self.depth+1):
            for st in self._states[d]:
                transitions += st.transitions
        return transitions

    @property
    def goals(self):
        if not self.is_goal: return []
        goals = []
        for g in self._goals:
            goals.append(g.goal)
        return goals

    @property
    def problem(self):
        return self._problem

    def encoding(self, max_depth=None, optimal=False):
        max_depth = self.depth if max_depth == None else min(self.depth, max_depth)
        sym = self.problem.const + self.problem.predicates
        for d in range(max_depth):
            for st in self._states[d]:
                sym += st.encoding + st.transitions
        for st in self._states[max_depth]:
            sym += st.encoding
        
        if optimal:
            assert(self.is_goal)
            n = self._optimal
            while n.parent != None:
                n = self._st_id[n.parent]
                sym += n.transitions
                for ch in n.children:
                    sym += self._st_id[ch].encoding
                if n.parent == None:
                    sym += n.encoding
            sym += self.relevant
        return sym

    @property
    def relevant(self) -> List[clingo.Symbol]:
        rel = []
        n = self._optimal
        while n.parent != None:
            rel.append(clingo.Function('relevant', [n.parent, n.id]))
            n = self._st_id[n.parent]
        return rel


class Sample:
    def __init__(self, instances: List[Instance]=None):
        self.instances = instances
        self.depth = None
        self.state_count = 0
        self.transition_count = 0

    def expand_states(self, depth=None, states=None, transitions=None, goal_req=False, complete=False):
        d = self.depth if self.depth != None else -1
        logging.debug('Expanding from {}'.format(d))
        s_n = self.state_count
        t_n = self.transition_count
        stop = False

        while not stop:
            stop = True
            s_aux = s_n
            for instance in self.instances:
                s_old = instance.count_states()
                if (depth != None and d < depth) or (states != None and s_n < states) \
                        or (transitions != None and t_n < transitions) \
                        or (goal_req and not instance.is_goal) or (complete and not instance.is_complete):
                    instance.expand(start_id=s_aux)
                    s_aux += instance.count_states() - s_old
                    stop = False
            
            if not stop:
                print('Expanded depth {}'.format(d+1))
                d += 1
                s_n = 0
                t_n = 0
                for instance in self.instances:
                    s_n += instance.count_states()
                    t_n += instance.count_transitions()
                print('States {}. Transitions {}.'.format(s_n, t_n))

        self.depth = d
        self.state_count = s_n
        self.transition_count = t_n

    def get_instances(self):
        return self.instances

    @property
    def is_goal(self):
        return all([inst.is_goal for inst in self.instances])

    @property
    def is_complete(self):
        return all(inst.is_complete for inst in self.instances)

    def get_states(self) -> List[clingo.Symbol]:
        states = []
        for instance in self.instances:
            states += instance.states
        return states

    def get_const(self) -> List[clingo.Symbol]:
        const = {}
        for instance in self.instances:
            for c in instance.problem.const:
                const[c] = True
        return list(const.keys())

    def get_transitions(self) -> List[clingo.Symbol]:
        transitions = []
        for instance in self.instances:
            transitions += instance.transitions
        return transitions

    def get_sample(self) -> List[clingo.Symbol]:
        result = []
        for instance in self.instances:
            result += instance.encoding()
        return result

    @staticmethod
    def load(path):
        def load(sample, depth, states, transitions, instances):
            sample.depth = depth
            sample.state_count = states
            sample.transition_count = transitions
            sample.instances = []
            for inst in instances:
                sample.instances.append(Instance.load(**inst))
        path = Path(path)
        with open(str(path/'sample.json'), 'r') as fp:
            info = json.load(fp)
        sample = Sample([])
        load(sample, **info)
        return sample

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
        for instance in self.instances:
            info['instances'].append(instance.store())
        with open(str(path/'sample.json'), 'w') as fp:
            json.dump(info, fp)

    def get_relevant(self):
        result = []
        for inst in self.instances:
            result += inst.relevant
        return result
    
    def get_view(self, depth=None, states=None, transitions=None,
                 goal_req=False, complete=False, optimal=False):
        print('Optimal',optimal)
        goal_req = goal_req or optimal
        logging.debug('View from sample of depth {}'.format(self.depth))
        self.expand_states(
            depth=depth, states=states, transitions=transitions,
            goal_req=goal_req, complete=complete
        )
        if complete: return SampleView(self, optimal=optimal)
        #TODO fix sum
        depth = depth if depth != None else 0
        s_n = 0
        t_n = 0
        d_s = 0 if states == None else None
        d_t = 0 if transitions == None else None
        for d in range(self.depth+1):
            for inst in self.instances:
                s_n += inst.count_states(depth=d)
                t_n += inst.count_transitions(depth=d)
            if (d_s == None and s_n >= states):
                d_s = d
            if (d_t == None and t_n >= transitions):
                d_t = d
            if (d_s != None and d_t != None):
                break
        print(depth, d_s, d_t)
        depth = max(depth, d_s, d_t+1)
        assert(
            self.is_complete or 
            ((states == None or s_n >= states) and (transitions == None or t_n >= transitions)))
        return SampleView(self, max_depth=depth, optimal=optimal)

    def print_info(self):
        print(('Number of states: {}\nNumber of transitions: {}\n'
            'Includes goals: {}\nComplete sample: {}\n')
            .format(self.state_count,self.transition_count,self.is_goal,self.is_complete))


class SampleView:

    def __init__(self, sample: Sample, max_depth=None, optimal=False):
        print('Generating SAMPLE AAAAAAAAAAAAAAAAAAa', optimal)
        max_depth = max_depth if max_depth != None else sample.depth
        inst = sample.get_instances()
        sym = []
        for i in inst:
            sym += i.problem.const + i.problem.predicates
            sym += i.encoding(max_depth=max_depth, optimal=optimal)
            if optimal:
                sym += i.relevant
                print(i.relevant)
        self.symbols = SymbolSet(sym)
        self._complete = sample.is_complete and sample.depth <= max_depth
        if max_depth == None or sample.depth <= max_depth:
            self.depth = sample.depth
        else:
            self.depth = max_depth
        self.optimal = optimal

    @property
    def is_goal(self):
        return self.symbols.count_atoms('goal', 1) > 0

    @property
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
        print(('Sample up to depth: {}. Optimal solution included: {}\n'
                'Number of states: {}\nNumber of transitions: {}\n'
                'Includes goals: {}\nComplete sample: {}\n')
                .format(
                    self.depth, self.optimal,
                    self.symbols.count_atoms('state', 1),
                    self.symbols.count_atoms('transition', 2),
                    self.is_goal,self.is_complete))

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
    parser.add_argument('--pddl', nargs='+', help='Input instances in PDDL files')
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
        instances = [Instance(Problem(p), numbered=not args.symbol) for p in args.pddl]
        sample = Sample(instances=instances)
    else:
        sample = Sample.load(args.load) #TODO fix this
    start = time.time()
    sample.expand_states(
        depth=args.depth,
        states=args.states,
        transitions=args.transitions,
        goal_req=args.goal,
        complete=args.complete
    )
    print('Time expandind: ', time.time() - start)
    sample.print_info()
    if args.relevant:
        print(sample.get_relevant())
    sample.store(args.out)
    print('Time on equality:', features.sample.problem.time_eq)