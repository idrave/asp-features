import clingo
import subprocess
import features.solver as solver
import features.logic
from features.logic import Logic
from features.model_util import get_symbols
from typing import List

def run_plasp(in_file):
    return subprocess.run([features.logic.PLASP_PATH, 'translate', in_file], stdout=subprocess.PIPE).stdout.decode('utf-8')

class Instance:
    def __init__(self, pddl):
        self.pddl = pddl
        self.rules = run_plasp(self.pddl)
        self.depth = None
        self.state_n = 0
        self.transition_n = 0
        self.goal = False

    def get_rules(self):
        return self.rules

    def init_solver(self):
        self.solver = solver.create_solver()
        self.solver.open()
        self.solver.load(Logic.sample_file)
        self.solver.load(Logic.sample_encoding)
        self.solver.add('base', [], self.pddl)
        self.solver.ground(Logic.base)

    def close_solver(self):
        self.solver.close()

    def next_depth(self):
        return 0 if self.depth is None else self.depth + 1

    def expand(self):
        self.solver.ground([("expand", [self.next_depth()])])
        self.solver.solve()
        self.solver.cleanup()
        self.solver.ground([("prune", [self.next_depth()])])
        summary = self.solver.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        self.solver.cleanup()
        self.state_n = get_symbols(summary, 'state_count', 1)[0].number
        self.transition_n = get_symbols(summary, 'transition_count', 1)[0].number
        self.goal = get_symbols(summary, 'goal_count', 1)[0].number > 0

    def encode(self):
        self.solver.ground(['encode', []])
        symbols = self.solver.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
        return symbols


class Sample:
    def __init__(self, instances: List[Instance]):
        self.instances = instances

    def init_solvers(self):
        self.solvers = []
        for inst in self.instances:
            self.solvers.append(solver.create_solver())
            self.solvers[-1].open()
            self.solvers[-1].load(Logic.sample_file)
            self.solvers[-1].load(Logic.sample_encoding)
            self.solvers[-1].add('base', [], inst.pddl)
            

    def close_solvers(self):
        for s in self.solvers:
            s.close()

    def expand_states(self, depth=None, min_state=None, goal_req=True):
        d = 0
        state_n = 0
        goal = False

        while (depth != None and d < depth) or (min_state != None and state_n < min_state) \
              or (goal_req and not goal):
            pass
