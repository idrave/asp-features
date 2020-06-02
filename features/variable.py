import clingo
import model_util

class ConceptClass:
    def __init__(self, file_, required_files, required_programs):
        self.depth = depth
        self.file = file_
        self.program = program
        self.args = args
        self.stored = stored
        self.requirements = requirements
        self.ctl = None

    def ground(ctl: clingo.Control):
        if not expanded:
            raise RuntimeError("Cannot ground concept that has not been generated")
        ctl.load(self.file_)
        ctl.ground([(self.program, self.args())])

    def expand(prune, compare):
        self.ctl = clingo.Control()
        for req in requirements:
            req.ground(ctl)
        ctl.solve()
        prune.ground(ctl)
        compare.ground(ctl)



