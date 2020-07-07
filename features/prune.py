from features.comparison import Comparison
import clingo
from knowledge import Solver
from typing import List

class Pruneable:
    prune_file = None

    @staticmethod
    def init_sets(max_atom, max_comp):
        raise NotImplementedError()

    @staticmethod
    def show_set(set):
        raise NotImplementedError()

    @staticmethod
    def join_set(in1, in2, out):
        return ('join_set', [in1, in2, out])

    @staticmethod
    def prune_set(out):
        return ('prune_set', [out])

    @staticmethod
    def difference_set(in1, in2, out):
        return ('difference_set', [in1, in2, out])

    @staticmethod
    def prune_atom(ctl, exp_groups, start, comp : Comparison):
        out = start
        for i in range(exp_groups):
            ctl.ground([Pruneable.join_set(i, i, out)])
            comp.compare_set(ctl, out)
            ctl.ground([Pruneable.prune_set(out)])
            ctl.solve()
            ctl.cleanup()
            out += 1

        current_groups = exp_groups
        index = start
        
        while current_groups > 1:
            next_index = out
            for i in range(index, index + current_groups, 2):
                ctl.ground([Pruneable.join_set(i, i + 1, out)])
                #print('Joining:', i, i + 1, 'Out: ', out)
                comp.compare_set(ctl, out)
                ctl.ground([Pruneable.prune_set(out)])
                ctl.solve()
                ctl.cleanup()
                out += 1
            index = next_index
            current_groups = (current_groups + 1) // 2
        return index

    @staticmethod
    def prune_comp(ctl, exp_group, conc_start, conc_groups, comp: Comparison):
        for i in range(conc_start, conc_start + conc_groups):
            out = exp_group + 1
            ctl.ground([Pruneable.difference_set(exp_group, i, out)])
            comp.compare_set(ctl, out)
            ctl.ground([Pruneable.prune_set(out)])
            ctl.solve()
            ctl.cleanup()
            exp_group = out
        return exp_group

    @classmethod
    def prune_symbols(cls, symbols: List[clingo.Symbol], comp_symbols, comp: Comparison, max_atoms = 1, max_comp = 1):
        with Solver.open() as ctl:
            ctl.load(cls.prune_file)
            ctl.load(comp_symbols)
            ctl.addSymbols(symbols)
            ctl.ground([('base', []), (cls.init_sets(max_atoms, max_comp))])
            ctl.solve()
            exp_groups = ctl.getAtoms('atom_groups', 1)[0].arguments[0].number
            conc_groups = ctl.getAtoms('comp_groups', 1)[0].arguments[0].number
            unique_exp = Pruneable.prune_atom(ctl, exp_groups, exp_groups + conc_groups, comp)
            final_exp = Pruneable.prune_comp(ctl, unique_exp, exp_groups, conc_groups, comp)
            ctl.ground([cls.show_set(final_exp)])
            result = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))[0]
        return result