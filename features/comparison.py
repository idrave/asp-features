from features.logic import Logic
from enum import Enum

class Comparison:
    def __init__(self, type_dict: dict, comp_type: int):
        self.type_ = type_dict.get(comp_type, None)
        if self.type_ is None:
            raise RuntimeError("Invalid comparison type {}.".format(comp_type))
        self.__compare, self.__compare_set = self.type_

    def compare(self, ctl):
        return self.__compare(self, ctl)

    def compare_set(self, ctl, num):
        return self.__compare_set(self, ctl, num)

class CompareConcept(Comparison):
    STANDARD = 1
    FAST = 2
    MIXED = 3

    def __init__(self, comp_type= STANDARD):
        self.file = str(Logic.logicPath/'differ.lp')

        def standard(self, ctl):
            ctl.load(str(self.file))
            ctl.ground([('standard_differ', [])])
            ctl.solve()

        def standard_set(self, ctl, num):
            ctl.load(str(self.file))
            ctl.ground([('standard_differ_set', [num])])
            ctl.solve()

        def fast(self, ctl):
            ctl.load(str(self.file))
            ctl.ground([('fast_differ', [])])
            ctl.solve()

        def fast_set(self, ctl, num):
            ctl.load(str(self.file))
            ctl.ground([('fast_differ_set', [num])])
            ctl.solve()

        def mixed(self, ctl):
            ctl.load(str(self.file))
            ctl.ground([('optimal_differ_start', [])])
            ctl.solve()
            ctl.ground([('optimal_differ_end', [])])
            ctl.solve()

        def mixed_set(self, ctl, num):
            ctl.load(str(self.file))
            ctl.ground([('optimal_differ_set_start', [num])])
            ctl.solve()
            ctl.ground([('optimal_differ_set_end', [num])])
            ctl.solve()

        types = {
            CompareConcept.STANDARD : (standard, standard_set),
            CompareConcept.FAST : (fast, fast_set),
            CompareConcept.MIXED : (mixed, mixed_set)
        }
        
        super(CompareConcept, self).__init__(types, comp_type)
        
class CompareFeature(Comparison):

    STANDARD = 1

    def __init__(self, comp_type = STANDARD):
        self.file = str(Logic.logicPath/'differ.lp')
        def feature(self, ctl):
            ctl.load(str(self.file))
            ctl.ground([('feature_differ', [])])
            ctl.solve()

        def feature_set(self, ctl, num):
            ctl.load(str(self.file))
            ctl.ground([('feature_differ_set', [num])])
            ctl.solve()

        types = {
            CompareFeature.STANDARD : (feature, feature_set)
        }

        super(CompareFeature, self).__init__(types, comp_type)