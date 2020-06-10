from pathlib import Path

class Logic:
    #logicPath = Path(__file__).parent.absolute()
    logicPath = Path('/home/ivan/Documents/ai/features/features')
    grammarFile = str(logicPath/'concept.lp')
    base = ('base', [])
    @staticmethod
    def primitive(depth):
        return ('primitive', [depth])
    
    roles = ('roles', [])

    @staticmethod
    def negation(depth):
        return ('negation', [depth])
    @staticmethod
    def equalRole(depth):
        return ('equal_role', [depth])
    @staticmethod
    def conjunction(depth, in1, in2):
        return ('conjunction', [depth, in1, in2])
    @staticmethod
    def uni(depth):
        return ('uni', [depth])
    @staticmethod
    def exi(depth):
        return ('exi', [depth])
    @staticmethod
    def enumerate(start, gsize):
        return ('enumerate', [start, gsize])

    classify = ('classify', [])
    pruneFile = str(logicPath/'prune.lp')
    keepExp = ('keep_exp', []) 
    compareExp = ('compare_exp', []) 
    pruneExp = ('prune_exp', [])
    compareExpConc = ('compare_exp_conc', []) 
    toConcept = ('exp2conc', [])
    keepRoles = ('get_roles', [])
    simplifySample = ('simplify', [])
    divide = ('divide', [])