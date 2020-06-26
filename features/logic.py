from pathlib import Path

class Logic:
    #logicPath = Path(__file__).parent.absolute()
    logicPath = Path('/home/ivan/Documents/ai/features/features')
    grammarFile = str(logicPath/'concept.lp')
    featureFile = str(logicPath/'features.lp')
    base = ('base', [])
    roles = ('roles', [])
    

    pruneFile = str(logicPath/'prune.lp')
    
    keepRoles = ('get_roles', [])
    @staticmethod
    def index_role(start):
        return ('index_role', [start])
    @staticmethod
    def number_state(start):
        return ('index_state', [start])
    simplifySample = ('simplify', [])
    divide = ('divide', [])

    primitiveFeature = ('primitiveFeature', [])
    conceptFeature = ('conceptFeature', [])
    @staticmethod
    def distFeature(k):
        return ('distFeature', [k])

    transitions = ('transitions', [])
    processFeat = ('feature', [])
    comparePreFeature = ('compare_prefeature', [])
    compareFeature = ('compare_feature', [])
    pruneFeature = ('prune_feature', [])
    toFeature = ('to_feature', [])
    @staticmethod
    def enumerateFeat(start, gsize):
        return ('enumerate_feat', [start, gsize])
    classifyFeat = ('classify_feat', [])
    
class Concept:
    cardinality = ('cardinality', [])
    compareExp = ('compare_exp', [])
    keepExp = ('keep_exp', [])
    pruneExp = ('prune_exp', [])
    compareExpConc = ('compare_exp_conc', [])

    @staticmethod
    def numberConc(start, first, gsize):
        return ('number_conc', [start, first, gsize])

    classify = ('classify', [])
    #classifyExp = ('classify_exp', [])

    @staticmethod
    def primitive(depth):
        return ('primitive', [depth])
    
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