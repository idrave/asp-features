
import clingo
import sys
import logging
import pathlib
import re
import argparse
from pathlib import Path
from model_util import check_multiple
from model_util import to_model_list, ModelUtil, filter_symbols
pathlib.Path(__file__).parent.absolute()


class Comparison:
    def __init__(self, path, comp_type='standard'):
        self.file = path/'differ.lp'
        types = {
            'standard' : self.standardCompare,
            'fast' : self.fastCompare,
            'mixed' : self.mixedCompare
        }
        self.compare = types.get(comp_type, None)
        if self.compare is None:
            raise RuntimeError("Invalid comparison type.")

    def standardCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('standard_differ', [])])
        ctl.solve()

    def fastCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('fast_differ', [])])
        ctl.solve()
    
    def mixedCompare(self, ctl: clingo.Control):
        ctl.load(str(self.file))
        ctl.ground([('optimal_differ_start', [])])
        ctl.solve()
        ctl.ground([('optimal_differ_end', [])])
        ctl.solve()

class Grammar:
    def __init__(self, sample, path, comp_type='standard'):
        self.logic_path = Path(__file__).parent.absolute()
        self.sample = Path(sample)
        self.concept_file = self.logic_path/'concept.lp'
        self.sample_code = 'sample'
        self.simple_code = 'simple'
        self.roles = 'roles'
        self.layers = {self.sample_code : [self.sample]}
        self.concepts = []
        self.path = Path(path)
        self.prune = self.logic_path/'prune.lp'
        self.prune_exp = ('prune_exp', [])
        self.prune_exp_conc = ('prune_exp_conc', [])
        self.keep_exp = ('keep_exp', [])
        self.get_roles= ('get_roles', [])
        self.to_concept= ('exp2conc', [])
        self.simplify= ('simplify', [])
        self.compare = Comparison(self.logic_path, comp_type=comp_type)
    
    def createDir(self):
        print(self.path.absolute())
        if not self.path.is_dir():
            try:
                self.path.mkdir()
            except (FileNotFoundError, FileExistsError) as e:
                print(repr(e))
                sys.exit()

    #Return set of symbols regarding certain concept
    def expandConceptSet(self, program, program_args, layers, out_file, roles=False, logg=False):
        ctl = clingo.Control()
        ctl.load(str(self.concept_file))
        for layer in layers:
            self.groundLayer(ctl, layer)
        ctl.ground([(program, program_args)])
        with ctl.solve(yield_=True) as models:
            model_list = [model for model in models]
            if len(model_list) != 1:
                raise RuntimeError("Only one model expected, more than one found.")

        symbols = []
        if not roles:
            symbols = self.prune_symbols(ctl, self.prune_exp, self.keep_exp)
        else:
            symbols = self.filter_symbols(ctl, self.get_roles)
        del(ctl)
        logging.debug("Expressions: {}".format(sum([symbol.name == 'exp' for symbol in symbols])))
        if not roles:
            for concept in self.concepts:
                ctl = clingo.Control()
                ctl.add('base', [], str(ModelUtil(symbols)))
                ctl.load(str(concept))
                ctl.ground([('base', [])])
                symbols = self.prune_symbols(ctl, self.prune_exp_conc, self.keep_exp)
                del(ctl)
            logging.debug("Expressions: {}".format(sum([symbol.name == 'exp' for symbol in symbols])))
            ctl = clingo.Control()
            ctl.add('base', [], str(ModelUtil(symbols)))
            ctl.ground([('base', [])])
            symbols = self.filter_symbols(ctl, self.to_concept)
            del(ctl)
            
        ModelUtil(symbols).write(str(out_file))
        if logg:
            count = self.countConcepts(out_file)
            print('{}: {} concepts.'.format(out_file, count))


    def groundLayer(self, ctl, i):
        if i not in self.layers:
            raise RuntimeError('Cannot access set of symbols that does not exist.')
        for concept_file in self.layers[i]:
            ctl.load(str(concept_file))
        ctl.ground([("base", [])])

    def expandLayer(self, depth, logg=False):

        self.layers[depth] = []
        if depth == 1:
            primitive_file = self.path/'primitive.lp'
            self.expandConceptSet('primitive', [depth], [self.sample_code], primitive_file, logg=logg)
            self.layers[depth].append(primitive_file)
            self.concepts.append(primitive_file)
            return

        if depth == 2:
            negation_file = self.path/'negation.lp'
            self.expandConceptSet('negation', [depth], [depth-1, self.simple_code], negation_file, logg=logg)
            self.layers[depth].append(negation_file)
            self.concepts.append(negation_file)
            return

        if depth == 3:
            equalrole_file = self.path/'equal_role.lp'
            self.expandConceptSet('equal_role', [depth], [self.roles, self.simple_code], equalrole_file, logg=logg)
            self.layers[depth].append(equalrole_file)
            self.concepts.append(equalrole_file)

        for i in range(1,depth-1):
            conjunction_file = self.path/'conjunction_{}_{}.lp'.format(depth, i)
            self.expandConceptSet('conjunction', [depth, i, depth-i-1],
                [i, depth-i-1, self.simple_code], conjunction_file, logg=logg)
            self.layers[depth].append(conjunction_file)
            self.concepts.append(conjunction_file)

        exi_file = self.path/'exi_{}.lp'.format(depth)
        self.expandConceptSet('exi', [depth], [depth-2,self.roles, self.simple_code], exi_file, logg=logg)
        self.layers[depth].append(exi_file)
        self.concepts.append(exi_file)
        
        uni_file = self.path/'uni_{}.lp'.format(depth)
        self.expandConceptSet('uni', [depth], [depth-2,self.roles, self.simple_code], uni_file, logg=logg)
        self.layers[depth].append(uni_file)
        self.concepts.append(uni_file)

    def prune_symbols(self, ctl, program, keep_prog):
        results = []
        ctl.load(str(self.prune))
        ctl.ground([program])
        self.compare.compare(ctl)
        ctl.ground([keep_prog])
        with ctl.solve(yield_ = True) as models:
            models = [model for model in models]
            if len(models) != 1:
                raise RuntimeError("Only one model expected, more than one found.")

            keep = filter(lambda symbol: symbol.name == "keep__", models[0].symbols(atoms=True))
            results = [symbol.arguments[0] for symbol in keep]
        #print("RESULTS: {}".format(len(results)))
        return results

    def filter_symbols(self, ctl, keep_prog):
        results = []
        ctl.load(str(self.prune))
        ctl.ground([keep_prog])
        with ctl.solve(yield_ = True) as models:
            models = [model for model in models]
            if len(models) != 1:
                raise RuntimeError("Only one model expected, more than one found.")

            keep = filter(lambda symbol: symbol.name == "keep__", models[0].symbols(atoms=True))
            results = [symbol.arguments[0] for symbol in keep]
        #print("RESULTS: {}".format(len(results)))
        return results

    def simplifySample(self, out_file):
        ctl = clingo.Control()
        self.groundLayer(ctl, self.sample_code)
        symbols = self.filter_symbols(ctl, self.simplify)
        ModelUtil(symbols).write(str(out_file))

    def expandGrammar(self, max_depth, logg=False):
        self.createDir()
        simple_sample = self.path/'simple.lp'
        self.simplifySample(simple_sample)
        self.layers[self.simple_code] = [simple_sample]

        role_file = self.path/'roles.lp'
        self.expandConceptSet('roles', [], [self.sample_code], role_file, roles=True)
        self.layers[self.roles] = [role_file]
        
        for depth in range(1, max_depth+1):
            if logg: print('Depth {}:'.format(depth))
            self.expandLayer(depth, logg=logg)
            if logg:
                count = 0
                for conc in self.layers[depth]:
                    aux = self.countConcepts(conc)
                    count += aux
                print("Total {}: {} concepts.\n".format(depth, count))


    def countConcepts(self, file_name):
        count = 0
        with open(str(file_name), 'r') as f:
            for line in f:
                if re.match(r'conc\(.*?,\d+?\)\.', line):
                    count += 1
        return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sample', type=str, help='Sample file path')
    parser.add_argument('out_dir', type=str, help='Output folder')
    parser.add_argument('max_depth', type=int, help='Maximum concept depth')
    parser.add_argument('--fast', action='store_true', help='Prunning with cardinality')
    parser.add_argument('--std', action='store_true', help='Standard sound prunning')
    parser.add_argument('--mix', action='store_true', help='Cardinality + standard prunning')
    parser.add_argument('-d', '--debug',help="Print debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    args = parser.parse_args()
    

    logging.basicConfig(level=args.loglevel)
    if sum((int(b) for b in (args.fast, args.std, args.mix))) > 1:
        RuntimeError('More than one prunning type specified')

    comp_type = 'fast' if args.fast else None
    comp_type = 'standard' if args.std else comp_type
    comp_type = 'mixed' if args.mix else comp_type
    comp_type = 'standard' if comp_type is None else comp_type    
    print(comp_type)
    grammar = Grammar(args.sample,args.out_dir, comp_type=comp_type)
    grammar.expandGrammar(args.max_depth, logg=True)
