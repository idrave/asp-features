import clingo

def to_model_list(solve_handle):
    return [model for model in solve_handle]

def symbol_to_str(symbols):
    model_str = ""
    for symbol in symbols:
        model_str += str(symbol)+'.\n'
    return model_str

def load_symbols(file, program="base", prog_args=[]):
    ctl = clingo.Control()
    ctl.load(file)
    ctl.ground([(program, prog_args)])
    with ctl.solve(yield_ = True) as models:
        models = to_model_list(models)
        return models[0].symbols(atoms = True)

def filter_symbols(ctl, program="base", prog_args=[]):
    results = []
    ctl.ground([(program, prog_args)])
    with ctl.solve(yield_ = True) as models:
        for model in models:
            keep = filter(lambda symbol: symbol.name == "keep__", model.symbols(atoms=True))
            results.append([symbol.arguments[0] for symbol in keep])
    return results
'''
def filter_symbols(symbols, rules, program="base", prog_args=[]):
    ctl = clingo.Control()
    ctl.load(rules)
    ctl.add(program, prog_args, str(ModelUtil(symbols)))
    results = []
    ctl.ground([("base", [])])
    ctl.ground([(program, prog_args)])
    with ctl.solve(yield_ = True) as models:
        for model in models:
            keep = filter(lambda symbol: symbol.name == "keep__", model.symbols(atoms=True))
            results.append([symbol.arguments[0] for symbol in keep])
    return results
'''
def get_symbols(symbols, filter):
    model_str = []
    for symbol in symbols:
        if (symbol.name, len(symbol.arguments)) in filter:
            model_str.append(symbol)
    return model_str

def check_multiple(models):
    if len(models) == 0:
            raise RuntimeError("Not satisfiable!")
    elif len(models) > 1:
        raise RuntimeWarning("More than one model found!")

class ModelUtil:
    def __init__(self, symbols : list):
        self.symbols = symbols

    def __str__(self):
        return symbol_to_str(self.symbols)

    def get_symbols(self, filter=None):
        if filter is None: return self.symbols
        model_str = []
        for symbol in self.symbols:
            if (symbol.name, len(symbol.arguments)) in filter:
                model_str.append(symbol)
        return model_str

    def write(self, filename):
        with open(filename, 'w') as file:
            file.write(str(self))

    def count_symbol(self, symbol_compare):
        count = 0
        for symbol in self.symbols:
            if (symbol.name,len(symbol.arguments)) == symbol_compare:
                count += 1
        return count



