import clingo

def to_model_list(solve_handle):
    return [model for model in solve_handle]

def symbol_to_str(symbols):
    model_str = [str(symbol)+'.\n' for symbol in symbols]
    return model_str

class ModelUtil:
    def __init__(self, symbols : list):
        self.symbols = symbols

    def __str__(self):
        symbol_to_str(self.symbols)

    def get_symbols(self, filter=None):
        if filter is None: return self.symbols
        model_str = []
        for symbol in self.symbols:
            if (symbol.name, len(symbol.arguments)) in filter:
                model_str.append(symbol)
        return model_str

    def write(self, filename):
        with open(filename, 'w') as file:
            file.write(self)



