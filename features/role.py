from features.model_util import SymbolSet

class Role:
    def __init__(self, symbols: SymbolSet, cost: int):
        self._symbols = symbols
        self._cost = cost
        role_id = symbols.get_atoms('roleId', 2)
        assert(len(role_id) == 1)
        self._id = role_id[0].arguments[1].number
        self._hash = None
        self._st_hash = {}
        self._calc_hash()

    @property
    def cost(self):
        return self._cost

    @property
    def id(self):
        return self._id

    @property
    def symbols(self):
        return self._symbols

    @property
    def role(self):
        return self._symbols.get_atoms('role', 2)

    @property
    def roleId(self):
        return self._symbols.get_atoms('roleId', 2)

    @property
    def belong(self):
        return [(sym.arguments[2], tuple(sym.arguments[0].arguments[0:2])) for sym in self.symbols.get_atoms('belong', 3)]

    def belong_state(self, state_id: Union[int, clingo.Symbol]):
        if isinstance(state_id, int):
            state_id = clingo.Number(state_id)
        return self._st_hash.get(SymbolHash(state_id), {})

    def _calc_hash(self):
        assert(self._hash == None)
        self._hash = 0
        for st, const in self.belong:
            self._hash += hash(str(st)+str(const))
            st = SymbolHash(st)
            if st not in self._st_hash:
                self._st_hash[st] = {}
            self._st_hash[st][SymbolHash(const)] = True
    
    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, ConceptObj):
            NotImplemented
        if hash(self) != hash(other): return False
        belong = self.belong
        if len(belong) != len(other.belong): return False
        for st, const in belong:
            if SymbolHash(const) not in other.belong_state(st):
                return False
        return True

    def to_concept(self, id):
        self._id = id
        with solver.create_solver() as ctl:
            ctl.load(Logic.grammarFile)
            ctl.addSymbols(self._symbols.get_all_atoms())
            ctl.ground([Logic.base, ('to_concept', [id])])
            sym = ctl.solve(solvekwargs=dict(yield_=True), symbolkwargs=dict(shown=True))
            self._symbols = SymbolSet(sym[0])