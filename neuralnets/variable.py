class Variable:
    def __init__(self, v):
        self.data = v
        self.info = {}

    def __repr__(self):
        return str(self.data)
