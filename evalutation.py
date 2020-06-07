class Evaluation:

    def __init__(self):
        self.true = 0
        self.false = 0
        self.total = 0

    def __str__(self):
        return f'True: {self.true}, False: {self.false}'

    def __repr__(self):
        return self.__str__()

    def add_true(self):
        self.true += 1
        self.total += 1

    def add_false(self):
        self.false += 1
        self.total += 1

    def precision(self):
        return float(self.true / self.total)


