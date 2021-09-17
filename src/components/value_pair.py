class ValuePair:
    def __init__(self, weight, raw_input):
        self.weight = weight
        self.raw_input = raw_input

    def get_weight(self):
        return self.weight

    def get_raw_input(self):
        return self.raw_input

    def set_weight(self, weight):
        self.weight = weight
