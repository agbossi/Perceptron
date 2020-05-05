class Normalizator:
    def __init__(self, max_value):
        self.max_value = max_value

    def normalize(self, value):
        return value/self.max_value

    def revert_normalization(self,value):
        return value * self.max_value
