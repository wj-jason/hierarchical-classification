class Classifier:
    def __init__(self):
        raise NotImplementedError
    
    def resize(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def binary_predictions(self):
        raise NotImplementedError

    def to_next_classifier(self):
        raise NotImplementedError
