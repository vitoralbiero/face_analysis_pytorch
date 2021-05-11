class EarlyStop:
    def __init__(self, patience=5, mode="max", threshold=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.mode = mode
        self.threshold = threshold
        self.val_acc = float("Inf")
        if mode == "max":
            self.val_acc *= -1

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc

        elif (val_acc < self.best_score + self.threshold and self.mode == "max") or (
            val_acc > self.best_score + self.threshold and self.mode == "min"
        ):
            self.counter += 1
            print(f"Early Stop counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
