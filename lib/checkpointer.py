

class Checkpointer:
    '''
    Accept a score for each epoch, record best_score, save checkpoint when
    best_score is updated, check early stopping status. 

    >>> save_fn = lambda score: deepcopy(model.state_dict())
    >>> load_fn = lambda best_checkpoint: model.load_state_dict(best_checkpoint)
    >>> checkpointer = Checkpointer(
            patience=10, delta=0, save_fn=save_fn, load_fn=load_fn)
    >>> for epoch in range(epochs):
            if checkpointer.early_stop():
                break
            loss = train_one_epoch(model, train_loader)
            score = eval_one_epoch(model, valid_loader)
            checkpointer.update(score)
        checkpointer.load()
    '''
    def default_save_fn(score):
        pass

    def default_load_fn(best_checkpoint):
        pass

    def __init__(
        self, patience=10, delta=0,
        save_every_epoch=False,
        save_fn=default_save_fn,
        load_fn=default_load_fn
    ):
        self.counter = 0
        self.patience = patience
        self.delta = delta
        self.save_every_epoch = save_every_epoch
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.best_score = None
        self.checkpoints = []
        self.best_checkpoint = None

        self._early_stop = False

    def update(self, score, save_fn=None, *args, **kwargs):
        save_fn = save_fn or self.save_fn

        if self.save_every_epoch:
            self.checkpoints.append(save_fn(score, *args, **kwargs))

        if (self.best_score is None) or (score > self.best_score + self.delta):
            self.best_score = score
            self.counter = 0

            if self.save_every_epoch:
                self.best_checkpoint = self.checkpoints[-1]
            else:
                self.best_checkpoint = save_fn(score, *args, **kwargs)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._early_stop = True

    def load(self, load_fn=None, *args, **kwargs):
        load_fn = load_fn or self.load_fn
        return load_fn(self.best_checkpoint, *args, **kwargs)

    def early_stop(self):
        return self._early_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.checkpoints = []
        self.best_checkpoint = None
        self._early_stop = False
