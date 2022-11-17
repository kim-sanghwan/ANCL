class GridSearchConfig():
    def __init__(self):
        self.params = {
            'general': {
                'lr_first': [5e-1, 1e-1, 5e-2],
                'lr': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
                'lr_searches': [3],
                'lr_min': 1e-4,
                'lr_factor': 3,
                'lr_patience': 10,
                'clipping': 10000,
                'momentum': 0.9,
                'wd': 0.0002
            },

            'finetuning': {
            },
            'freezing': {
            },
            'joint': {
            },
            'ewc': {
                'lamb': 10000
            },
            'a-ewc': {
                'lamb': 10000,
                'lamb-a': 10
            },
            'mas': {
                'lamb': 50
            },
            'a-mas': {
                'lamb': 50,
                'lamb-a': 5
            },
            'lwf': {
                'lamb': 10,
                'T': 2
            },
            'a-lwf': {
                'lamb': 10,
                'lamb-a': 1,
                'T': 2
            },
            'lfl': {
                'lamb': 400
            },
            'a-lfl': {
                'lamb': 400,
                'lamb-a': 100
            },
            'dmc': {
            },
            'lwm': {
                'beta': 10,
                'gamma': 1.0
            }
        }
        self.current_lr = self.params['general']['lr'][0]
        self.current_tradeoff = 0

    def get_params(self, approach):
        return self.params[approach]
