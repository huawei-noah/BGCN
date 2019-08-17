class flags:
    def __init__(self):
        # Hyper-parameter for the GNN model
        self.epochs = 300
        self.weight_samples_n = 60
        self.epoch_to_start_collect_weights = 240
        self.pretrain_n = 200
        self.hidden1 = 16
        self.learning_rate = 0.01
        self.dropout = 0.5
        self.weight_decay = 5e-4
        self.max_degree = 3
        self.features = 1

        # Hyper-parameter for the MMSBM model
        self.max_itr = 200
        self.delta = 0.0001
        self.batch_size = 500
        self.sampled_non_edges_ratio = 0.01
        self.gamma_scale = 0.001

