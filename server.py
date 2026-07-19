import copy


class Server:
    """Holds the global model and aggregates client updates with FedAvg."""

    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_models, client_sizes):
        global_weights = copy.deepcopy(client_models[0])
        total_samples = sum(client_sizes)

        for key in global_weights.keys():
            global_weights[key] = global_weights[key] * client_sizes[0]
            for i in range(1, len(client_models)):
                global_weights[key] = (
                    global_weights[key] + client_models[i][key] * client_sizes[i]
                )
            global_weights[key] = global_weights[key] / total_samples

        self.global_model.load_state_dict(global_weights)
