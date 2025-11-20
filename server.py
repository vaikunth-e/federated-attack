import flwr as fl

def fit_config(rnd: int):
    return {"round": rnd}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    on_fit_config_fn=fit_config,   
)

if __name__ == "__main__":
    print("Starting Flower server on 127.0.0.1:8080")

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )
