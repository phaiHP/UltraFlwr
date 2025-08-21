import os

import numpy as np

import flwr as fl
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from ultralytics import YOLO

from FedYOLO.train.server_utils import write_yolo_config
from FedYOLO.train.strategies import (
    FedAvg, FedMedian,
    FedHeadAvg, FedHeadMedian,
    FedNeckAvg, FedNeckMedian,
    FedBackboneAvg, FedBackboneMedian,
    FedNeckHeadAvg, FedNeckHeadMedian,
    FedBackboneHeadAvg, FedBackboneHeadMedian,
    FedBackboneNeckAvg, FedBackboneNeckMedian
)

from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME


def fit_config(server_round: int) -> dict:
    """Return training configuration for each round."""
    return {"epochs": YOLO_CONFIG["epochs"], 
            "server_round": server_round}


def get_parameters(net: YOLO, strategy_name: str) -> list[np.ndarray]:
    """Extract relevant model parameters from YOLO model based on the strategy."""
    from FedYOLO.train.strategies import get_section_parameters
    
    current_state_dict = net.model.state_dict()
    backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)
    
    # Define strategy parameter filters
    strategy_filters = {
        # FedAvg variations - all parameters
        'FedAvg': {'backbone': True, 'neck': True, 'head': True},
        'FedHeadAvg': {'backbone': False, 'neck': False, 'head': True},
        'FedNeckAvg': {'backbone': False, 'neck': True, 'head': False},
        'FedBackboneAvg': {'backbone': True, 'neck': False, 'head': False},
        'FedNeckHeadAvg': {'backbone': False, 'neck': True, 'head': True},
        'FedBackboneHeadAvg': {'backbone': True, 'neck': False, 'head': True},
        'FedBackboneNeckAvg': {'backbone': True, 'neck': True, 'head': False},
        
        # FedMedian variations - same filters as FedAvg
        'FedMedian': {'backbone': True, 'neck': True, 'head': True},
        'FedHeadMedian': {'backbone': False, 'neck': False, 'head': True},
        'FedNeckMedian': {'backbone': False, 'neck': True, 'head': False},
        'FedBackboneMedian': {'backbone': True, 'neck': False, 'head': False},
        'FedNeckHeadMedian': {'backbone': False, 'neck': True, 'head': True},
        'FedBackboneHeadMedian': {'backbone': True, 'neck': False, 'head': True},
        'FedBackboneNeckMedian': {'backbone': True, 'neck': True, 'head': False},
    }
    
    # Get filter for this strategy
    if strategy_name not in strategy_filters:
        # Default to all parameters if strategy not found
        filters = {'backbone': True, 'neck': True, 'head': True}
    else:
        filters = strategy_filters[strategy_name]
    
    # Get relevant parameters in sorted order (consistent with client)
    relevant_parameters = []
    for k in sorted(current_state_dict.keys()):
        if (filters['backbone'] and k in backbone_weights) or \
           (filters['neck'] and k in neck_weights) or \
           (filters['head'] and k in head_weights):
            relevant_parameters.append(current_state_dict[k].cpu().numpy())
    
    return relevant_parameters


def create_yolo_yaml(dataset_name: str, num_classes: int, task: str) -> YOLO:
    """Initialize YOLO model with the specified dataset, number of classes, and task."""
    write_yolo_config(dataset_name, num_classes)
    yaml_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml"
    if task == "segment":
        return YOLO("yolo11n-seg.pt")
    else:
        return YOLO(yaml_path)

def server_fn(context: Context):
    """Start the FL server with custom strategy."""
    # Make the directory HOME/FedYOLO/yolo_configs if it does not exist
    os.makedirs(f"{HOME}/FedYOLO/yolo_configs", exist_ok=True)

    # Use detection model as the base architecture for all clients
    # This ensures consistent parameter structure across different tasks
    from FedYOLO.config import CLIENT_TASKS
    
    # Always use detection model architecture for server initialization
    # This provides a common parameter structure that can be adapted by clients
    base_task = "detect"  # Use detection as base architecture
    
    # Create dataset specific YOLO yaml using detection task
    model = create_yolo_yaml(SPLITS_CONFIG["dataset_name"], SPLITS_CONFIG["num_classes"], base_task)

    # Initialize server side parameters based on the strategy
    strategy_name = SERVER_CONFIG["strategy"]
    
    # For heterogeneous client architectures, use flexible initialization
    print(f"Client tasks: {CLIENT_TASKS}")
    if len(set(CLIENT_TASKS)) > 1:
        print("Heterogeneous client architectures detected.")
        print("Using flexible parameter initialization to support cross-architecture federated learning.")
        # Initialize with server model parameters but allow client flexibility
        initial_parameters = ndarrays_to_parameters(get_parameters(model, strategy_name))
    else:
        # All clients use same architecture - standard initialization
        initial_parameters = ndarrays_to_parameters(get_parameters(model, strategy_name))

    # Map of available strategies
    strategies = {
        # FedAvg variations
        "FedAvg": FedAvg,
        "FedHeadAvg": FedHeadAvg,
        "FedNeckAvg": FedNeckAvg,
        "FedBackboneAvg": FedBackboneAvg,
        "FedNeckHeadAvg": FedNeckHeadAvg,
        "FedBackboneHeadAvg": FedBackboneHeadAvg,
        "FedBackboneNeckAvg": FedBackboneNeckAvg,
        
        # FedMedian variations
        "FedMedian": FedMedian,
        "FedHeadMedian": FedHeadMedian,
        "FedNeckMedian": FedNeckMedian,
        "FedBackboneMedian": FedBackboneMedian,
        "FedNeckHeadMedian": FedNeckHeadMedian,
        "FedBackboneHeadMedian": FedBackboneHeadMedian,
        "FedBackboneNeckMedian": FedBackboneNeckMedian
    }

    # Get the strategy class from config
    if strategy_name not in strategies:
        raise ValueError(
            f"Invalid strategy '{strategy_name}'. Available strategies: {', '.join(strategies.keys())}"
        )
    
    strategy_class = strategies[strategy_name]
    
    # Initialize the strategy
    strategy = strategy_class(
        fraction_fit=SERVER_CONFIG["sample_fraction"],
        min_fit_clients=SERVER_CONFIG["min_num_clients"],
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=SERVER_CONFIG["rounds"])

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
