# config.py
import yaml

def get_nc_from_yaml(yaml_path):
    """Get number of classes from data.yaml file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data.get('nc', None)

import os as os_module

def generate_client_config(num_clients, dataset_path):
    """Dynamically generate client configuration for n clients."""
    return {
        i: {
            'cid': i,
            'data_path': os_module.path.join(dataset_path, 'partitions', f'client_{i}', 'data.yaml')
        }
        for i in range(num_clients)
    }

import os

# Base Configuration
# AUTO: infer base path from this config file location (works on Windows/Linux).
# If you want to override, set BASE to the parent folder of the `UltraFlwr` repo.
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HOME = os.path.join(BASE, 'UltraFlwr')
DATASET_NAME = 'pest24'
DATASET_PATH = os.path.join(HOME, 'datasets', DATASET_NAME)
DATA_YAML = os.path.join(DATASET_PATH, 'data.yaml')
NC = get_nc_from_yaml(DATA_YAML)

# Number of clients can be easily modified here
NUM_CLIENTS = 2  # Change this to desired number of clients

# Generate equal ratios for n clients
CLIENT_RATIOS = [1/NUM_CLIENTS] * NUM_CLIENTS

SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'num_clients': NUM_CLIENTS,
    'ratio': CLIENT_RATIOS
}

# Dynamically generate client config
CLIENT_CONFIG = generate_client_config(NUM_CLIENTS, DATASET_PATH)

SERVER_CONFIG = {
    # Use localhost for client connections (0.0.0.0 is not routable for clients)
    # and it will still bind correctly for the server.
    'server_address': "127.0.0.1:8080",
    'rounds': 2,
    'sample_fraction': 1.0,
    'min_num_clients': NUM_CLIENTS,
    'max_num_clients': NUM_CLIENTS * 2,  # Adjusted based on number of clients
    'strategy': 'FedAvg',
}

YOLO_CONFIG = {
    'batch_size': 4,
    'epochs': 100,
}