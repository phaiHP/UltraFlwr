# config.py
import os
import yaml

def get_nc_from_yaml(yaml_path, task=None):
    """Get number of classes from data.yaml file and update nc dynamically for pose tasks."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Only update nc if the task is 'pose'
    if task == 'pose':
        num_classes = len(data.get('names', {}))
        data['nc'] = num_classes  # Update nc dynamically

        # Write the updated data back to the file
        with open(yaml_path, 'w') as file:
            yaml.safe_dump(data, file)
    else:
        num_classes = data.get('nc', None)

    return num_classes

def get_nc_from_classification_dataset(data_path):
    """Get number of classes from a classification dataset by checking train directory structure."""
    # For classification datasets, check if we have a partition structure
    if os.path.exists(data_path) and 'partitions' in os.listdir(data_path):
        # Look for the first client partition that has a train directory
        partitions_path = os.path.join(data_path, 'partitions')
        for partition in os.listdir(partitions_path):
            partition_path = os.path.join(partitions_path, partition)
            if os.path.isdir(partition_path):
                train_path = os.path.join(partition_path, 'train')
                if os.path.exists(train_path):
                    # Count the number of class directories
                    classes = [d for d in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, d))]
                    return len(classes)
    
    return 0

def generate_client_config(num_clients, dataset_path, client_tasks):
    """Dynamically generate client configuration for n clients with specific tasks."""
    if len(client_tasks) != num_clients:
        raise ValueError("Length of client_tasks must match num_clients")
    return {
        i: {
            'cid': i,
            'data_path': f"{dataset_path}/partitions/client_{i}/data.yaml",
            'task': client_tasks[i]  # Assign task based on client index
        }
        for i in range(num_clients)
    }

# Base Configuration
BASE = "/home/localssk23"
HOME = f"{BASE}/UltraFlwr"

# --- Multi-client, multi-task configuration ---

# Specify number of clients per dataset as a dictionary  
DETECTION_CLIENTS = {'redistributed_CAMMA_cholec': 3}         # dataset_name: num_clients
SEGMENTATION_CLIENTS = {}                   # No segmentation partitions available
POSE_CLIENTS = {}                  # Use client_0 partition  
CLASSIFICATION_CLIENTS = {}       # Use client_0 partition


client_specs = []
for ds, n_clients in DETECTION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({'dataset_name': ds, 'task': 'detect', 'client_idx': i})
for ds, n_clients in SEGMENTATION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({'dataset_name': ds, 'task': 'segment', 'client_idx': i})
for ds, n_clients in POSE_CLIENTS.items():
    for i in range(n_clients):
        # Fix: Pose dataset has test data in client_2, not client_0
        if ds == 'pose':
            client_idx = 2  # Use client_2 partition which has test data
        else:
            client_idx = i
        client_specs.append({'dataset_name': ds, 'task': 'pose', 'client_idx': client_idx})
for ds, n_clients in CLASSIFICATION_CLIENTS.items():
    for i in range(n_clients):
        client_specs.append({'dataset_name': ds, 'task': 'classify', 'client_idx': i}) 

NUM_CLIENTS = len(client_specs)

# Build CLIENT_CONFIG
CLIENT_CONFIG = {}
for cid, spec in enumerate(client_specs):
    dataset_name = spec['dataset_name']
    task = spec['task']
    client_idx = spec['client_idx']
    dataset_path = f"{HOME}/datasets/{dataset_name}"
    
    # For classification tasks, use the dataset directory directly
    if task == 'classify':
        data_path = f"{dataset_path}/partitions/client_{client_idx}"
        nc = get_nc_from_classification_dataset(dataset_path)
    else:
        data_yaml = f"{dataset_path}/data.yaml"
        data_path = f"{dataset_path}/partitions/client_{client_idx}/data.yaml"
        nc = get_nc_from_yaml(data_yaml, task=task)  # Pass the task to the function
        
    CLIENT_CONFIG[cid] = {
        'cid': cid,
        'dataset_name': dataset_name,
        'num_classes': nc,
        'data_path': data_path,
        'task': task,
    }

CLIENT_TASKS = [CLIENT_CONFIG[i]['task'] for i in range(NUM_CLIENTS)]
CLIENT_RATIOS = [1/NUM_CLIENTS] * NUM_CLIENTS

# For backward compatibility, set the first detection, segmentation, pose, and classification dataset names
DATASET_NAME = list(DETECTION_CLIENTS.keys())[0] if DETECTION_CLIENTS else ''
DATASET_NAME_SEG = list(SEGMENTATION_CLIENTS.keys())[0] if SEGMENTATION_CLIENTS else ''
DATASET_NAME_POSE = list(POSE_CLIENTS.keys())[0] if POSE_CLIENTS else ''
DATASET_NAME_CLS = list(CLASSIFICATION_CLIENTS.keys())[0] if CLASSIFICATION_CLIENTS else ''
DATASET_PATH = f'{HOME}/datasets/{DATASET_NAME}'
DATASET_PATH_SEG = f'{HOME}/datasets/{DATASET_NAME_SEG}'
DATASET_PATH_POSE = f'{HOME}/datasets/{DATASET_NAME_POSE}'
DATASET_PATH_CLS = f'{HOME}/datasets/{DATASET_NAME_CLS}'
DATA_YAML = f"{DATASET_PATH}/data.yaml"
DATA_YAML_SEG = f"{DATASET_PATH_SEG}/data.yaml"
DATA_YAML_POSE = f"{DATASET_PATH_POSE}/data.yaml"
DATA_YAML_CLS = f"{DATASET_PATH_CLS}/data.yaml"
NC = get_nc_from_yaml(DATA_YAML) if DATASET_NAME else None
NC_SEG = get_nc_from_yaml(DATA_YAML_SEG) if DATASET_NAME_SEG else None
NC_POSE = get_nc_from_yaml(DATA_YAML_POSE) if DATASET_NAME_POSE else None
NC_CLS = get_nc_from_classification_dataset(DATASET_PATH_CLS) if DATASET_NAME_CLS else None

SPLITS_CONFIG = {
    'dataset_name': DATASET_NAME,
    'num_classes': NC,
    'dataset': DATASET_PATH,
    'num_clients': NUM_CLIENTS,
    'ratio': CLIENT_RATIOS
}

SERVER_CONFIG = {
    'server_address': "0.0.0.0:8080",
    'rounds': 20,
    'sample_fraction': 1.0,
    'min_num_clients': NUM_CLIENTS,
    'max_num_clients': NUM_CLIENTS * 2,  # Adjusted based on number of clients
    'strategy': 'FedAvg',
}

YOLO_CONFIG = {
    'batch_size': 8,
    'epochs': 20,
    'seed_offset': 0
}
