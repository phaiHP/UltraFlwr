import argparse
import warnings
from collections import OrderedDict
import torch
import flwr as fl
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME
from flwr.common import Context
from flwr.client import ClientApp
from FedYOLO.test.extract_final_save_from_client import extract_results_path
from FedYOLO.train.server_utils import write_yolo_config
# Import the function from strategies
from FedYOLO.train.strategies import get_section_parameters

warnings.filterwarnings("ignore", category=UserWarning)


NUM_CLIENTS = SERVER_CONFIG['max_num_clients']

def train(net, data_path, cid, strategy):
    net.train(data=data_path, epochs=YOLO_CONFIG['epochs'], workers=0, seed=cid+YOLO_CONFIG['seed_offset'], 
              batch=YOLO_CONFIG['batch_size'], project=strategy)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path, dataset_name, num_classes, strategy_name, task):
        # Initialize model config for this client
        yaml_path = f"{HOME}/FedYOLO/yolo_configs/yolo11n_{dataset_name}.yaml"
        # Load segmentation weights or detection config
        if task == "segment":
            self.net = YOLO("yolo11n-seg.pt")
        elif task == "pose":
            self.net = YOLO("yolo11n-pose.pt")
        elif task == "classify":
            self.net = YOLO("yolo11n-cls.pt")
        else:
            self.net = YOLO(yaml_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cid = cid
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.task = task

    def get_parameters(self, config=None):
        """Get relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()
        # Use the imported function
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

        # Define strategy groups (same as in set_parameters) - Corrected lists
        backbone_strategies = [
            'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian'
        ]
        neck_strategies = [
            'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian'
        ]
        head_strategies = [
            'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
            'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian'
        ]

        # Determine which parts to send based on strategy
        send_backbone = self.strategy_name in backbone_strategies
        send_neck = self.strategy_name in neck_strategies
        send_head = self.strategy_name in head_strategies

        # Get all parameters in consistent order (same as set_parameters)
        all_keys = sorted(current_state_dict.keys())
        relevant_parameters = []
        
        for k in all_keys:
            if (send_backbone and k in backbone_weights) or \
               (send_neck and k in neck_weights) or \
               (send_head and k in head_weights):
                relevant_parameters.append(current_state_dict[k].cpu().numpy())
        
        print(f"Client {self.cid} ({self.task}) sending {len(relevant_parameters)} parameters to server")
        return relevant_parameters

    def set_parameters(self, parameters):
        """Set relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

        # Define strategy groups - Corrected lists
        backbone_strategies = [
            'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian'
        ]
        neck_strategies = [
            'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian'
        ]
        head_strategies = [
            'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
            'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian'
        ]

        # Determine which parts to update based on strategy
        update_backbone = self.strategy_name in backbone_strategies
        update_neck = self.strategy_name in neck_strategies
        update_head = self.strategy_name in head_strategies

        # Get relevant keys in consistent order (same as server and get_parameters)
        relevant_keys = []
        for k in sorted(current_state_dict.keys()):
            if (update_backbone and k in backbone_weights) or \
               (update_neck and k in neck_weights) or \
               (update_head and k in head_weights):
                relevant_keys.append(k)

        print(f"Strategy: {self.strategy_name}")
        print(f"Parameters received: {len(parameters)}")
        print(f"Expected relevant parameters: {len(relevant_keys)}")
        print(f"Task: {self.task}")

        # Handle architecture mismatches for heterogeneous federated learning
        if len(parameters) != len(relevant_keys):
            print(f"Warning: Parameter count mismatch. Client architecture may differ from server.")
            print(f"This is expected for heterogeneous federated learning with different task types.")
            print(f"Attempting intelligent parameter matching for compatible layers...")
            
            # Advanced parameter matching by shape for cross-architecture compatibility
            updated_weights = {}
            server_params_used = [False] * len(parameters)
            matched_count = 0
            
            # First pass: try to match parameters by shape
            for k in relevant_keys:
                client_shape = current_state_dict[k].shape
                matched = False
                
                # Look for a server parameter with matching shape
                for i, server_param in enumerate(parameters):
                    if not server_params_used[i] and server_param.shape == client_shape:
                        updated_weights[k] = torch.tensor(server_param)
                        server_params_used[i] = True
                        matched = True
                        matched_count += 1
                        print(f"✓ Matched {k} with shape {client_shape}")
                        break
                
                if not matched:
                    # Keep original client parameter for unmatched layers
                    updated_weights[k] = current_state_dict[k]
                    print(f"✗ No match for {k} with shape {client_shape}, keeping original")
            
            print(f"Successfully matched {matched_count}/{len(relevant_keys)} parameters")
            
            if matched_count == 0:
                print("Warning: No parameters could be matched. Using original client parameters.")
            else:
                print(f"Federated learning proceeding with {matched_count} shared parameters")
        else:
            # Perfect match - proceed normally with shape validation
            params_dict = zip(relevant_keys, parameters)
            updated_weights = {}
            for k, v in params_dict:
                expected_shape = current_state_dict[k].shape
                param_array = v
                
                # Ensure parameter has correct shape
                if param_array.shape != expected_shape:
                    # Try to reshape if possible
                    if param_array.size == torch.Size(expected_shape).numel():
                        print(f"Reshaping parameter {k} from {param_array.shape} to {expected_shape}")
                        param_array = param_array.reshape(expected_shape)
                    else:
                        print(f"ERROR: Cannot reshape {k}: received shape {param_array.shape}, expected {expected_shape}")
                        print(f"       Received size: {param_array.size}, expected size: {torch.Size(expected_shape).numel()}")
                        # Keep original parameter as fallback
                        updated_weights[k] = current_state_dict[k]
                        continue
                
                updated_weights[k] = torch.tensor(param_array)

        # Load the updated parameters into the model, keeping existing weights for other parts
        final_state_dict = current_state_dict.copy()
        final_state_dict.update(updated_weights)
        
        self.net.model.load_state_dict(final_state_dict, strict=True) # Use strict=True if all expected keys are present

    def fit(self, parameters, config):
        if config["server_round"] != 1:
            del self.net
            torch.cuda.empty_cache()
            # get the path of the saved model weight
            logs_path = f"{HOME}/logs/client_{self.cid}_log_{self.dataset_name}_{self.strategy_name}.txt"
            weights_path = extract_results_path(logs_path)
            weights = f"{HOME}/{weights_path}/weights/best.pt"
            print(weights)

            self.net = YOLO(weights)

        self.set_parameters(parameters) # Now handles partial updates
        train(self.net, self.data_path, self.cid, f"logs/Ultralytics_logs/{self.strategy_name}_{self.dataset_name}_{self.cid}")
        # Return only the relevant parameters based on the strategy
        return self.get_parameters(), 10, {}
    
def client_fn(context: Context):
    from FedYOLO.config import CLIENT_CONFIG
    cid = context.node_config.get("cid", 0)
    cfg = CLIENT_CONFIG[cid]
    data_path = context.node_config.get("data_path", cfg["data_path"])
    dataset_name = cfg["dataset_name"]
    num_classes = cfg["num_classes"]
    task = context.node_config.get("task", cfg["task"])
    assert cid < NUM_CLIENTS
    return FlowerClient(cid, data_path, dataset_name, num_classes, SERVER_CONFIG['strategy'], task).to_client()

app = ClientApp(
    client_fn,
)