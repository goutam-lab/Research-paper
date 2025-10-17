# Standard Libraries
import importlib
import math
import os
import sys
import time
import warnings
from typing import List, Tuple

# Third Party Imports
import flwr as fl
import tensorflow as tf
from memory_profiler import memory_usage
from rlwe_xmkckks import RLWE, Rq
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Local Imports
# <<-- MODIFICATION: Import the new data loader
from new_data_loader import load_filtered_data_from_csv
from utils import set_initial_params, get_flat_weights, next_prime, set_model_params, pad_to_power_of_2, remove_padding, unflatten_weights, get_model_parameters

# Get absolute paths to let a user run the script from anywhere
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.basename(current_directory)
working_directory = os.getcwd()
# Add parent directory to Python's module search path
sys.path.append(os.path.join(current_directory, '..'))
# Compare paths
if current_directory == working_directory:
    from cnn import CNN
    import utils
else:
    # Add current directory to Python's module search path
    CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
    import utils


def discrete_gaussian(n, q, mean=0., std=1.):
    """
    Guassian distribution to add errors for the partial decryption
    Must have larger standard deviation than the errors used for encryption
    """
    coeffs = np.round(std * np.random.randn(n))
    return Rq(coeffs, q)


if __name__ == "__main__":
    # <<-- MODIFICATION: Load data from the clean CSV file
    # You can set a limit here for testing, e.g., limit=200
    X_train, X_test, y_train, y_test = load_filtered_data_from_csv(limit=None)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1)
    memory_usage_start = memory_usage()[0]

    # RLWE SETTINGS
    WEIGHT_DECIMALS = 8
    model = CNN(WEIGHT_DECIMALS)
    set_initial_params(model)
    params, _ = get_flat_weights(model)
    print(params[0:20])

    # find closest 2^x larger than number of weights
    num_weights = len(params)
    n = 2 ** math.ceil(math.log2(num_weights))
    print(f"n: {n}")

    # decide value range t of plaintext
    max_weight_value = 10**WEIGHT_DECIMALS
    num_clients = 8
    t = next_prime(num_clients * max_weight_value * 2)
    print(f"t: {t}")

    # decide value range q of encrypted plaintext
    q = next_prime(t * 50)
    print(f"q: {q}")

    # create rlwe instance for this client
    std = 3  # standard deviation of Gaussian distribution
    rlwe = RLWE(n, q, t, std)



    class CnnClient(fl.client.NumPyClient):
        """
        Custom class adapted for cnn models, extended from NumPyClient (base-class from Flower)
        """
        def __init__(self, rlwe_instance: RLWE, WEIGHT_DECIMALS: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rlwe = rlwe_instance
            self.allpub = None
            self.model_shape = None
            self.model_length = None
            self.flat_params = None
            self.WEIGHT_DECIMALS = WEIGHT_DECIMALS

            self.model = CNN(WEIGHT_DECIMALS)
            set_initial_params(self.model)

        def get_parameters(self, config):
            return get_model_parameters(self.model)

        def fit(self, parameters, config):
            set_model_params(self.model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # <<-- RECOMMENDATION: Increase epochs for better client-side learning
                self.model.fit(X_train, y_train, X_val, y_val, epochs=5)
            return get_model_parameters(self.model), len(X_train), {}
        
        def evaluate(self, parameters, config):
            set_model_params(self.model, parameters)
            loss, accuracy = self.model.evaluate(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

        # ... (rest of the encryption/decryption methods remain the same)
        def generate_pubkey(self, vector_a: List[int]) -> List[int]:
            vector_a = self.rlwe.list_to_poly(vector_a, "q")
            self.rlwe.set_vector_a(vector_a)
            (_, pub) = rlwe.generate_keys()
            return pub[0].poly_to_list()

        def store_aggregated_pubkey(self, allpub: List[int]) -> bool:
            aggregated_pubkey = self.rlwe.list_to_poly(allpub, "q")
            self.allpub = (aggregated_pubkey, self.rlwe.get_vector_a())
            return True

        def encrypt_parameters(self, request) -> Tuple[List[int], List[int]]:
            flattened_weights, self.model_shape = get_flat_weights(self.model)
            flattened_weights, self.model_length = pad_to_power_of_2(flattened_weights, self.rlwe.n, self.WEIGHT_DECIMALS)
            poly_weights = Rq(np.array(flattened_weights), self.rlwe.t)
            
            if request == "gradient":
                gradient = list(np.array(flattened_weights) - np.array(self.flat_params))
                poly_weights = Rq(np.array(gradient), self.rlwe.t)

            c0, c1 = self.rlwe.encrypt(poly_weights, self.allpub)
            c0 = list(c0.poly.coeffs)
            c1 = list(c1.poly.coeffs)
            return c0, c1

        def compute_decryption_share(self, csum1) -> List[int]:
            std = 5
            csum1_poly = self.rlwe.list_to_poly(csum1, "q")
            error = discrete_gaussian(n, q, 5)
            d1 = self.rlwe.decrypt(csum1_poly, self.rlwe.s, error)
            d1 = list(d1.poly.coeffs)
            return d1

        def receive_updated_weights(self, server_flat_weights) -> bool:
            server_flat_weights = list(np.array(server_flat_weights, dtype=np.float64))
            
            if self.flat_params is None:
                self.flat_params = server_flat_weights
            else:
                self.flat_params = list(np.array(self.flat_params) + np.array(server_flat_weights))

            server_flat_weights_unpadded = remove_padding(self.flat_params, self.model_length)
            server_weights = unflatten_weights(server_flat_weights_unpadded, self.model_shape)
            
            set_model_params(self.model, server_weights)
            # Evaluation metrics for client-side visibility
            loss, accuracy = self.model.evaluate(X_test, y_test)
            print(f"\nClient-side evaluation after update: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}\n")
            return True


    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=CnnClient(rlwe, WEIGHT_DECIMALS)
    )

    memory_usage_end = memory_usage()[0]
    memory_usage_total = memory_usage_end - memory_usage_start
    print("Memory usage:", memory_usage_total, "MiB")