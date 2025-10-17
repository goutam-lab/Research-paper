# Standard Libraries
import time
import numpy as np
import pandas as pd
import math

# Third Party Imports
# Make sure you have rlwe_xmkckks installed (e.g., via pip install .)
from rlwe_xmkckks import RLWE, Rq

# This is a helper function from your test script to simulate client-side error generation
def discrete_gaussian(n, q, mean=0., std=1.):
    """
    Gaussian distribution to add errors for the partial decryption.
    Must have a larger standard deviation than the errors used for encryption.
    """
    coeffs = np.round(std * np.random.randn(n))
    return Rq(coeffs, q)

def generate_table1_encryption_decryption(param_sizes, num_clients=3):
    """
    Generates data for Table 1.
    Runs an experiment to measure encryption and decryption times for different parameter sizes.
    """
    print("🚀 Starting Experiment for Table 1: Encryption & Decryption Stats...")
    results = {
        "Parameters": [],
        "Encryption": [],
        "Decryption": []
    }
    
    # RLWE settings from your project
    q = 100_000_000_003  # prime number, q = 1 (mod 2n)   
    t = 200_000_001      # prime number, t < q
    std = 3              # standard deviation

    for size in param_sizes:
        # Determine the power-of-2 dimension 'n' required for the given parameter size
        n = 2 ** math.ceil(math.log2(size))
        print(f"\n--- Testing Parameters: {size} (n={n}) ---")
        
        results["Parameters"].append(f"{size//1000}k")

        # 1. SETUP RLWE INSTANCE AND KEYS
        rlwe = RLWE(n, q, t, std)
        rlwe.generate_vector_a()

        # Simulate clients and generate their keys
        keys = [rlwe.generate_keys() for _ in range(num_clients)]
        secret_keys = [k[0] for k in keys]
        public_keys = [k[1] for k in keys]

        # Create the shared public key for encryption
        sum_of_b_keys = sum([p[0] for p in public_keys], Rq(np.array([0]), q))
        allpub = (sum_of_b_keys, public_keys[0][1])

        # Create dummy plaintext data (simulating model weights)
        plaintexts = [Rq(np.random.randint(int(t/20), size=n), t) for _ in range(num_clients)]

        # 2. MEASURE ENCRYPTION TIME
        # We measure the time it takes for one client to encrypt its data
        enc_start_time = time.time()
        ciphertexts = [rlwe.encrypt(p, allpub) for p in plaintexts]
        enc_end_time = time.time()
        
        # The time for a single encryption is the total time divided by the number of clients
        single_encryption_time = (enc_end_time - enc_start_time) / num_clients
        results["Encryption"].append(round(single_encryption_time, 2))
        print(f"  Encryption (avg per client): {single_encryption_time:.2f}s")
        
        # This part is needed for the decryption step
        csum0 = sum([c[0] for c in ciphertexts], Rq(np.array([0]), q))
        csum1 = sum([c[1] for c in ciphertexts], Rq(np.array([0]), q))

        # 3. MEASURE DECRYPTION TIME
        dec_start_time = time.time()
        
        # Step 3a: All clients perform partial decryption in parallel
        decryption_shares = []
        for sk in secret_keys:
            error = discrete_gaussian(n, q, std=5) # Use larger std for decryption error
            d_share = rlwe.decrypt(csum1, sk, error)
            decryption_shares.append(d_share)
        
        # Step 3b: The server combines the shares to get the final result
        dec_sum = csum0 + sum(decryption_shares, Rq(np.array([0]), q))
        _ = Rq(dec_sum.poly.coeffs, t) # Final conversion to plaintext space
        
        dec_end_time = time.time()
        total_decryption_time = dec_end_time - dec_start_time
        results["Decryption"].append(round(total_decryption_time, 2))
        print(f"  Decryption (total): {total_decryption_time:.2f}s")

    return pd.DataFrame(results)


def generate_table2_aggregation(param_sizes, user_counts):
    """
    Generates data for Table 2.
    Runs an experiment to measure aggregation time for different parameter sizes and user counts.
    """
    print("\n🚀 Starting Experiment for Table 2: Aggregation Stats...")
    # Using a 2D list to store results, which will be converted to a DataFrame
    all_results = []
    
    # RLWE settings from your project
    q = 100_000_000_003
    t = 200_000_001
    std = 3

    for size in param_sizes:
        n = 2 ** math.ceil(math.log2(size))
        row_results = {"Parameters": f"{size//1000}k"}
        print(f"\n--- Testing Parameters: {size} (n={n}) ---")

        # Setup RLWE instance once per parameter size
        rlwe = RLWE(n, q, t, std)
        rlwe.generate_vector_a()
        
        for users in user_counts:
            # 1. SETUP: Generate keys and encrypt data for all users
            keys = [rlwe.generate_keys() for _ in range(users)]
            public_keys = [k[1] for k in keys]
            sum_of_b_keys = sum([p[0] for p in public_keys], Rq(np.array([0]), q))
            allpub = (sum_of_b_keys, public_keys[0][1])
            
            plaintexts = [Rq(np.random.randint(int(t/20), size=n), t) for _ in range(users)]
            ciphertexts = [rlwe.encrypt(p, allpub) for p in plaintexts]

            # 2. MEASURE AGGREGATION TIME
            agg_start_time = time.time()
            
            # The server sums the ciphertexts received from all clients
            _ = sum([c[0] for c in ciphertexts], Rq(np.array([0]), q))
            _ = sum([c[1] for c in ciphertexts], Rq(np.array([0]), q))
            
            agg_end_time = time.time()
            
            aggregation_time = agg_end_time - agg_start_time
            print(f"  Users: {users}, Aggregation Time: {aggregation_time:.2f}s")
            row_results[users] = round(aggregation_time, 2)
        
        all_results.append(row_results)

    # Convert results to a pandas DataFrame for clean printing
    df = pd.DataFrame(all_results)
    df = df.set_index("Parameters")
    return df

if __name__ == '__main__':
    # Define the parameters and user counts to test, matching your example image
    param_sizes_to_test = [50000, 100000, 150000, 200000, 250000]
    user_counts_to_test = [50, 100, 150, 200, 250]

    # --- Generate and Print Table 1 ---
    df_table1 = generate_table1_encryption_decryption(param_sizes_to_test)
    print("\n\n" + "="*60)
    print("Table 1: Time statistics for encryption and decryption (unit: s).")
    print("="*60)
    print(df_table1.to_string(index=False))
    
    # --- Generate and Print Table 2 ---
    df_table2 = generate_table2_aggregation(param_sizes_to_test, user_counts_to_test)
    print("\n\n" + "="*80)
    print("Table 2: Time statistics for aggregation at different numbers of users and parameters (unit: s).")
    print("="*80)
    print(df_table2)
    print("\n")