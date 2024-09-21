import tinycudann as tcnn
import torch

# Define the encoding configuration
encoding_config = {
    "otype": "Grid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.0,
    # "hash": "CoherentPrime"
    "hash": "JumpConsistent"
}

# Create the encoding
n_input_dims = 3  # For 3D input
encoding = tcnn.Encoding(n_input_dims, encoding_config)

# Test the encoding with some random input
batch_size = 1024
input_tensor = torch.rand(batch_size, n_input_dims, device='cuda')

# Encode the input
encoded = encoding(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Encoded shape: {encoded.shape}")
print(f"Encoding output width: {encoding.n_output_dims}")