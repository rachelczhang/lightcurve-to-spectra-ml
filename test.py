import regression
import benchmark_mlp_reg

def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params

num_channels = 32
input_size = 6249
cnn1 = regression.CNN1D(num_channels, 2, input_size)
cnn2 = regression.CNN1D(num_channels, 2, input_size)
mlp1 = benchmark_mlp_reg.MLP(4, 2)
mlp2 = benchmark_mlp_reg.MLP(4, 2)

num_params1 = count_parameters(cnn1)
print(f"Total number of parameters for CNN1: {num_params1}")

num_params2 = count_parameters(cnn2)
print(f"Total number of parameters for CNN2: {num_params2}")

num_params3 = count_parameters(mlp1)
print(f"Total number of parameters for MLP1: {num_params3}")

num_params4 = count_parameters(mlp2)
print(f"Total number of parameters for MLP2: {num_params4}")
