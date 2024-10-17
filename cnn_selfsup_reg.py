import torch
import torch.nn as nn
import numpy as np 
import wandb
import self_supervised
from torch.optim.lr_scheduler import ReduceLROnPlateau
from regression import read_hdf5_data, preprocess_data, create_dataloaders, test_loop, train_loop
import cnn_selfsup 
import run_cnn

torch.manual_seed(42)
np.random.seed(42)


class CNN1DFrozenEverything(nn.Module):
	def __init__(self, pretrained_encoder, nonpretrained_projector):
		super().__init__()
		self.encoder = pretrained_encoder
		# freeze the encoder
		for param in self.encoder.parameters():
			param.requires_grad = False

		#  # use a dummy input to dynamically determine the output dimension
		# dummy_input = torch.randn(1, 1, input_size, device=device)  # batch size of 1, 1 channel, and initial input size
		# dummy_output = self.encoder(dummy_input)
		# self.output_dim = dummy_output.numel() // dummy_output.shape[0]  # total number of features divided by batch size
		# add classification layers
		self.flatten = nn.Flatten()
		self.projector = nonpretrained_projector
		for param in self.projector.parameters():
			param.requires_grad = False
		# self.fc_layers = nn.Sequential(
		#     nn.Linear(self.output_dim, 128),
		#     nn.ReLU(),
		#     nn.Linear(128, 64),  
		#     nn.ReLU(),
		#     nn.Linear(64, output_size)
		# )

	def forward(self, x):
		x = self.encoder(x) 
		print('after encoder: ', x.shape)
		x = self.flatten(x)
		print('after flatten: ', x.shape)
		x = self.projector(x)
		print('after projector: ', x.shape)
		return x

if __name__ == '__main__':
	wandb.init(project="lightcurve-to-spectra-ml-self-supervised-reg", entity="rczhang")
	# data preprocessing steps
	power, Teff, logg, Msp, frequencies, tic_id = read_hdf5_data('/mnt/sdceph/users/rzhang/tessOregression.h5')  
	power_tensor, labels_tensor, norm_params = preprocess_data(power, Teff, logg, Msp, frequencies)
	learning_rate = 1e-5
	batch_size = 32
	train_dataloader, test_dataloader, train_dataset, test_dataset = create_dataloaders(power_tensor, labels_tensor, batch_size)
	num_channels = 32
	input_size = len(power.iloc[0]) 
	print('input size in cnn_selfsup_reg', input_size)
	# encoder_output_dim = num_channels * 2 * (input_size // 4)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	### manually modify size ###
	##### load in pretrained encoder #####
	pretrained_model = self_supervised.SimCLR(self_supervised.EncoderCNN1D(num_channels, input_size), 256)
	# # # load in 3 convolutional layers model
	# # pretrained_model.load_state_dict(torch.load('best_selfsup40_3conv.pth', map_location=device))
	# load in 2 convolutional layers model
	# pretrained_model.load_state_dict(torch.load('best_selfsup42_2conv.pth', map_location=device))
	# pretrained_model.load_state_dict(torch.load('best_selfsup44_embdim3.pth', map_location=device))
	# load in 1 conv layer model
	pretrained_model.load_state_dict(torch.load('best_selfsup46_1conv.pth'))
	pretrained_model.to(device)
	model = cnn_selfsup.CNN1DFrozenConv(pretrained_model.encoder, 2, input_size, device).to(device)

	# # keep running the best regression model
	# model = cnn_selfsup.CNN1DFrozenConv(self_supervised.SimCLR(self_supervised.EncoderCNN1D(num_channels, input_size), 256).encoder.to(device), 2, input_size, device).to(device)
	# model.load_state_dict(torch.load('best_selfsup_reg_deep-capybara-72.pth', map_location=device))
	
	##### test 08/02: use the best regression model conv_layers to be the pretrained_model.encoder #####
	# non_pretrained_model = run_cnn.CNN1D(num_channels, 3, input_size)
	# non_pretrained_model.load_state_dict(torch.load('best_reg_glad_moon_27.pth', map_location=device))
	# non_pretrained_model.to(device)
	# model = cnn_selfsup.CNN1DFrozenConv(non_pretrained_model.conv_layers, 3, input_size, device).to(device)

	#### test 08/06 pretrained encoder + not pretrained projector, run for 1 epoch #####
	# model = CNN1DFrozenEverything(non_pretrained_model.conv_layers, pretrained_model.projector).to(device)
	
	# loss function 
	best_loss = float('inf')
	loss_fn = nn.MSELoss().to(device)
	# t = -1
	# current_loss = test_loop(test_dataloader, model, loss_fn, t, norm_params)
	# optimizer only updates parameters of non-convolutional layers
	optimizer = torch.optim.Adam(model.fc_layers.parameters(), lr=learning_rate)
	patience = 300 # number of epochs to wait for improvement before stopping
	patience_counter = 0
	epochs = 10000
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loop(train_dataloader, model, loss_fn, optimizer, t, norm_params)
		current_loss = test_loop(test_dataloader, model, loss_fn, t, norm_params)
		scheduler.step(current_loss)
		if current_loss < best_loss:
			best_loss = current_loss
			patience_counter = 0
			torch.save(model.state_dict(), f"best_selfsup_reg_{wandb.run.name}.pth")
		else:
			patience_counter += 1
		if patience_counter >= patience:
			print('Early stopping triggered')
			break 
	# get_spectroscopic_lum_info(all_ys, all_preds)
	print("Done!")
	wandb.finish()