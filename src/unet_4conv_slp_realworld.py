# Description:
#
# Following [Xiantao et al., 2020] approach: Test U-Net to reconstruct sparse inputs.
# Opposed to their net, only have 4 instead of 5 convolutional layers.
# Work with sea level pressure (slp) real world data.
# Omit last row to have dimension 72, that can be evenly divided by 2, three times, required for max pooling and later up sampling.
# 
# Apply sparsity mask: Randomly set inputs to zero. Random mask for different samples is not unique!
# And only use each sample once, no data augmentation in this experiment.

import os
from pathlib import Path
from json import dump, load

import numpy as np

from data import load_data, clone_data, create_sparsity_mask, split_and_scale_data
from models import build_unet_4conv


## Set parameters upfront:

# Data loading and preprocessing:
source = 'slp_realworld' # Choose either sea level pressure (slp) or sea surface temperature (sst) from real world or model data.
mask_type = 'variable' # Can have random sparsity mask, individually for each data sample ('variable'), or create only a single random mask, that is then applied to all samples identically ('fixed').
augmentation_factor = 3 # Number of times, each sample is to be cloned, keeping the original order.
train_val_split = 0.8 # Set rel. amount of samples used for training.
sparsity_all = [0.99, 0.95, 0.9, 0.75, 0.5] # Set array for desired sparsity of input samples: 0.9 means, that 90% of the values are missing.
scale_to = 'zero_one' # Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.

# Build and compile model:
CNN_filters = [64,128,256,512]# [2,4,8,16] # Number of filters. Originally, [Xintao et al.] used [64,128,256,512]. Training time 16 mins for single epoch!
CNN_kernel_size = 5 # Kernel size
learning_rate = 0.0005
loss_function = 'mse' 

# Train model:
epochs = 10
batch_size = 10

# Model configuration, to store results:
model_config = 'unet_4conv'

# Create directory to store results: Raise error, if path already exists, to avoid overwriting existing results. 
path = Path('GitHub/MarcoLandtHayen/reconstruct-sparse-inputs/results/'+model_config+'_'+source+'_'+mask_type+'_factor_'+str(augmentation_factor))
os.makedirs(path, exist_ok=False)
    
# Store parameters as json:
parameters = {
    'model_config': model_config,
    'source': source,
    'mask_type': mask_type,
    'augmentation_factor': augmentation_factor,
    'train_val_split': train_val_split,
    'sparsity_all': sparsity_all,
    'scale_to': scale_to,
    'CNN_filters': CNN_filters,
    'CNN_kernel_size': CNN_kernel_size,
    'learning_rate': learning_rate,
    'loss_function': loss_function,
    'epochs': epochs,
    'batch_size': batch_size
}

with open(path / 'parameters.json', 'w') as f:
    dump(parameters, f)

## Train models:

# Loop over array of desired sparsity:
for i in range(len(sparsity_all)):
    
    # Get current sparsity:
    sparsity = sparsity_all[i]
    
    # Print status:
    print("Sparsity: ", i+1, " of ", len(sparsity_all))
    
    # Create sub-directory to store results: Raise error, if path already exists, to avoid overwriting existing results.
    os.makedirs(path / 'sparsity_' f'{int(sparsity*100)}', exist_ok=False)
    
    # Load data:
    data = load_data(source=source)
    
    # Extend data, if desired:
    data = clone_data(data=data, augmentation_factor=augmentation_factor)
    
    # Create sparsity mask:
    sparsity_mask = create_sparsity_mask(data=data, sparsity=sparsity, mask_type=mask_type)
    
    # Store sparsity mask:
    np.save(path / 'sparsity_' f'{int(sparsity*100)}' / 'sparsity_mask.npy', sparsity_mask)

    # Use sparse data as inputs and complete data as targets. Split sparse and complete data into training and validation sets. 
    # Scale or normlalize data according to statistics obtained from only training data.
    train_input, val_input, train_target, val_target, train_min, train_max, train_mean, train_std = split_and_scale_data(
        data, 
        sparsity_mask, 
        train_val_split, 
        scale_to
    )
    
    # Build and compile U-Net model:
    model = build_unet_4conv(input_shape=(train_input.shape[1],train_input.shape[2],1),
                         CNN_filters=CNN_filters,
                         CNN_kernel_size=CNN_kernel_size,
                         learning_rate=learning_rate,
                         loss_function=loss_function,
                        )
    
    # Save untrained model:
    model.save(path / 'sparsity_' f'{int(sparsity*100)}' / f'epoch_{0}')
    
    # Initialize storage for training and validation loss:
    train_loss=[]
    val_loss=[]
    
    # Get model predictions on train and validation data FROM UNTRAINED MODEL!
    train_pred = model.predict(train_input)
    val_pred = model.predict(val_input)
    
    # Store loss on training and validation data:
    train_loss.append(np.mean((train_pred[:,:,:,0]-train_target)**2))
    val_loss.append(np.mean((val_pred[:,:,:,0]-val_target)**2))
    
    # Loop over number of training epochs:
    for j in range(epochs):
        
        # Print status:
        print("  Epoch: ", j+1, " of ", epochs)
        
        # Train model on sparse inputs with complete 2D fields as targets, for SINGLE epoch:
        history = model.fit(train_input, train_target, epochs=1, verbose=0, shuffle=True,
                            batch_size=batch_size, validation_data=(val_input, val_target))
        
        # Save trained model after current epoch:
        model.save(path / 'sparsity_' f'{int(sparsity*100)}' / f'epoch_{j+1}')
        
        # Get model predictions on train and validation data AFTER current epoch:
        train_pred = model.predict(train_input)
        val_pred = model.predict(val_input)

        # Store loss on training and validation data:
        train_loss.append(np.mean((train_pred[:,:,:,0]-train_target)**2))
        val_loss.append(np.mean((val_pred[:,:,:,0]-val_target)**2))  
        
    # Save loss:
    np.save(path / 'sparsity_' f'{int(sparsity*100)}' / 'train_loss.npy', train_loss)
    np.save(path / 'sparsity_' f'{int(sparsity*100)}' / 'val_loss.npy', val_loss)