import xarray as xr
import numpy as np

def load_data(source="slp_realworld"):
    
    """Load and preprocess either sea level pressure (slp) or sea surface temperature fields as real world or model data.

    Parameters
    ----------
    source: str
        Specify desired data set. Defaults to "slp_realworld".
   
    Returns
    -------
    numpy.ndarray
        Preprocessed data set.
    """

    if source=="slp_realworld":
        
        # Open data set:
        slp_dataset=xr.open_dataset("GitHub/reconstruct-sparse-inputs/data/raw/pres.sfc.mon.mean.nc")

        # Start with raw slp fields as lat/lon grids in time, from 1948 to date:
        slp_fields = (
            slp_dataset.pres
            .sel(time=slice('1948-01-01', '2021-11-01'))
        )

        # Compute monthly climatology (here 1980 - 2009) for whole world:
        slp_climatology_fields = (
            slp_dataset.pres
            .sel(time=slice('1980-01-01','2009-12-01'))
            .groupby("time.month")
            .mean("time")
        )

        # Get slp anomaly fields by subtracting monthly climatology from raw slp fields:
        slp_anomaly_fields = slp_fields.groupby("time.month") - slp_climatology_fields

        # Remove last row (latidute), to have equal number of steps in latitude (=72). This serves as 'quick-and-dirty'
        # solution to avoid problems with UPSAMPLING in U-Net. There must be a more elegant way, take care of it later!
        slp_anomaly_fields = slp_anomaly_fields.values[:,:-1,:]
    
        # Return slp anomaly fields' values:
        return slp_anomaly_fields
    
    else:
        raise NotImplementedError('Unknown source file, unable to load data.')
        
def create_sparsity_mask(data, sparsity):
    
    """Create zero-inflated sparsity mask fitting complete data's dimensions.

    Parameters
    ----------
    data: numpy.ndarray
        Data set containing complete 2D fields.
    sparsity: float
        Desired sparsity, e.g. 0.9 means, that 90% of the values are set to zero.
   
    Returns
    -------
    numpy.ndarray
        Sparsity mask.
    """
    
    # Get sparsity mask from random uniform distribution in [0,1]:
    sparsity_mask = (np.random.uniform(low=0.0, high=1.0, size=data.shape)>sparsity).astype(int)

    return sparsity_mask

def split_and_scale_data(data, sparsity_mask, train_val_split, scale_to):
    
    """Apply sparsity mask to complete data. Then split data into training and validation sets and optionally scale or normalize values, 
    according to statistics obtained from training data.

    Parameters
    ----------
    data: numpy.ndarray
        Data set containing complete 2D fields, used as targets.
    sparsity_mask: numpy.ndarray
        Sparsity mask fitting complete data's dimensions.
    train_val_split: float
        Relative amount of training data.
    scale_to: string
        Specifies the desired scaling. Choose to scale inputs to [-1,1] ('one_one') or [0,1] ('zero_one') or 'norm' to normalize inputs or 'no' scaling.
   
    Returns
    -------
    train_input, val_input, train_target, val_target: numpy.ndarray
        Data sets containing training and validation inputs and targets, respectively.
    train_min, train_max, train_mean, train_std: float
        Statistics obtained from training data: Minimum, maximum, mean and standard deviation, respectively.    
    """
    
    # Get sparse data by applying given sparsity mask to complete data:
    data_sparse = data * sparsity_mask
    
    # Get number of train samples:
    n_train = int(len(data) * train_val_split)

    # Optionally scale inputs to [-1,1] or [0,1], according to min/max obtained from only train inputs. 
    # Or normalize inputs to have zero mean and unit variance. 

    # Remenber min/max used for scaling.
    train_min = np.min(data_sparse[:n_train])
    train_max = np.max(data_sparse[:n_train])

    # Remenber mean and std dev used for scaling.
    train_mean = np.mean(data_sparse[:n_train])
    train_std = np.std(data_sparse[:n_train])

    # Scale or normalize inputs depending on desired scaling parameter:
    if scale_to == 'one_one':
        # Scale inputs to [-1,1]:
        data_sparse_scaled = 2 * (data_sparse - train_min) / (train_max - train_min) - 1

    elif scale_to == 'zero_one':
        # Alternatively scale inputs to [0,1]
        data_sparse_scaled = (data_sparse - train_min) / (train_max - train_min)

    elif scale_to == 'norm':
        # Alternatively scale inputs to [0,1]
        data_sparse_scaled = (data_sparse - train_mean) / train_std

    ## Split inputs and targets:
    train_input = data_sparse_scaled[:n_train]
    val_input = data_sparse_scaled[n_train:]
    train_target = data[:n_train]
    val_target = data[n_train:]

    # Add dimension for number of channels, required for Conv2D:
    train_input = np.expand_dims(train_input, axis=-1)
    val_input = np.expand_dims(val_input, axis=-1)
    
    return train_input, val_input, train_target, val_target, train_min, train_max, train_mean, train_std