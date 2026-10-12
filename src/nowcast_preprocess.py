from pathlib import Path
import numpy as np

def file_name_to_datetime64(file_path):
    '''
    Convert string from file name into np.datetime64 to be assigned as new coordinate to dataset.
    Assumes filename is in the format ".../YYYYmmddTHHMM..."
    
    INPUTS
    file_path (PosixPath): path to the file

    OUTPUTS
    np.datetime64 corresponding to the file name
    '''
    time_str = file_path.stem
    
    year = time_str[0:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hour = time_str[9:11]
    minute = time_str[11:13]

    return np.datetime64(f'{year}-{month}-{day}T{hour}:{minute}')

def preprocess_nowcast_dataset(
    ds,
    f_dim_name, # name of forecast dim in the netcdf file
    new_f_dim_name, # name to use for forecast dim in xarray
    relative_leadtime=False):
    '''
    Adds a time dimension to the dataset using the filename.

    If relative_leadtime=True, converts times in the forecast_time dimension from absolute time values
    to times relative to "time", i.e. time since forecast intialised
    '''
    file_path = Path(ds.encoding["source"])
    t0 = file_name_to_datetime64(file_path)

    # Rename original time → forecast_time
    ds = ds.rename({f_dim_name: new_f_dim_name})

    if relative_leadtime:
        # Convert to lead time (timedelta)
        lead_time = ds[new_f_dim_name] - t0
        ds = ds.assign_coords({new_f_dim_name:lead_time})

    return ds.expand_dims(time=[t0])

