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

def datetime_to_lead_time(ds):
    '''
    Convert times in absoluate datetime formate to lead time ahead of the time that the forecast was made. Allows easier comparison with model forecasts.
    '''
    file_path = Path(ds.encoding["source"])
    t0 = file_name_to_datetime64(file_path)

    # Rename original time → forecast_time
    ds = ds.rename({'time': 'forecast_time'})

    # Convert to lead time (timedelta)
    lead_time = ds['forecast_time'] - t0

    return (
        ds
        .assign_coords(forecast_time=lead_time)
        .expand_dims(time=[t0])
    )