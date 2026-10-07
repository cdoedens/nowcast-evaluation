# Nowcast Evaluation

Read nowcasts from multiple models and evaluate them using the following methods:
    - Performance metric vs lead time, all nowcasts
    - Performance metric vs lead time, subset of nowcasts based off meteorology
    - Case studies

Performance metrics used:
    - RMSE
    - MAE
    - FSS
    - CRPS (ensemble nowcasts only)

Nowcast datasets should be saved as:
    - Netcdf file format
    - One forecast per file
    - UTC used for file naming
    - Directory structured as .../YEAR/MONTH/DAY/YYYYmmDDTHHMMSS.nc
        - e.g. a forecast initialised at 01:30 UTC 27th Feb 2025: .../2025/02/27/20250227T013000.nc
    - Time dimension, that indicates the forecast lead time from the initialisation time
    - Latitude and longitude dimension
    - Surface solar irradiance variable

Configurations for the evaluation that can be specified in a configs/config.yaml:
    - Lat and lon bounded region to perform the evaluation in
    - Max lead time to perform the evaluation for
    - Performance metrics to include
    - Names for each of the nowcasting models to evaluate, and the associated directories where their nowcasts are saved to 