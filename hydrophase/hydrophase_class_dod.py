#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:45:38 2025

This script processes CMAC processed radar PPI data using the HydroPhase (hp) methodology. It reads CMAC files,
classifies the hydrometero ids using the PyART, CSU Summer and Winter classification schemes, and maps the results
to HydroPhase categories. The processed data is then saved to output netcdf files.
"""

#infiles = '/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradarcmacS2.c1/ppi/202305/gucxprecipradarcmacppiS2.c1.20230501.031140.nc'



import glob
import sys
import numpy as np
import os
import gc
import logging
import argparse
import subprocess
from datetime import datetime
from csu_radartools import csu_fhc
import pyart
import act

import shutil
from netCDF4 import Dataset, default_fillvals


from dask import delayed, compute


logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


# Constants
INPUT_FILE_PATTERN = 'gucxprecipradarcmacppiS2.c1'
OUTPUT_FILE_PATTERN = 'gucxprecipradarcmacppihpS2.c1'
LOG_FILE_NAME = "hp_processing.log"
FILL_VALUE = -32768
FILTER_FIELDS = ['corrected_reflectivity', 'corrected_differential_reflectivity',
                 'corrected_specific_diff_phase', 'RHOHV', 'sounding_temperature',
                 'hp_ssc', 'hp_fhc']
X_GRID_LIMITS = (-20_000., 20_000.)
Y_GRID_LIMITS = (-20_000., 20_000.)
Z_GRID_LIMITS = (500., 5_000.)
GRID_RESOLUTION = 250
ADDITIONAL_FIELDS = ["corrected_reflectivity", 'lowest_height'] # to be added to the output file

# Metadata constants
RADAR_NAME = 'gucxprecipradar'
ATTRIBUTIONS = (
    "This data is collected by the ARM Climate Research facility. Radar system is operated by the radar "
    "engineering team radar@arm.gov and the data is processed by the precipitation radar products team."
)
VAP_NAME = 'hp'
PROCESS_VERSION = "HP v1.0"
KNOWN_ISSUES = (
    "CMAC issues like, false phidp jumps, and some snow below melting layer, may affect classification. "
    "The Semisupervised method and fuzzy logic methods do not agree very well near melting layer."
)
INPUT_DATASTREAM = 'xprecipradarcmacppi'
DEVELOPERS = (
    "Bhupendra Raut, ANL; Robert Jackson, ANL; Zachary Sherman, ANL; Maxwell Grover, ANL; Joseph OBrien, ANL"
)
DATASTREAM = "gucxprecipradarhpS2.c1"
PLATFORM_ID = "xprecipradarhp"
DOD_VERSION = "xprecipradarhp-c1-1.0"
DOI = "10.5439/2530631"


FACILITY_ID = "S2",
DATA_LEVEL = "c1",

LOCATION_DESCRIPTION = "Gunnison, Colorado",

TRANSLATOR = "https://www.arm.gov/capabilities/instruments/xprecipradar",
MENTORS = "https://www.arm.gov/connect-with-arm/organization/instrument-mentors/list#xprecipradar",
SOURCE = "Colorado State University's X-Band Precipitation Radar (XPRECIPRADAR)",



# Mappings for CSU Summer, Winter, and Py-ART classifications to HydroPhase (hp)
csu_summer_to_hp = np.array([0, 1, 1, 2, 2, 4, 2, 3, 3, 3, 1])
csu_winter_to_hp = np.array([0, 2, 2, 2, 2, 4, 3, 1])
pyart_to_hp = np.array([0, 2, 2, 1, 3, 1, 2, 4, 4, 3])



def unprocessed_files(files, output_dir):
    unprocessed = []
    for file in files:
        basename = os.path.basename(file)
        output_file = basename.replace(INPUT_FILE_PATTERN, OUTPUT_FILE_PATTERN)
        output_path = os.path.join(output_dir, output_file)
        if not os.path.exists(output_path):
            unprocessed.append(file)
    return unprocessed

def read_radar(file, sweep=None):
    radar = pyart.io.read(file)
    if sweep is not None:
        radar = radar.extract_sweeps([sweep])
    return radar

def classify_hydrophase_csurt(radar, season='summer', band = "X"):
    """
    Run CSU hydrometeor classification (summer or winter) using corrected radar fields.
    
    Parameters:
        radar (pyart.core.Radar): The radar object.
        season (str): 'summer' or 'winter'.
        
    Returns:
        np.ndarray: HydroPhase classification (mapped from CSU output).
    """
    if season not in ['summer', 'winter']:
        raise ValueError("Season must be 'summer' or 'winter'")

    logging.info(f"Running CSU {season.capitalize()} classification using corrected fields")

    # Extract corrected radar moments (assumed present)
    dbz = radar.fields['corrected_reflectivity']['data']
    zdr = radar.fields['corrected_differential_reflectivity']['data']
    kdp = radar.fields['corrected_specific_diff_phase']['data']
    rho = radar.fields['RHOHV']['data']  # Not corrected
    rtemp = radar.fields['sounding_temperature']['data']

    if season == 'summer':
        scores = csu_fhc.csu_fhc_summer(
            dz=dbz,
            zdr=zdr,
            rho=rho,
            kdp=kdp,
            use_temp=True,
            band=band,
            T=rtemp
        )
        return csu_summer_to_hp[scores]

    else:  # winter
        snr = radar.fields['signal_to_noise_ratio']['data']
        #phidp = radar.fields['corrected_differential_phase']['data']
        azimuths = radar.azimuth['data']
        heights_km = radar.fields['height']['data'] / 1000.0

        hcawinter = csu_fhc.run_winter(
            dz=np.ma.masked_invalid(dbz),
            zdr=np.ma.masked_invalid(zdr),
            kdp=np.ma.masked_invalid(kdp),
            rho=np.ma.masked_invalid(rho),
            sn=np.ma.masked_invalid(snr),
            azimuths=azimuths,
            sn_thresh=-30,
            expected_ML=2.0,
            T=rtemp,
            heights=heights_km,
            nsect=36,
            scan_type=radar.scan_type,
            verbose=False,
            use_temp=True,
            band=band,
            return_scores=False
        )
        return csu_winter_to_hp[hcawinter]



def classify_hydrophase_pyart(radar):
    logging.info("Running Py-ART classification")
    radar.instrument_parameters['frequency'] = {'long_name': 'Radar frequency', 'units': 'Hz', 'data': [9.2e9]}
    hydromet_class = pyart.retrieve.hydroclass_semisupervised(
        radar,
        refl_field="corrected_reflectivity",
        zdr_field="corrected_differential_reflectivity",
        kdp_field="filtered_corrected_specific_diff_phase",
        rhv_field="RHOHV",
        temp_field="sounding_temperature",
    )
    return pyart_to_hp[hydromet_class['hydro']['data']]

def add_classification_to_radar(classified_data, radar, field_name, description):
    logging.info(f"Adding field: {field_name} to radar obj")
    fill_value = FILL_VALUE
    masked_data = np.ma.asanyarray(classified_data)
    masked_data.mask = masked_data == fill_value

    # dz_field = 'DBZ' if 'winter' in field_name else 'corrected_reflectivity' # keep it
    dz_field = 'corrected_reflectivity' # use corrected_reflectivity for both

    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        masked_data.mask = np.logical_or(masked_data.mask, radar.fields[dz_field]['data'].mask)
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {
        'data': masked_data,
        'units': '',
        'long_name': description,
        'standard_name': 'hydrometeor phase',
        '_FillValue': fill_value,
        "valid_min": 0,
        "valid_max": 4,
        "classification_description": "0: Unclassified, 1:Liquid, 2:Frozen, 3:High-Density Frozen, 4:Melting",
    }
    radar.add_field(field_name, field_dict, replace_existing=True)

def filter_fields(radar):
    radar.fields = {k: radar.fields[k] for k in FILTER_FIELDS if k in radar.fields}
    return radar


# taken from Max's script for gridding for Squire
# Setup a Helper Function and Configure our Grid
def compute_number_of_points(extent, resolution):
    """
    Create a helper function to determine number of points
    """
    return int((extent[1] - extent[0])/resolution)


def grid_radar(radar,
               x_grid_limits=X_GRID_LIMITS,
               y_grid_limits=Y_GRID_LIMITS,
               z_grid_limits=Z_GRID_LIMITS,
               grid_resolution=GRID_RESOLUTION,
               ):
    """
    Grid the radar using some provided parameters
    """
    
    x_grid_points = compute_number_of_points(x_grid_limits, grid_resolution)
    y_grid_points = compute_number_of_points(y_grid_limits, grid_resolution)
    z_grid_points = compute_number_of_points(z_grid_limits, grid_resolution)
    
    grid = pyart.map.grid_from_radars(radar,
                                      grid_shape=(z_grid_points,
                                                  y_grid_points,
                                                  x_grid_points),
                                      grid_limits=(z_grid_limits,
                                                   y_grid_limits,
                                                   x_grid_limits),
                                      method='nearest'
                                     )
    return grid.to_xarray()

def subset_lowest_vertical_level(ds, additional_fields=ADDITIONAL_FIELDS):
    """
    Filter the dataset based on the lowest vertical level (From Max)
    """
    hp_fields = [var for var in list(ds.variables) if "hp" in var] + additional_fields
    
    # Create a new 4-d height field.
    #ds["lowest_height"] = (ds.z * (ds[hp_fields[0]] / ds[hp_fields[0]])).fillna(5_000)
    # I changed the above lian but note sure if the earlier lines had any 
    # other reason to exist.
    
    valid_fields = ds[hp_fields[0]].where(np.isfinite(ds[hp_fields[0]]) &
                                          (ds[hp_fields[0]] != 0))
    
    ds["lowest_height"] = ds.z.where(valid_fields.notnull(), FILL_VALUE)
    
    
    # Find the minimum height index
    min_index = ds.lowest_height.argmin(dim='z',
                                        skipna=True)
    
    # Subset our hp fields based on this new index
    subset_ds = ds[hp_fields].isel(z=min_index)
    
    return subset_ds

def update_metadata(ds):
    # Update available fields
    available_fields = list(ds.data_vars.keys())
    ds.attrs['field_names'] = ', '.join(available_fields)
    
    # Update metadata
    ds.attrs['radar_name'] = RADAR_NAME
    current_time = subprocess.check_output(['date'], encoding='utf-8').strip()  # Use `date` for system time
    system_info = subprocess.check_output(['uname', '-n'], encoding='utf-8').strip()
    ds.attrs['history'] = f"created by Bhupendra Raut on {current_time} on: {system_info}"
    
    ds.attrs['attributions'] = ATTRIBUTIONS
    ds.attrs['vap_name'] = VAP_NAME
    ds.attrs['process_version'] = PROCESS_VERSION
    ds.attrs['known_issues'] = KNOWN_ISSUES
    ds.attrs['input_datastream'] = INPUT_DATASTREAM
    ds.attrs['developers'] = DEVELOPERS
    ds.attrs['datastream'] = DATASTREAM
    ds.attrs['platform_id'] = PLATFORM_ID
    ds.attrs['dod_version'] = DOD_VERSION
    ds.attrs['doi'] = DOI
    
    # Add the command line used to run the script
    ds.attrs['command_line'] = " ".join(sys.argv)

def make_squire_grid(radar):
    # Grid the radar ppi data
    ds = grid_radar(radar)
    # Subset the lowest vertical level
    ds = subset_lowest_vertical_level(ds)
    # update metadata
    update_metadata(ds)
    
    return ds


def write_dimensions(nc, ds, mapping):
    logging.info("Copying dimension values...")
    for dim_name in nc.dimensions:
        if dim_name not in nc.variables:
            continue
        if dim_name not in mapping:
            logging.warning(f"No mapping for dim '{dim_name}' → skipping.")
            continue
        ds_dim_name = mapping[dim_name]
        if ds_dim_name not in ds:
            logging.error(f"Dimension '{ds_dim_name}' not in dataset → skipping.")
            continue

        logging.info(f"  ↳ '{dim_name}' ← ds['{ds_dim_name}']")
        dim_data = ds[ds_dim_name].values

        if np.issubdtype(dim_data.dtype, np.datetime64) or "Datetime" in str(type(dim_data[0])):
            try:
                dim_data = np.array([
                    int((v - datetime(1970, 1, 1)).total_seconds())
                    for v in dim_data
                ], dtype='int64')
            except Exception as e:
                logging.error(f"Failed to convert time-like dim '{dim_name}': {e}")
                continue

        try:
            if len(dim_data) == 1:
                nc.variables[dim_name][0] = dim_data[0]
            else:
                nc.variables[dim_name][:] = dim_data
        except Exception as e:
                logging.error(f"Failed to write dimension '{dim_name}': {e}")


def write_variables(nc, ds, mapping):
    logging.info("Writing variable data...")
    for nc_var in nc.variables:
        if nc_var in ["time", "base_time", "time_offset"]:
            continue
        if nc_var not in mapping:
            continue
        ds_var = mapping[nc_var]
        if ds_var not in ds:
            logging.error(f"Variable '{ds_var}' not in ds")
            continue
        try:
            data = ds[ds_var]

            # Handle 2D lat/lon → 1D
            if nc_var == "x_lon" and data.ndim == 2:
                data = data.isel(y=0)
            elif nc_var == "y_lat" and data.ndim == 2:
                data = data.isel(x=0)

            fill_value = getattr(nc.variables[nc_var], '_FillValue',
                                 default_fillvals.get(nc.variables[nc_var].dtype.str[1:], np.nan))
            filled_data = data.fillna(fill_value).values

            if "time" in data.dims:
                nc.variables[nc_var][0] = filled_data[0]
            else:
                nc.variables[nc_var][:] = filled_data

            logging.info(f"Wrote '{nc_var}' from ds['{ds_var}']")

        except Exception as e:
            logging.error(f"Failed to write variable '{nc_var}': {e}")


def write_time_variables(nc, ds, mapping):
    logging.info("Writing time variables...")
    try:
        time_val = ds[mapping["time"]].values[0]
        if hasattr(time_val, "strftime"):
            time_val = datetime(time_val.year, time_val.month, time_val.day,
                                time_val.hour, time_val.minute, time_val.second)

        base_time_dt = datetime(time_val.year, time_val.month, time_val.day)
        base_time = int(base_time_dt.timestamp())
        time_offset = (time_val - base_time_dt).total_seconds()

        if "base_time" in nc.variables:
            nc.variables["base_time"][0] = base_time
        if "time_offset" in nc.variables:
            nc.variables["time_offset"][0] = time_offset
        if "time" in nc.variables:
            nc.variables["time"][0] = base_time + time_offset

        logging.info(f"  ↳ base_time: {base_time}")
        logging.info(f"  ↳ time_offset: {time_offset}")
        logging.info(f"  ↳ time: {base_time + time_offset}")

    except Exception as e:
        logging.error(f"Time write failed: {e}")
            

def update_global_attributes(nc, global_attrs):
    logging.info("Updating runtime DOD attributes in the NetCDF file...")

    for attr in nc.ncattrs():
        try:
            current_val = nc.getncattr(attr)

            # If value is not empty, keep it
            if current_val not in [None, '', ' ']:
                logging.info(f"  ↳ Keeping existing global attribute '{attr}' = '{current_val}'")
                continue

            # Value is empty
            if attr in global_attrs:
                new_val = global_attrs[attr]
                nc.setncattr(attr, new_val)
                logging.info(f"Global attribute '{attr}' was empty. Added to DOD at runtime with value: '{new_val}'")
            else:
                logging.error(f"Global attribute '{attr}' is empty, but no default value found in global_attrs – not updated.")

        except Exception as e:
            logging.error(f"Failed to process global attribute '{attr}': {e}")

    logging.info("Only empty DOD attributes were updated at runtime.")



def write_ds_to_dod_netcdf(ds_data, dod_file, output_path, mapping, global_attrs):
    """
    High-level wrapper to write data to a new NetCDF file based on a DOD template.
    """
    
    # copy dod template file as output file
    logging.info(f"Copying DOD template to output file: {output_path}")
    shutil.copyfile(dod_file, output_path)
    
    logging.info(f"Opening NetCDF file for update: {output_path}")
    nc =  Dataset(output_path, 'a')

    write_dimensions(nc, ds_data, mapping)
    write_variables(nc, ds_data, mapping)
    write_time_variables(nc, ds_data, mapping)
    update_global_attributes(nc, global_attrs)

    nc.close()
    logging.info(f"File written: {output_path}")


def process_file(file, season, output_dir, dod_file, mapping, global_attrs):
    """
    Process a single radar file through HydroPhase classification,
    gridding, lowest-level subsetting, and output to a DOD-formatted NetCDF file.

    Parameters:
        file (str): Path to input radar CMAC PPI file.
        season (str): 'summer' or 'winter' for classification scheme.
        output_dir (str): Directory to write the processed file.
        dod_file_path (str): Path to the DOD template NetCDF file.
        mapping (dict): Dictionary mapping NetCDF var names to dataset var names.
        global_attrs (dict): Global metadata attributes to embed in output file.
    """
    file_start_time = datetime.now()
    logging.info(f"Processing file: {file} with scheme: {season}")

    # Step 1: Read Radar
    radar = read_radar(file)

    # Step 2: Classify using CSU (seasonal)
    hp_fhc = classify_hydrophase_csurt(radar, season)
    add_classification_to_radar(hp_fhc, radar, 'hp_fhc', f'HydroPhase from CSU {season.title()}')

    # Step 3: Classify using PyART Semi-Supervised
    hp_ssc = classify_hydrophase_pyart(radar)
    add_classification_to_radar(hp_ssc, radar, 'hp_ssc', 'HydroPhase from Py-ART')

    # Step 4: Clean radar object (only required fields)
    filter_fields(radar)

    # Step 5: Grid radar → xarray dataset
    ds = grid_radar(radar)

    # Step 6: Subset to lowest vertical level
    ds = subset_lowest_vertical_level(ds)

    # Step 7: Write output NetCDF (copy DOD template first)
    output_filename = os.path.basename(file).replace(INPUT_FILE_PATTERN, OUTPUT_FILE_PATTERN)
    output_path = os.path.join(output_dir, output_filename)

    write_ds_to_dod_netcdf(
        ds_data=ds,
        dod_file=dod_file,
        output_path=output_path,
        mapping=mapping,
        global_attrs=global_attrs
    )

    logging.info(f"Saved processed file to: {output_path}")

    # Step 8: Cleanup
    del radar
    del ds
    gc.collect()

    file_end_time = datetime.now()
    logging.info(f"Finished file: {file}")
    logging.info(f"Time taken: {file_end_time - file_start_time}")



# %% Doing Dask parallel proc
@delayed
def safe_process_file(file, season, output_dir, dod_file, mapping, global_attrs):
    logging.info("running safe_process_file()")
    try:
        process_file(file, season, output_dir, dod_file, mapping, global_attrs)
        return f"✔ {os.path.basename(file)}"
    except Exception as e:
        logging.error(f"❌ Failed to process file: {file}")
        logging.error(str(e))
        return f"✖ {os.path.basename(file)}"

def main():
    start_time = datetime.now()
    logging.info(f"Script started at {start_time}")

    # CLI argument parser
    parser = argparse.ArgumentParser(description="Process radar files using HydroPhase classification.")

    # Define CLI args with defaults based on your script
    parser.add_argument("--year", type=int, default=2023, help="Year of the data (e.g., 2023)")
    parser.add_argument("--month", type=int, default=5, help="Month of the data (e.g., 5)")
    parser.add_argument("--season", type=str, default='summer', choices=['summer', 'winter'], help="CSU classification scheme to use (summer or winter)")
    parser.add_argument("--indir", type=str, default='/Users/bhupendra/projects/sail/data/test/', help="Input directory where CMAC files are located")
    parser.add_argument("--outdir", type=str, default='/Users/bhupendra/projects/sail/temp', help="Output directory where processed files will be saved")
    parser.add_argument("--dod_file", type=str, default='/Users/bhupendra/projects/sail/data/dod_v1_xprecipradarhp.nc', help="Path to DOD NetCDF template file")
    parser.add_argument("--rerun", action='store_true', help="If set, reprocess all files even if they were already processed.")

    args = parser.parse_args()

    # Unpack args
    year = args.year
    month = args.month
    season = args.season
    indir = args.indir
    outdir = args.outdir
    rerun = args.rerun
    dod_file_path = args.dod_file

    # Load files
    files = sorted(glob.glob(os.path.join(indir, "*.nc")))
    output_dir = outdir

    #if not rerun:
    #     files = unprocessed_files(files, output_dir)

    logging.info(f"Year: {year}, Month: {month}, Season: {season}")
    logging.info(f"Found {len(files)} unprocessed files.")

    if not files:
        logging.info("No files to process. Exiting.")
        return

    # Create mapping and global attributes
    mapping = {
        "hp_fhc": "hp_fhc",
        "hp_ssc": "hp_ssc",
        "corrected_reflectivity": "corrected_reflectivity",
        "lowest_height": "lowest_height",
        "z": "z",
        "x": "x",
        "y": "y",
        "time": "time",
        "x_lon": "lon",
        "y_lat": "lat"
    }

    global_attrs = {
        "command_line": " ".join(sys.argv) if hasattr(sys, "argv") else "hp_processing.py",
        "process_version": PROCESS_VERSION,
        "dod_version": DOD_VERSION,
        "site_id": RADAR_NAME[:3],
        "platform_id": PLATFORM_ID,
        "facility_id": FACILITY_ID,
        "data_level": DATA_LEVEL,
        "location_description": LOCATION_DESCRIPTION,
        "datastream": DATASTREAM,
        "input_datastreams": INPUT_DATASTREAM,
        "doi": DOI,
        "attributions": ATTRIBUTIONS,
        "known_issues": KNOWN_ISSUES,
        "developers": DEVELOPERS,
        "translator": TRANSLATOR,
        "mentors": MENTORS,
        "source": SOURCE,
        "history": f"File creation time {datetime.utcnow().isoformat()}Z"
    }

    # Ensure DOD file exists, or create it
    if not os.path.exists(dod_file_path):
        ds_dod = act.io.arm.create_ds_from_arm_dod('xprecipradarhp.c1',  
                                                   set_dims = {'time':0}, 
                                                   version='1.0',
                                                   scalar_fill_dim='time')
        ds_dod.to_netcdf(dod_file_path)

    # Use Dask Delayed to process all files in parallel
    delayed_tasks = [
        safe_process_file(
            file=file,
            season=season,
            output_dir=output_dir,
            dod_file=dod_file_path,
            mapping=mapping,
            global_attrs=global_attrs
        ) for file in files
    ]
    
    results = compute(*delayed_tasks, scheduler='processes')  # Or 'threads' if CPU-bound
    
    for r in results:
        logging.info(r)
        
    end_time = datetime.now()
    logging.info(f"Script finished at {end_time}")
    logging.info(f"Total time taken: {end_time - start_time}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # safe on all OSes
    main()














