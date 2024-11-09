from datetime import datetime
from pathlib import Path
import pathlib
from csv import writer
import pandas as pd
import time
from common_functions.file_proc import get_approximate_csv_file_lines
from common_functions.proc_handle import update_progress_in_command_line, \
    update_progress_in_command_line_with_success
import os
import numpy as np


def delete_files_with_nan(directory, uniq_flats, relative_error_limit, df_date_range, start_stat_period: datetime, end_stat_period: datetime):
    """This function deletes all csv files that have more than 10% nan values
    and check for any dublicates in the datetime column"""

    # DataFrame to fill missing Date time Values
    df_date_range = pd.DataFrame({'DateTime': pd.date_range(start=start_stat_period, end=end_stat_period, freq='30min',
                                                            inclusive='left')})
    df_date_range.set_index('DateTime', inplace=True)

    print(df_date_range)


    print(f"\nValidate Nan values limit in flats consumption files from {directory}:")
    start_time = time.time()
    max_iterations = len(uniq_flats)

    for iteration_counter, curr_flat in enumerate(uniq_flats):

        file_path = directory.joinpath(str(curr_flat) + ".csv")

        df = pd.read_csv(directory.joinpath(str(curr_flat) + ".csv"))
        file_path.unlink()

        # Convert the "DateTime" column to a datetime type
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['DateTime'] = df['DateTime'].astype('datetime64[s]')

        df.set_index('DateTime', inplace=True)

        # If there are duplicate datetime values resample value using mean
        df_group = df.resample('30min').mean()

        # If there are missing Date time values resample them based on simulation datetime range
        df_group = df_group.reindex(df_date_range.index, fill_value=np.nan)

        # Set index as column again
        df_group.reset_index(inplace=True)
        df_group.rename(columns={"index": "DateTime"}, inplace=True)

        nan_count = df_group['consumption'].isnull().sum()
        total_count = len(df_group)


        # checks if there are more than relative_error_limit nan values
        if nan_count / total_count > relative_error_limit:
            print(f'File {curr_flat} unlinked because of relative error {nan_count / total_count}')
            continue

        #if sum(df['consumption']) < 100:
        #    print(f'File {curr_flat} unlinked because of total annual load of {sum(df["consumption"])}#####################')
        #    continue

        #if sum(df['consumption']) > 10000:
        #    print(f'File {curr_flat} unlinked because of total annual load of {sum(df["consumption"])}#####################')
        #    continue

        else:
            with open(directory.joinpath(str(curr_flat) + ".csv"), "w+", newline='') as file:
                w = writer(file, lineterminator='\r')

                headers = ["DateTime", "consumption"]
                w.writerows([headers])
            # kann sein dass er hier einfach unten ansetzt un nicht von neu schreibt
            df_group.to_csv(directory.joinpath(str(curr_flat) + ".csv"), mode='a', index=False, header=False)
            print(f'File {curr_flat} kept and potentially adjusted')


        # Update progress in the command line
        update_progress_in_command_line(percent_progess=100.0 * iteration_counter / max_iterations,
                                        time_worked_in_seconds=time.time() - start_time)

    # final progress in the command line update
    update_progress_in_command_line_with_success(time.time() - start_time)


def dataset_convert(init_dataset_filepath: Path, directory: Path, relative_error_limit: float,
                    start_stat_period: datetime, end_stat_period: datetime, chunksize: int) -> bool:
    """This function convert dataset from init_dataset_filepath
    to result_dataset_filepath with data structure required for
    simulation. Return True if result file was created"""

    uniq_flats = []

    # DataFrame to fill missing Date time Values
    df_date_range = pd.DataFrame({'DateTime': pd.date_range(start=start_stat_period, end=end_stat_period, freq='30min')})
    df_date_range.set_index('DateTime', inplace=True)

    # create directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)

    # define progress show in command line
    init_dataset_number_of_lines = get_approximate_csv_file_lines(init_dataset_filepath)
    read_lines = 0
    print(f"\nReading {init_dataset_filepath} to form flats consumption data files:")
    start_time = time.time()

    for df in pd.read_csv(init_dataset_filepath, na_values=["Null"], chunksize=chunksize):

        # data frame with conditions:
        # 1) flag "Std" in "stdorToU" field
        # 2) One or more data rows in datetime range
        df_condition = df[df["stdorToU"] == "Std"].loc[(df['DateTime'] >= str(start_stat_period)) &
                                                       (df['DateTime'] < str(end_stat_period))]

        # get dataframe of unique ids flat, that matched required conditions
        df_uniq_flat_id_in_chunk = pd.unique(df_condition['LCLid'].values.ravel())

        # add new unique flat
        for curr_flat in df_uniq_flat_id_in_chunk:
            if not (curr_flat in uniq_flats):
                uniq_flats.append(curr_flat)
                with open(directory.joinpath(str(curr_flat) + ".csv"), "w+", newline='') as file:
                    w = writer(file, lineterminator='\r')

                    headers = ["DateTime", "consumption"]
                    w.writerows([headers])

        # group condition dataframe by id for calc sum of "KWH/hh (per half hour) " for every flat
        df_group_by_id = df_condition.groupby("LCLid")

        for curr_flat in df_uniq_flat_id_in_chunk:
            # data frame group with current flat (get only "KWH/hh (per half hour) " values in group)
            df_group = df_group_by_id.get_group(curr_flat)[["DateTime", "KWH/hh (per half hour) "]]
            df_group['DateTime'] = df_group['DateTime'].astype('datetime64[s]')
            df_group.to_csv(directory.joinpath(str(curr_flat) + ".csv"), mode='a', index=False, header=False)

        # Update progress in the command line
        read_lines += chunksize
        update_progress_in_command_line(percent_progess=100.0 * read_lines / init_dataset_number_of_lines,
                                        time_worked_in_seconds=time.time() - start_time)

    # final progress in the command line update
    update_progress_in_command_line_with_success(time.time() - start_time)

    delete_files_with_nan(directory, uniq_flats, relative_error_limit, df_date_range, start_stat_period, end_stat_period)

    return True
