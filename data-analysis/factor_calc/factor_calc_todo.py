from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path
from csv import reader
import random as rnd
import pandas as pd
import numpy as np
import time
import sys
import os


class DeltaTimeData:
    """Class for storing information about delta time period in Analysis.
    start_time, delta_time, end_time: time limits of current period
    value: KWH for current period"""

    def __init__(self, start_time: datetime, delta_time: timedelta, value: float = 0.0):
        self.start_time = start_time
        self.delta_time = delta_time
        self.end_time: datetime = self.start_time + self.delta_time
        self.value: float = value


class FlatDataInTimePeriod:
    """Class for storing information about flat characteristics in current time period.
    id: flat id
    peak_load: max load of flat in current time period
    sum_load: sum of flat load in current time period"""

    def __init__(self, id_value: str, peak_load: float = 0.0, sum_load: float = 0.0):
        self.id = id_value
        self.peak_load = peak_load
        self.sum_load = sum_load


def init_empty_delta_time_list(start_time: datetime, delta_time: timedelta, end_time: datetime) -> list[DeltaTimeData]:
    """Function to initialize list of empty DeltaTimeData objects by time limits."""
    current_time = start_time
    delta_time_sum_load = list()
    while True:
        delta_time_sum_load.append(DeltaTimeData(start_time=current_time, delta_time=delta_time))
        current_time += delta_time
        if current_time >= end_time:
            break
    return delta_time_sum_load


class Simulation:
    """Class for storing information about simulation.
    flat_ids: list of flat ids in current simulation
    delta_time_sum_load: list of DeltaTimeData objects, that store sum
    of loads of flats from current simulation in current delta time period"""

    def __init__(self, flat_ids: list[str], start_time: datetime, delta_time: timedelta, end_time: datetime):
        self.flat_ids = flat_ids
        self.delta_time_sum_load: list[DeltaTimeData] = init_empty_delta_time_list(start_time, delta_time, end_time)


class Experiment:
    """Class for storing and generating simulations.
    simulations: dict of Simulation ojects with key equal number of flats in current simulation"""

    def __init__(self, flat_ids: list[str], n_flat_in_simulation_list: list[int],
                 start_time: datetime, delta_time: timedelta, end_time: datetime, is_empty_flat_ids: bool = False):
        self.simulations: dict[int: Simulation] = dict()
        self.init_simulations_plan(flat_ids, n_flat_in_simulation_list, start_time, delta_time, end_time,
                                   is_empty_flat_ids)

    def init_simulations_plan(self, flat_ids: list[str], n_flat_in_simulation_list: list[int], start_time: datetime,
                              delta_time: timedelta, end_time: datetime, is_empty_flat_ids: bool = False):
        """Method to initialize simulations by random flats from received flat_ids list if is_empty_flat_ids
        equal True or by empty flats if is_empty_flat_ids equal False."""
        self.simulations = dict()
        for n_flat in n_flat_in_simulation_list:
            if not is_empty_flat_ids:
                self.simulations[n_flat] = Simulation(flat_ids=rnd.sample(flat_ids, n_flat), start_time=start_time,
                                                      delta_time=delta_time, end_time=end_time)
            else:
                self.simulations[n_flat] = Simulation(flat_ids=list(), start_time=start_time,
                                                      delta_time=delta_time, end_time=end_time)


class Curve:
    """Class for storing curve data."""

    def __init__(self):
        self.points: list[list[float]] = list()
        self.color: str = "b"
        self.alpha: float = 1.0

    def append_point(self, point: list[float]):
        """Method to append point in points list"""
        self.points.append(point)

    def get_axis_values(self, axis: int):
        """Method return data of specific axis"""
        return [point[axis] for point in self.points]


class ResultGraph:
    """Class for storing curves graph data."""

    def __init__(self):
        self.curves: list[Curve] = list()


class Analysis:
    """Class for storing, handle, processing all task information.
    stat_filepath: Path objects of dataset file
    start_stat_period, delta_time_stat_period, end_stat_period: time limits of calculated period
    na_values: tuple of values that equal zero value in dataset
    chunksize: number of handled strings for one iteration
    relative_complete_data_threshold: thereshold of not zero data level flats in analys
    relative_n_flat_in_simulation_list: list of relative value of number of flats in every experiment
    flat_group_size: size of flat group in every experiment
    n_flat_in_simulation_list: list of number of flats in every experiment
    n_experiments: number of experiments in analysis
    max_delta_times_periods: number of delta time in time period
    directory: directory of searching and saving analysis files
    filepaths: dict of Path objects of analysis files
    ready_flat_ids: list of flat ids from current analysis
    experiments: list of Experiments objects of current analysis
    flats: dict of FlatDataInTimePeriod objects of current analysis with keys equal flat ids
    is_init: flag of initialization state of current analysis
    coincidence_factor: ResultGraph object of coincidence factor result
    load_factor: ResultGraph object of load factor result"""

    def __init__(self, stat_filepath: Path, start_stat_period: datetime, delta_time: timedelta,
                 end_stat_period: datetime, n_experiments: int,  relative_n_flat_in_simulation_list: list[float],
                 na_values: tuple[str], chunksize: int, relative_complete_data_threshold: float,
                 max_delta_times_periods: int, directory: Path):

        self.stat_filepath = stat_filepath
        self.start_stat_period = start_stat_period
        self.delta_time_stat_period = delta_time
        self.end_stat_period = end_stat_period
        self.na_values = na_values
        self.chunksize = chunksize
        self.relative_complete_data_threshold = relative_complete_data_threshold
        self.relative_n_flat_in_simulation_list = relative_n_flat_in_simulation_list
        self.flat_group_size: int = 0
        self.n_flat_in_simulation_list: list[int] = list()
        self.n_experiments = n_experiments
        self.max_delta_times_periods = max_delta_times_periods
        self.directory: Path = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.filepaths: dict[str: Path] = {
            "ready_flat_ids": Path(directory).joinpath("ready_flat_ids.csv"),
            "flats": Path(directory).joinpath("flats.csv"),
            "simulations_flat_ids": Path(directory).joinpath("simulations_flat_ids.csv"),
            "simulations_delta_time_sum_load": Path(directory).joinpath("simulations_delta_time_sum_load.csv"),
            "analysis_settings": Path(directory).joinpath("analysis_settings.csv"),
        }

        self.ready_flat_ids: list[str] = list()
        self.experiments: list[Experiment] = list()
        self.flats: dict[str: FlatDataInTimePeriod] = dict()

        self.coincidence_factor: Union[ResultGraph, None] = None
        self.load_factor: Union[ResultGraph, None] = None

        self.is_init: bool = False

    def run(self, initialize: bool = False):
        """Method to run analysis process"""
        if initialize or (not self.is_init):
            self.init_analysis()
        self.extract_analysis_data_from_stat_csv_file()

    def init_analysis(self):
        """Method to initialize analysis process"""
        if self.filepaths["ready_flat_ids"].is_file():
            self._read_ready_flat_ids_from_csv_file()
        else:
            self.extract_ready_flat_ids_from_stat_csv_file()
        self.init_variables()
        self.init_experiments()
        self.init_flats()
        self.is_init = True

    def init_flats(self):
        """Method to initialize flat list by current ready_flat_ids"""
        self.flats: dict[str: FlatDataInTimePeriod] = dict(
            (flat_id, FlatDataInTimePeriod(id_value=flat_id, peak_load=0.0, sum_load=0.0))
            for flat_id in self.ready_flat_ids)

    def init_experiments(self):
        """Method to initialize experiment list by current ready_flat_ids"""
        self.experiments = list()
        generate_flat_id_list = self.ready_flat_ids.copy()
        for _ in range(self.n_experiments):
            flat_ids = rnd.sample(generate_flat_id_list, self.flat_group_size)
            self.experiments.append(Experiment(
                flat_ids=flat_ids, n_flat_in_simulation_list=self.n_flat_in_simulation_list,
                start_time=self.start_stat_period, delta_time=self.delta_time_stat_period,
                end_time=self.end_stat_period))
            for flat_id in flat_ids:
                try:
                    generate_flat_id_list.remove(flat_id)
                except ValueError:
                    pass

    def init_variables(self):
        """Method to initialize flat_group_size and n_flat_in_simulation_list list values"""
        self.flat_group_size: int = len(self.ready_flat_ids) // self.n_experiments
        self.n_flat_in_simulation_list: list[int] = [int(rel * self.flat_group_size)
                                                     for rel in self.relative_n_flat_in_simulation_list]
        self.n_flat_in_simulation_list = [2,4,8,16,32,64]


    def simulation_range(self):
        """Generator for loop throw all simulations in Analysis"""
        experiments_iterator = self.experiments.__iter__()
        while True:
            try:
                simulations_itertor = experiments_iterator.__next__().simulations.values().__iter__()
                while True:
                    try:
                        current_simulation = simulations_itertor.__next__()
                        yield current_simulation.flat_ids, current_simulation.delta_time_sum_load
                    except StopIteration:
                        break
            except StopIteration:
                break

    @staticmethod
    def get_approximate_csv_file_lines(file_path: Path) -> int:
        """Method to get approximate number of lines in csv file"""
        max_check_lines: int = 1000
        read_lines: int = 0
        red_data: str = ""
        with open(file_path) as file:
            for _ in range(max_check_lines):
                red_data += file.readline()
                read_lines += 1
            max_check_lines = read_lines if read_lines < max_check_lines else max_check_lines
            n_lines = int(os.path.getsize(file_path) / (len(red_data.encode('utf-8')) / float(max_check_lines)))
        return n_lines

    def extract_ready_flat_ids_from_stat_csv_file(self):
        """Method to initialize analysis ready_flat_ids from dataset file or dataset directory"""
        # TODO: two cases - 1) stat_filepath - file, 2) stat_filepath - directory
        if self.stat_filepath.is_file():
            uniq_flats: dict[str: int] = dict()
            total_lines = Analysis.get_approximate_csv_file_lines(file_path=self.stat_filepath)
            total_lines_read = 0
            print(f"Reading {self.stat_filepath} to extract ready flat ids:\n")
            start_time = time.time()
            for df in pd.read_csv(self.stat_filepath, na_values=self.na_values, chunksize=self.chunksize):
                df = df[df["stdorToU"] == "Std"].loc[
                    (df['DateTime'] >= self.start_stat_period.strftime("%Y-%m-%d %H:%M:%S.%f")) &
                    (df['DateTime'] < self.end_stat_period.strftime("%Y-%m-%d %H:%M:%S.%f"))]

                df_uniq_flat_id_in_chunk = pd.unique(df['LCLid'].values.ravel())
                df_group_by_id = df.groupby("LCLid")

                for curr_flat in df_uniq_flat_id_in_chunk:
                    if not (curr_flat in uniq_flats.keys()):
                        uniq_flats[curr_flat] = 0
                    df_group_float = df_group_by_id.get_group(curr_flat)["KWH/hh (per half hour) "].astype(float)
                    uniq_flats[curr_flat] += df_group_float.loc[(df_group_float == 0)].shape[0]
                total_lines_read += self.chunksize
                sys.stdout.write('\r')
                sys.stdout.write(f"read_progress ~ " +
                                 f"{'{:.3f}'.format(float(100.0 * total_lines_read / total_lines).__round__(3))}" +
                                 r"%" + f"  {time.time() - start_time} seconds  wait...   ")
                sys.stdout.flush()

            sys.stdout.write('\r')
            sys.stdout.write(f"read_progress = {'{:.3f}'.format(100.0)}" + r"%" +
                             f"  total time = {time.time() - start_time} seconds  ready!" + "\n\n")
            sys.stdout.flush()

            self.ready_flat_ids = list()
            for key, value in uniq_flats.items():
                if (1.0 - value / float(self.max_delta_times_periods)) >= self.relative_complete_data_threshold:
                    self.ready_flat_ids.append(key)
        else:
            # TODO: get all filenames without extensions and fill flat ids list with it
            self.ready_flat_ids = list()
            filenames_without_extensions = [Path(file).stem for file in os.listdir(self.stat_filepath) if
                                            os.path.isfile(os.path.join(self.stat_filepath, file))]
            for filename in filenames_without_extensions:
                self.ready_flat_ids.append(filename)

    def extract_analysis_data_from_stat_csv_file(self):
        """Method to initialize analysis flats peak and sum load values in current time period,
        sum load in every simulation from dataset file"""
        # TODO: two cases - 1) stat_filepath - file, 2) stat_filepath - directory
        if self.stat_filepath.is_file():
            total_lines = Analysis.get_approximate_csv_file_lines(file_path=self.stat_filepath)
            total_lines_read = 0
            print(f"Reading {self.stat_filepath} to extract analysis data:\n")
            start_time = time.time()
            for df in pd.read_csv(self.stat_filepath, na_values=self.na_values, chunksize=self.chunksize):
                df = df[df["stdorToU"] == "Std"].loc[
                    (df['DateTime'] >= self.start_stat_period.strftime("%Y-%m-%d %H:%M:%S.%f")) &
                    (df['DateTime'] < self.end_stat_period.strftime("%Y-%m-%d %H:%M:%S.%f"))]

                df = df.loc[df['LCLid'].isin(self.ready_flat_ids)]

                df_uniq_flat_id_in_chunk = pd.unique(df['LCLid'].values.ravel())
                df = df.groupby("LCLid")
                for flat_id in df_uniq_flat_id_in_chunk:
                    df_curr_flat = df.get_group(flat_id)["KWH/hh (per half hour) "]
                    max_load_in_chunk = df_curr_flat.max()
                    self.flats[flat_id].sum_load += df_curr_flat.sum()
                    if max_load_in_chunk > self.flats[flat_id].peak_load:
                        self.flats[flat_id].peak_load = max_load_in_chunk

                for flat_ids, simulation_delta_time_sum_load in self.simulation_range():
                    for flat_id in df_uniq_flat_id_in_chunk:
                        if flat_id in flat_ids:
                            for curr_period in simulation_delta_time_sum_load:
                                curr_group = df.get_group(flat_id)
                                curr_period.value += curr_group.loc[
                                    (curr_group['DateTime'] >= curr_period.start_time.strftime("%Y-%m-%d %H:%M:%S.%f")) &
                                    (curr_group['DateTime'] < curr_period.end_time.strftime("%Y-%m-%d %H:%M:%S.%f"))][
                                    "KWH/hh (per half hour) "].sum()

                total_lines_read += self.chunksize
                sys.stdout.write('\r')
                sys.stdout.write(f"read_progress ~ " +
                                 f"{'{:.3f}'.format(float(100.0 * total_lines_read / total_lines).__round__(3))}" +
                                 r"%" + f"  {time.time() - start_time} seconds  wait...   ")
                sys.stdout.flush()

            sys.stdout.write('\r')
            sys.stdout.write(f"read_progress = {'{:.3f}'.format(100.0)}" + r"%" +
                             f"  total time = {time.time() - start_time} seconds  ready!" + "\n\n")
            sys.stdout.flush()
        else:
            # TODO: read every file in the directory
            for flat_id in self.ready_flat_ids:
                df = pd.read_csv(self.stat_filepath.joinpath(Path(str(flat_id) + ".csv")), na_values=self.na_values)
                df = df.loc[
                    (df['DateTime'] >= self.start_stat_period.strftime("%Y-%m-%d %H:%M:%S.%f")) &
                    (df['DateTime'] < self.end_stat_period.strftime("%Y-%m-%d %H:%M:%S.%f"))]
                df_consumption = df["consumption"]
                self.flats[flat_id].sum_load = df_consumption.sum()
                self.flats[flat_id].peak_load = df_consumption.max()
                for flat_ids, simulation_delta_time_sum_load in self.simulation_range():
                    if flat_id in flat_ids:
                        for curr_period in simulation_delta_time_sum_load:
                            curr_period.value += df.loc[
                                (df['DateTime'] >= curr_period.start_time.strftime("%Y-%m-%d %H:%M:%S.%f")) &
                                (df['DateTime'] < curr_period.end_time.strftime("%Y-%m-%d %H:%M:%S.%f"))][
                                "consumption"].sum()

    def _write_ready_flat_ids_to_csv_file(self):
        """Method to write flat ids in csv file"""
        with open(self.filepaths["ready_flat_ids"], "w") as file:
            for flat_id in self.ready_flat_ids:
                file.write(flat_id + "\n")

    def _read_ready_flat_ids_from_csv_file(self):
        """Method to read and initialize current analysis flat ids from csv file"""
        with open(self.filepaths["ready_flat_ids"], "r") as file:
            self.ready_flat_ids = list()
            for line in reader(file):
                self.ready_flat_ids.append(line[0])

    def _write_flats_in_csv_file(self):
        """Method to write flats data in csv file"""
        with open(self.filepaths["flats"], "w") as file:
            for flat_id, flat in self.flats.items():
                file.write(f"{flat_id},{flat.peak_load},{flat.sum_load}\n")

    def _read_flats_from_csv_file(self):
        """Method to read and initialize current analysis flats data from csv file"""
        with open(self.filepaths["flats"], "r") as file:
            self.flats = dict()
            for line in reader(file):
                self.flats[line[0]] = FlatDataInTimePeriod(id_value=line[0], peak_load=float(line[1]),
                                                           sum_load=float(line[2]))

    def _write_simulations_in_csv_file(self):
        """Method to write simulations data in csv files"""
        with open(self.filepaths["simulations_flat_ids"], "w") as file:
            for i_exp, experiment in enumerate(self.experiments):
                for key_sim, simulation in experiment.simulations.items():
                    for flat_id in simulation.flat_ids:
                        file.write(f"{str(i_exp)},{str(key_sim)},{flat_id}\n")
        with open(self.filepaths["simulations_delta_time_sum_load"], "w") as file:
            for i_exp, experiment in enumerate(self.experiments):
                for key_sim, simulation in experiment.simulations.items():
                    for i_delta_time_data, delta_time_data in enumerate(simulation.delta_time_sum_load):
                        file.write(f"{str(i_exp)},{str(key_sim)},{str(i_delta_time_data)},"
                                   f"{str(delta_time_data.start_time)},{str(delta_time_data.delta_time)},"
                                   f"{str(delta_time_data.end_time)},{str(delta_time_data.value)}\n")

    def init_empty_experiments(self):
        """Method to initialize current analysis simulations with empty values"""
        self.experiments = list()
        for _ in range(self.n_experiments):
            self.experiments.append(Experiment(
                flat_ids=list(), n_flat_in_simulation_list=self.n_flat_in_simulation_list,
                start_time=self.start_stat_period, delta_time=self.delta_time_stat_period,
                end_time=self.end_stat_period, is_empty_flat_ids=True))

    def _read_simulations_from_csv_file(self):
        """Method to read and initialize current analysis simulations data in csv files"""
        with open(self.filepaths["simulations_flat_ids"], "r") as file:
            self.init_empty_experiments()
            for line in reader(file):
                i_exp = int(line[0])
                key_sim = int(line[1])
                flat_id = line[2]
                self.experiments[i_exp].simulations[key_sim].flat_ids.append(flat_id)
        with open(self.filepaths["simulations_delta_time_sum_load"], "r") as file:
            for line in reader(file):
                i_exp = int(line[0])
                key_sim = int(line[1])
                i_delta_time_data = int(line[2])
                self.experiments[i_exp].simulations[key_sim].delta_time_sum_load[i_delta_time_data].start_time = \
                    datetime.strptime(line[3], "%Y-%m-%d %H:%M:%S")
                self.experiments[i_exp].simulations[key_sim].delta_time_sum_load[i_delta_time_data].delta_time = \
                    datetime.strptime(line[4], "%H:%M:%S")
                self.experiments[i_exp].simulations[key_sim].delta_time_sum_load[i_delta_time_data].end_time = \
                    datetime.strptime(line[5], "%Y-%m-%d %H:%M:%S")
                self.experiments[i_exp].simulations[key_sim].delta_time_sum_load[i_delta_time_data].value = \
                    float(line[6])

    def _write_analysis_settings_info_in_csv_file(self):
        """Method to write current analysis settings in csv files"""
        with open(self.filepaths["analysis_settings"], "w") as file:
            file.write(f"{self.start_stat_period},{self.delta_time_stat_period},{self.end_stat_period},"
                       f"{self.n_experiments},\"{self.relative_n_flat_in_simulation_list}\"\n")

    def _read_analysis_settings_info_from_csv_file(self):
        """Method to read and initialize current analysis settings from csv files"""
        with open(self.filepaths["analysis_settings"], "r") as file:
            for line in reader(file):
                self.start_stat_period = datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S")
                d_time = datetime.strptime(line[1], "%H:%M:%S")
                self.delta_time_stat_period = timedelta(hours=d_time.hour, minutes=d_time.minute, seconds=d_time.second)
                self.end_stat_period = datetime.strptime(line[2], "%Y-%m-%d %H:%M:%S")
                self.n_experiments = int(line[3])
                self.relative_n_flat_in_simulation_list = [float(curr) for curr in line[4][1:-1].split(",")]

    def save_analysis_data(self):
        """Method to save state and results of analysis in csv files"""
        self._write_analysis_settings_info_in_csv_file()
        self._write_ready_flat_ids_to_csv_file()
        self._write_flats_in_csv_file()
        self._write_simulations_in_csv_file()

    def load_analysis_data(self):
        """Method to load state and results of analysis from csv files"""
        self._read_analysis_settings_info_from_csv_file()
        self._read_ready_flat_ids_from_csv_file()
        self.init_variables()
        self._read_flats_from_csv_file()
        self._read_simulations_from_csv_file()

    def proc_data(self):
        """Method to calculate coincidence factor and load factor from data extracted from dataset"""
        self.coincidence_factor = ResultGraph()
        self.load_factor = ResultGraph()
        self.coincidence_factor.curves = [Curve() for _ in range(11)]
        self.load_factor.curves = [Curve() for _ in range(11)]
        percentil_values = [100.0 * (0.0 + i * 0.1) for i in range(11)]
        print(percentil_values)
        for i, n_flat in enumerate(self.n_flat_in_simulation_list):
            coincidence_factor_value_list = list()
            load_factor_value_list = list()
            for experiment in self.experiments:
                peak_of_sum_of_loads = max([val.value
                                            for val in experiment.simulations[n_flat].delta_time_sum_load])
                sum_of_loads_peak = sum([self.flats[flat_id].peak_load
                                         for flat_id in experiment.simulations[n_flat].flat_ids])
                aver_load = sum([self.flats[flat_id].sum_load
                                 for flat_id in experiment.simulations[n_flat].flat_ids]) / self.max_delta_times_periods
                coincidence_factor_value_list.append(peak_of_sum_of_loads / sum_of_loads_peak)
                load_factor_value_list.append(aver_load / peak_of_sum_of_loads)

            for j, percentile_value in enumerate(percentil_values):
                self.coincidence_factor.curves[j].append_point(
                    [n_flat, np.percentile(coincidence_factor_value_list, percentile_value)])
                self.load_factor.curves[j].append_point(
                    [n_flat, np.percentile(load_factor_value_list, percentile_value)])

        center_index = len(percentil_values) // 2
        print(center_index)

        i = 0
        for coincidence_curve, load_curve in zip(self.coincidence_factor.curves, self.load_factor.curves):
            if i == center_index:
                i += 1
                continue
            coincidence_curve.color = load_curve.color = "#C42724"
            coincidence_curve.alpha = load_curve.alpha = (0.5 - abs(len(percentil_values) // 2 - i) / 10.0) * 2.0

            if coincidence_curve.alpha == 0 or load_curve.alpha == 0:
                load_curve.alpha = coincidence_curve.alpha = 0.1
            i += 1
        self.coincidence_factor.curves[center_index].color = self.load_factor.curves[center_index].color = "black"
        self.coincidence_factor.curves[center_index].alpha = self.load_factor.curves[center_index].alpha = 1.0

    def plot(self):
        """Method to plot result graphs"""
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(12.5, 6.5, forward=True)
        fig.set_dpi(50)
        ax[0].grid()
        ax[1].grid()
        ax[0].set_ylim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.0])
        ax[0].set_ylabel('Daily Coincidence Factor')
        ax[1].set_ylabel('Daily Load Factor')
        ax[0].set_xlabel('Number of aggregated load profiles')
        ax[1].set_xlabel('Number of aggregated load profiles')
        center_index = len(self.coincidence_factor.curves) // 2
        for i in range(len(self.coincidence_factor.curves)):
            if i == center_index:
                continue
            if i < center_index:
                ax[0].fill_between(self.coincidence_factor.curves[i].get_axis_values(0),
                                   self.coincidence_factor.curves[i].get_axis_values(1),
                                   self.coincidence_factor.curves[i + 1].get_axis_values(1),
                                   color=self.coincidence_factor.curves[i].color,
                                   alpha=self.coincidence_factor.curves[i].alpha)
                ax[1].fill_between(self.load_factor.curves[i].get_axis_values(0),
                                   self.load_factor.curves[i].get_axis_values(1),
                                   self.load_factor.curves[i + 1].get_axis_values(1),
                                   color=self.load_factor.curves[i].color,
                                   alpha=self.load_factor.curves[i].alpha)
            elif i > center_index:
                ax[0].fill_between(self.coincidence_factor.curves[i].get_axis_values(0),
                                   self.coincidence_factor.curves[i].get_axis_values(1),
                                   self.coincidence_factor.curves[i - 1].get_axis_values(1),
                                   color=self.coincidence_factor.curves[i].color,
                                   alpha=self.coincidence_factor.curves[i].alpha)
                ax[1].fill_between(self.load_factor.curves[i].get_axis_values(0),
                                   self.load_factor.curves[i].get_axis_values(1),
                                   self.load_factor.curves[i - 1].get_axis_values(1),
                                   color=self.load_factor.curves[i].color,
                                   alpha=self.load_factor.curves[i].alpha)
        ax[0].plot(self.coincidence_factor.curves[center_index].get_axis_values(0),
                   self.coincidence_factor.curves[center_index].get_axis_values(1),
                   color=self.load_factor.curves[center_index].color, alpha=self.load_factor.curves[center_index].alpha)
        ax[0].scatter(self.coincidence_factor.curves[center_index].get_axis_values(0),
                      self.coincidence_factor.curves[center_index].get_axis_values(1),
                      color=self.load_factor.curves[center_index].color,
                      alpha=self.load_factor.curves[center_index].alpha)
        ax[1].plot(self.load_factor.curves[center_index].get_axis_values(0),
                   self.load_factor.curves[center_index].get_axis_values(1),
                   color=self.load_factor.curves[center_index].color, alpha=self.load_factor.curves[center_index].alpha)
        ax[1].scatter(self.load_factor.curves[center_index].get_axis_values(0),
                      self.load_factor.curves[center_index].get_axis_values(1),
                      color=self.load_factor.curves[center_index].color,
                      alpha=self.load_factor.curves[center_index].alpha)
        plt.show()


def main():
    # example
    # stat_filename = Path("D:\Projects\Ampigrid\load_profile_task\data\CC_LCL-FullData.csv")
    stat_folder = Path(r"C:\Projects\community\simulation_study_run\data\2013")
    work_directory = Path("data/test_day_analysis/test2/")
    na_values = ("Null",)
    chunksize = 2000000
    start_stat_period = datetime(year=2013, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    delta_time = timedelta(weeks=0, days=0, hours=0, minutes=30, seconds=0, milliseconds=0, microseconds=0)
    end_stat_period = datetime(year=2013, month=1, day=7, hour=0, minute=0, second=0, microsecond=0)
    max_delta_time_periods = (end_stat_period - start_stat_period) // delta_time
    relative_complete_data_threshold = 0.9
    realative_n_flat_in_simulation_list = [0.5, 0.8, 1.0] # [0.025, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_experiments = 100

    analysis = Analysis(stat_filepath=stat_folder, start_stat_period=start_stat_period, delta_time=delta_time,
                        end_stat_period=end_stat_period, n_experiments=n_experiments,
                        relative_n_flat_in_simulation_list=realative_n_flat_in_simulation_list, na_values=na_values,
                        chunksize=chunksize, relative_complete_data_threshold=relative_complete_data_threshold,
                        max_delta_times_periods=max_delta_time_periods, directory=work_directory)

    # This block for create new analysis
    analysis.init_analysis()
    analysis.run()
    analysis.save_analysis_data()
    analysis.load_analysis_data()
    analysis.proc_data()
    analysis.plot()

    # This block for load analysis
    # analysis.load_analysis_data()
    # analysis.proc_data()
    # analysis.plot()


if __name__ == "__main__":
    main()
