from __future__ import annotations
from csv import DictReader
from pathlib import Path
import random as rnd
import sys
from datetime import datetime
from os import listdir
from os.path import isfile, join
from common_functions.objects_proc import write_data_from_object_to_csv_file, load_data_from_dict_to_objects
import sys
import csv


class InitialDataService:
    """This class contains all required data for start simulation."""

    def __init__(self, num_monte_carlo_runs: int, max_flats_number: int, num_flats_step: int,
                 consumption_data_directory: Path, generation_data_filepath: Path,
                 start_stat_period: datetime, end_stat_period: datetime,
                 pool_sell_price: float, pool_buy_price: float, grid_sell_price: float, grid_buy_price: float,
                 coop_sell_price: float, price_per_kwp: float, kwp_installed: float, lat: float, long: float,
                 time_zone: str, building_altitude: int, location_name: str, tilt: float, azimuth: float,
                 pv_lifetime: int, flats_num_list: list[int] | None, random_seed: int | None = None):

        self.num_monte_carlo_runs = num_monte_carlo_runs
        self.max_flats_number = max_flats_number
        self.num_flats_step = num_flats_step
        self.consumption_data_directory = consumption_data_directory
        self.generation_data_filepath = generation_data_filepath
        self.start_stat_period = start_stat_period
        self.end_stat_period = end_stat_period
        self.pool_sell_price = pool_sell_price
        self.pool_buy_price = pool_buy_price
        self.grid_sell_price = grid_sell_price
        self.grid_buy_price = grid_buy_price
        self.coop_sell_price = coop_sell_price
        self.price_per_kwp = price_per_kwp
        self.kwp_installed = kwp_installed
        self.lat = lat
        self.long = long
        self.time_zone = time_zone
        self.building_altitude = building_altitude
        self.location_name = location_name
        self.tilt = tilt
        self.azimuth = azimuth
        self.pv_lifetime = pv_lifetime
        self.flat_sets: list[list[str]] = list()
        self.flats_num_list = flats_num_list
        self.random_seed = random_seed

        # init random flat sets for monte carlo simulations
        self.generate_sets_of_flats_for_every_monte_carlo_simulation()

        # Variables names to write in csv file for saving.
        self.write_attributes_names_list = [
            "num_monte_carlo_runs", "max_flats_number", "num_flats_step", "consumption_data_directory",
            "generation_data_filepath", "start_stat_period", "end_stat_period", "pool_sell_price", "pool_buy_price",
            "grid_sell_price", "grid_buy_price", "grid_buy_price", "coop_sell_price", "price_per_kwp", "kwp_installed",
            "lat", "long", "time_zone", "building_altitude", "location_name", "tilt", "azimuth", "pv_lifetime",
            "flat_sets", "flats_num_list","random_seed" ]

    def generate_sets_of_flats_for_every_monte_carlo_simulation(self, random_seed=None):
        if (self.random_seed is None) and (random_seed is None):
            self.random_seed = rnd.randrange(sys.maxsize)
        elif random_seed is not None:
            self.random_seed = random_seed
        rnd.seed(self.random_seed)

        directory_flat_ids = [flat_filepath.stem for flat_filepath in self.consumption_data_directory.iterdir()]



        if self.flats_num_list == None:
            """keep old set for each market size increase"""
            for monte_carlo_sim in range(self.num_monte_carlo_runs):
                simulation_flat_sets = [[]]
                directory_flat_ids_copy = directory_flat_ids.copy()
                for i_flats_in_sim in range(self.max_flats_number // self.num_flats_step):
                    random_set = rnd.sample(directory_flat_ids_copy, self.num_flats_step)
                    simulation_flat_sets.append(simulation_flat_sets[-1] + random_set)
                    for flat in random_set:
                        directory_flat_ids_copy.remove(flat)
                simulation_flat_sets = simulation_flat_sets[1:]
                self.flat_sets.append(simulation_flat_sets)##########

            """for random set in each market size increase"""
            # for monte_carlo_sim in range(self.num_monte_carlo_runs):
            #     simulation_flat_sets = list()
            #     for i_flats_in_sim in range(self.max_flats_number // self.num_flats_step):
            #         simulation_flat_sets.append(rnd.sample(directory_flat_ids, (i_flats_in_sim + 1) * self.num_flats_step))
            #     self.flat_sets.append(simulation_flat_sets)#######################################################################################


        else:
            """keep old set for each market size increase"""
            for monte_carlo_sim in range(self.num_monte_carlo_runs):
                simulation_flat_sets = [[]]
                directory_flat_ids_copy = directory_flat_ids.copy()
                for i in range(len(self.flats_num_list)):
                    if i == 0:
                        random_set = rnd.sample(directory_flat_ids_copy, self.flats_num_list[i])
                        simulation_flat_sets.append(simulation_flat_sets[-1] + random_set)
                        for flat in random_set:
                            directory_flat_ids_copy.remove(flat)
                    else:
                        random_set = rnd.sample(directory_flat_ids_copy, self.flats_num_list[i] - self.flats_num_list[i - 1])
                        simulation_flat_sets.append(simulation_flat_sets[-1] + random_set)
                        for flat in random_set:
                            directory_flat_ids_copy.remove(flat)
                simulation_flat_sets = simulation_flat_sets[1:]
                self.flat_sets.append(simulation_flat_sets)##########
            """for random set in each market size increase """
            # for monte_carlo_sim in range(self.num_monte_carlo_runs):
            #     simulation_flat_sets = list()
            #     for i in range(len(self.flats_num_list)):
            #         simulation_flat_sets.append(rnd.sample(directory_flat_ids, self.flats_num_list[i]))
            #     self.flat_sets.append(simulation_flat_sets)



    def write_data_in_csv_file(self, filepath: Path) -> bool:
        """This method write all simulation necessary data to csv filepath.
        Return true in success case"""
        write_data_from_object_to_csv_file(obj=self, attributes_names_list=self.write_attributes_names_list,
                                           filepath=filepath)
        return True

    def read_data_from_csv_file(self, filepath: Path) -> InitialDataService:
        """This method read initial data from csv filepath and
        fill self data variables with this data. Return
        self reference"""

        #### code to increase field size limit
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        #### code end
        with open(filepath, "r") as file:
            reader = DictReader(file)
            for row in reader:
                load_data_from_dict_to_objects(obj=self, load_data_dict=row)
                return self

    def __str__(self):
        res_str = f"InitialDataService object id = {id(self)}:\n"
        for attrinute_name in self.write_attributes_names_list:
            res_str += f"{attrinute_name}: {type(getattr(self, attrinute_name))} = {getattr(self, attrinute_name)}\n"
        return res_str
