from __future__ import annotations
from itertools import chain
from pathlib import Path
from csv import DictReader, writer
import pandas as pd
import numpy as np
import time
import math
from numba import njit, typed
from datetime import datetime
from common_functions.objects_proc import load_data_from_dict_to_objects
from common_functions.proc_handle import update_progress_in_command_line, \
    update_progress_in_command_line_with_success


@njit
def iteration_raw(flats_consumption, p_pool_buy, p_pool_sell, p_grid_buy, p_grid_sell, generation):

    flat_generation = generation / len(flats_consumption)
    flats_requests = [flat_generation - curr_flat_consumption for curr_flat_consumption in flats_consumption]

    pool_buy_request = 0.0
    pool_sell_request = 0.0

    for i in range(len(flats_consumption)):
        if flat_generation < flats_consumption[i]:
            pool_buy_request += flats_requests[i]
        else:
            pool_sell_request += flats_requests[i]

    buy_request_flats_ids = list()
    buy_request_flats = list()
    sell_request_flats_ids = list()
    sell_request_flats = list()

    flat_money_change = [0.0 for _ in range(len(flats_consumption))]
    pool_sell = [0.0 for _ in range(len(flats_consumption))]
    pool_buy = [0.0 for _ in range(len(flats_consumption))]
    grid_sell = [0.0 for _ in range(len(flats_consumption))]
    pv_consume = [0.0 for _ in range(len(flats_consumption))]
    grid_buy = [0.0 for _ in range(len(flats_consumption))]

    for i in range(len(flats_consumption)):
        if flat_generation >= flats_consumption[i]:
            pv_consume[i] = flats_consumption[i]
        else:
            pv_consume[i] = flat_generation

    for i, flat_requests in enumerate(flats_requests):
        if flat_requests < 0:
            buy_request_flats.append(flat_requests)
            buy_request_flats_ids.append(i)
        else:
            sell_request_flats.append(flat_requests)
            sell_request_flats_ids.append(i)

    if pool_buy_request * (-1) > pool_sell_request:
        number_of_buy_request = len(buy_request_flats)
        # buy_request_flats_new_indices = sorted(range(len(buy_request_flats)),
        # key=buy_request_flats.__getitem__, reverse=True)
        buy_request_flats_new_indices = np.flip(np.argsort(np.array(buy_request_flats)))
        buy_request_flats = [buy_request_flats[i] for i in buy_request_flats_new_indices]
        every_flat_pool_max_energy_buy = pool_sell_request / number_of_buy_request
        every_flat_pool_max_money_buy = every_flat_pool_max_energy_buy * p_pool_buy
        for flat_id in sell_request_flats_ids:
            flat_money_change[flat_id] = flats_requests[flat_id] * p_pool_sell
            pool_sell[flat_id] = flats_requests[flat_id]
        for i in range(len(buy_request_flats)):
            if buy_request_flats[i] * (-1) <= every_flat_pool_max_energy_buy:
                flat_money_change[
                    buy_request_flats_ids[buy_request_flats_new_indices[i]]] = buy_request_flats[i] * p_pool_buy
                pool_buy[buy_request_flats_ids[buy_request_flats_new_indices[i]]] = -buy_request_flats[i]
                pool_sell_request += buy_request_flats[i]
                number_of_buy_request -= 1
            else:
                flat_energy_buy_on_grid = (-1) * buy_request_flats[i] - every_flat_pool_max_energy_buy
                flat_money_buy_on_grid = flat_energy_buy_on_grid * p_grid_buy
                flat_money_change[
                    buy_request_flats_ids[buy_request_flats_new_indices[i]]] = -flat_money_buy_on_grid \
                                                                               - every_flat_pool_max_money_buy
                pool_buy[buy_request_flats_ids[buy_request_flats_new_indices[i]]] = every_flat_pool_max_energy_buy
                grid_buy[buy_request_flats_ids[buy_request_flats_new_indices[i]]] = flat_energy_buy_on_grid
                pool_sell_request -= every_flat_pool_max_energy_buy
                number_of_buy_request -= 1
            if number_of_buy_request != 0:
                every_flat_pool_max_energy_buy = pool_sell_request / number_of_buy_request
                every_flat_pool_max_money_buy = every_flat_pool_max_energy_buy * p_pool_buy
    else:
        number_of_sell_request = len(sell_request_flats)
        # sell_request_flats_new_indices = sorted(range(len(sell_request_flats)),
        # key=sell_request_flats.__getitem__)
        sell_request_flats_new_indices = np.argsort(np.array(sell_request_flats))
        sell_request_flats = [sell_request_flats[i] for i in sell_request_flats_new_indices]
        every_flat_pool_max_energy_sell = (-1) * pool_buy_request / number_of_sell_request
        every_flat_pool_max_money_sell = every_flat_pool_max_energy_sell * p_pool_sell
        for flat_id in buy_request_flats_ids:
            flat_money_change[flat_id] = flats_requests[flat_id] * p_pool_buy
            pool_buy[flat_id] = -flats_requests[flat_id]
        for i in range(len(sell_request_flats)):
            if sell_request_flats[i] <= every_flat_pool_max_energy_sell:
                flat_money_change[
                    sell_request_flats_ids[sell_request_flats_new_indices[i]]] = sell_request_flats[i] * p_pool_sell
                pool_sell[sell_request_flats_ids[sell_request_flats_new_indices[i]]] = sell_request_flats[i]
                pool_buy_request += sell_request_flats[i]
                number_of_sell_request -= 1
            else:
                flat_energy_sell_on_grid = sell_request_flats[i] - every_flat_pool_max_energy_sell
                flat_money_sell_on_grid = flat_energy_sell_on_grid * p_grid_sell
                flat_money_change[
                    sell_request_flats_ids[sell_request_flats_new_indices[i]]] = \
                    flat_money_sell_on_grid + every_flat_pool_max_money_sell
                pool_sell[sell_request_flats_ids[sell_request_flats_new_indices[i]]] = every_flat_pool_max_energy_sell
                grid_sell[sell_request_flats_ids[sell_request_flats_new_indices[i]]] = flat_energy_sell_on_grid
                pool_buy_request += every_flat_pool_max_energy_sell
                number_of_sell_request -= 1
            if number_of_sell_request != 0:
                every_flat_pool_max_energy_sell = (-1) * pool_buy_request / number_of_sell_request
                every_flat_pool_max_money_sell = every_flat_pool_max_energy_sell * p_pool_sell

    return flat_money_change, pool_sell, pool_buy, grid_sell, pv_consume, grid_buy


class SimulationService:
    """This class handle simulation process."""

    def __init__(self, init_data_filepath: Path, results_directory: Path):
        self.init_data_filepath = init_data_filepath
        self.results_directory = results_directory
        self.results_directory.mkdir(parents=True, exist_ok=True)
        if self.init_data_filepath is None:
            # It is just example of init values. This values must be initilized in InitialDataService.
            self.num_monte_carlo_runs: int = 10
            self.max_flats_number: int = 10
            self.num_flats_step: int = 5
            self.consumption_data_directory: Path = Path(r"data\one_week_test")
            self.generation_data_filepath: Path = Path(r"data\pv_output_2020_37kWp.csv")
            self.start_stat_period: datetime = datetime(year=2012, month=12, day=24, hour=0, minute=0)
            self.end_stat_period: datetime = datetime(year=2012, month=12, day=31, hour=23, minute=30)
            self.pool_sell_price: float = 0.2
            self.pool_buy_price: float = 0.2
            self.grid_sell_price: float = 0.04
            self.grid_buy_price: float = 0.3404
            self.flat_sets: list[list[str]] = list()
            self.random_seed: int = 123123
        else:
            load_data_from_dict_to_objects(obj=self, load_data_dict=self._get_load_from_init_data_file())

    def _get_load_from_init_data_file(self) -> dict:
        with open(self.init_data_filepath, "r") as file:
            reader = DictReader(file)
            return reader.__next__()

    @staticmethod
    def read_generation_data_from_file(filepath: Path) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def run(self):
        """Function runs monte carlo simulation."""

        # define command line progress output
        print(f"\nRun simulation process:")
        start_time = time.time()
        # total_progress_number = len(self.flat_sets) * len(self.flat_sets[0])
        total_progress_number = len(self.flat_sets) * self.max_flats_number // self.num_flats_step
        total_progress_number_counter = 0

        df_generation = self.read_generation_data_from_file(self.generation_data_filepath)
        df_generation["DateTime"] = pd.to_datetime(df_generation["DateTime"])
        df_generation['DateTime'] = df_generation['DateTime'].astype('datetime64[s]')
        df_generation = df_generation.loc[(df_generation['DateTime'] >= self.start_stat_period) &
                                          (df_generation['DateTime'] <= self.end_stat_period)]#####################################################<=
        df_generation = df_generation.set_index("DateTime")
        date_range = pd.date_range(start=self.start_stat_period, end=self.end_stat_period, freq="30min")



        # iteration_raw([1.0, -1.0], 0.0, 0.0, 0.0, 0.0, 0.0)

        # cycle for every monte carlo simulation
        for iteration_counter, simulation_flats_sets in enumerate(self.flat_sets):

            list_of_uniq_flat_ids_in_current_simulation = list(set(chain(*simulation_flats_sets)))
            flat_consumption = {
                flat_id: pd.read_csv(self.consumption_data_directory.joinpath(flat_id + ".csv"), na_values=["Null"])
                for flat_id in list_of_uniq_flat_ids_in_current_simulation
            }
            for flat_id in flat_consumption.keys():
                flat_consumption[flat_id]["DateTime"] = pd.to_datetime(flat_consumption[flat_id]['DateTime'])
                flat_consumption[flat_id]['DateTime'] = flat_consumption[flat_id]['DateTime'].astype('datetime64[s]')
                flat_consumption[flat_id] = flat_consumption[flat_id].set_index("DateTime")

            simulation_filename = f"simulation_{iteration_counter + 1}.csv"
            if self.results_directory.joinpath(simulation_filename).is_file():
                self.results_directory.joinpath(simulation_filename).unlink()
            with open(self.results_directory.joinpath(simulation_filename), "a+") as file:
                csv_writer = writer(file, lineterminator='\r')
                headers = ["flat_money_change", "pool_sell", "pool_buy", "grid_sell", "pv_consume"]
                csv_writer.writerow(headers)
                # cycle for every flat set in simulation
                for flat_set in simulation_flats_sets:
                    res = [[0.0 for _ in range(len(flat_set))] for _ in range(5)]
                    for date_period in date_range:
                        flats_consumption_list = list()
                        for flat in flat_set:
                            # convert to raw float instead numpy.float64?
                            try:
                                consumption_value = flat_consumption[flat].loc[date_period]["consumption"]
                                if hasattr(consumption_value, '__iter__'):
                                    flats_consumption_list.append(float(consumption_value[0]))
                                else:
                                    flats_consumption_list.append(float(consumption_value))
                            except TypeError as e:
                                print(f"Handled Error: {e}, {flat_consumption[flat].loc[date_period]['consumption'][0]}")
                                pass
                            except KeyError as e:
                                # fill flats_consumption_list with zero if consumption for current date period not found
                                flats_consumption_list.append(0.0)
                                print(f"Handled KeyError: {e}, flat_id={flat}")


                        print(flats_consumption_list)

                        current_res = iteration_raw(
                            flats_consumption=typed.List(flats_consumption_list),
                            p_pool_buy=self.pool_buy_price,
                            p_pool_sell=self.pool_sell_price,
                            p_grid_buy=self.grid_buy_price,
                            p_grid_sell=self.grid_sell_price,
                            generation=float(df_generation.loc[date_period]["kWh/hh"])
                        )
                        # print(f"{date_period}: {current_res}")
                        for char_i, res_list in enumerate(current_res):
                            for i, flat_char in enumerate(res_list):
                                res[char_i][i] += flat_char

                    csv_writer.writerow(res)

                    # Update progress in the command line
                    total_progress_number_counter += 1
                    update_progress_in_command_line(
                        percent_progess=100.0 * total_progress_number_counter / total_progress_number,
                        time_worked_in_seconds=time.time() - start_time)

        # final progress in the command line update
        update_progress_in_command_line_with_success(time.time() - start_time)

    def run_speed_test(self):
        """Function runs monte carlo simulation."""

        # define command line progress output
        print(f"\nRun simulation process:")
        start_time = time.time()
        # total_progress_number = len(self.flat_sets) * len(self.flat_sets[0])
        total_progress_number = len(self.flat_sets) * self.max_flats_number // self.num_flats_step
        total_progress_number_counter = 0

        df_generation = self.read_generation_data_from_file(self.generation_data_filepath)
        df_generation["DateTime"] = pd.to_datetime(df_generation["DateTime"])
        df_generation['DateTime'] = df_generation['DateTime'].astype('datetime64[s]')
        df_generation = df_generation.loc[(df_generation['DateTime'] >= self.start_stat_period) &
                                          (df_generation['DateTime'] <= self.end_stat_period)]#####################################################<=
        df_generation = df_generation.set_index("DateTime")
        date_range = pd.date_range(start=self.start_stat_period, end=self.end_stat_period, freq="30min")

        df_peak = pd.DataFrame(index=range(self.num_monte_carlo_runs), columns=range(len(self.flats_num_list)))
        df_week_peak_demand = pd.DataFrame(index=range(335), columns=range(len(self.flats_num_list)))
        df_week_peak_generation = pd.DataFrame(index=range(335), columns=range(len(self.flats_num_list)))



        # iteration_raw([1.0, -1.0], 0.0, 0.0, 0.0, 0.0, 0.0)

        # cycle for every monte carlo simulation
        for iteration_counter, simulation_flats_sets in enumerate(self.flat_sets):

            list_of_uniq_flat_ids_in_current_simulation = list(set(chain(*simulation_flats_sets)))
            # flat_consumption = {
            #     flat_id: pd.read_csv(self.consumption_data_directory.joinpath(flat_id + ".csv"), na_values=["Null"])
            #     for flat_id in list_of_uniq_flat_ids_in_current_simulation
            # }
            # flat_consumption_generator = {
            #     flat_id: None
            #     for flat_id in list_of_uniq_flat_ids_in_current_simulation
            # }
            # for flat_id in flat_consumption.keys():
            #     flat_consumption[flat_id]["DateTime"] = pd.to_datetime(flat_consumption[flat_id]['DateTime'])
            #     flat_consumption[flat_id]['DateTime'] = flat_consumption[flat_id]['DateTime'].astype('datetime64[s]')
            #     # flat_consumption[flat_id] = flat_consumption[flat_id].set_index("DateTime")
            #     flat_consumption[flat_id] = flat_consumption[flat_id].values
            #     flat_consumption_generator[flat_id] = (val for val in flat_consumption[flat_id])
            # flat_consumption_curr_value = {
            #     flat_id: next(flat_consumption_generator[flat_id])
            #     for flat_id in list_of_uniq_flat_ids_in_current_simulation
            # }

            simulation_filename = f"simulation_{iteration_counter + 1}.csv"
            if self.results_directory.joinpath(simulation_filename).is_file():
                self.results_directory.joinpath(simulation_filename).unlink()
            with open(self.results_directory.joinpath(simulation_filename), "a+") as file:
                csv_writer = writer(file, lineterminator='\r')
                headers = ["flat_money_change", "pool_sell", "pool_buy", "grid_sell", "pv_consume", "grid_buy"]
                csv_writer.writerow(headers)
                # cycle for every flat set in simulation #########################################################################
                for j, flat_set in enumerate(simulation_flats_sets):
                    #####################################################################################################

                    flat_consumption = {
                        flat_id: pd.read_csv(self.consumption_data_directory.joinpath(flat_id + ".csv"),
                                             na_values=["Null"])
                        for flat_id in list_of_uniq_flat_ids_in_current_simulation
                    }
                    print(flat_consumption)



                    flat_consumption_generator = {
                        flat_id: None
                        for flat_id in list_of_uniq_flat_ids_in_current_simulation
                    }
                    print('this is flat consumption generator')
                    print(flat_consumption_generator)

                    for flat_id in flat_consumption.keys():
                        #print(flat_id)
                        flat_consumption[flat_id]['DateTime'] = pd.to_datetime(flat_consumption[flat_id]['DateTime'])
                        flat_consumption[flat_id]['DateTime'] = flat_consumption[flat_id]['DateTime'].astype('datetime64[s]')
                        # flat_consumption[flat_id] = flat_consumption[flat_id].set_index("DateTime")
                        flat_consumption[flat_id] = flat_consumption[flat_id].values
                        flat_consumption_generator[flat_id] = (val for val in flat_consumption[flat_id])
                    print('this is flat consumption id')
                    print(flat_consumption_generator)
                    try:
                        flat_consumption_curr_value = {
                            flat_id: next(flat_consumption_generator[flat_id])
                            for flat_id in list_of_uniq_flat_ids_in_current_simulation
                        }
                    except StopIteration:
                        print('Stop iteration error')

                    print('this is flat consumption curr value ')

                    print(flat_consumption_curr_value)

                    #####################################################################################################
                    res = [[0.0 for _ in range(len(flat_set))] for _ in range(6)]
                    for flat in flat_set:
                        flat_consumption[flat] = (val for val in flat_consumption[flat])

                    sum_peak_demand = 0
                    sum_peak_generation = 0
                    peak_to_peak = 0

                    curr_date = self.start_stat_period.date()
                    curr_week = self.start_stat_period.isocalendar().week
                    generation = list()
                    consumption = list()

                    for i, date_period in enumerate(date_range):

                        flats_consumption_list = list()
                        for flat in flat_set:
                            try:
                                while True:
                                    if flat_consumption_curr_value[flat][0] == date_period:
                                        # this line does the same as the if not ... else statement below
                                        #flats_consumption_list.append(flat_consumption_curr_value[flat][1] if not math.isnan(flat_consumption_curr_value[flat][1]) else 0)
                                        if not math.isnan(flat_consumption_curr_value[flat][1]):
                                            flats_consumption_list.append(flat_consumption_curr_value[flat][1])
                                        else:
                                            flats_consumption_list.append(0.0)
                                            #print('0 added ##################################################################')
                                        break
                                    elif flat_consumption_curr_value[flat][0] < date_period:
                                        flat_consumption_curr_value[flat] = next(flat_consumption_generator[flat])
                                    elif flat_consumption_curr_value[flat][0] > date_period:
                                        flats_consumption_list.append(0.0)
                                        break
                            except StopIteration:
                                flats_consumption_list.append(0.0)

                            except KeyError:
                                flats_consumption_list.append(0.0)


                        #### Code to print out image
                        if curr_date == date_period.date():

                            load = abs(sum(flats_consumption_list)) - abs(df_generation.loc[date_period]["kWh/hh"])
                            if load > 0:
                                #print(flats_consumption_list)
                                #print(sum(flats_consumption_list))
                                if abs(load) > sum_peak_demand:
                                    sum_peak_demand = abs(load)
                                    peak_day = curr_date
                                #print(f'this is sum_peak_demand {sum_peak_demand}')

                            #print(df_generation.loc[date_period]["kWh/hh"])

                            else:
                                if abs(load) > sum_peak_generation:
                                    sum_peak_generation = abs(load)
                                    peak_day = curr_date
                            #print(f'This is sum_peak_generation {sum_peak_generation}')

                        else:
                            #if day_peak_demand == day_peak_generation:
                            if abs(sum_peak_demand) + abs(sum_peak_generation) > peak_to_peak:
                                peak_to_peak = abs(sum_peak_demand) + abs(sum_peak_generation)
                                day_of_peak = peak_day
                            #print(f'This is peak_to_peak {peak_to_peak}')



                        if curr_week == curr_date.isocalendar().week:

                        #if curr_week == 18:

                            consumption.append(abs(sum(flats_consumption_list)))
                            generation.append(abs(df_generation.loc[date_period]["kWh/hh"]))
                        else:
                            #print(f'this is the week of the last peak {day_of_peak.isocalendar().week}')
                           # if day_of_peak.isocalendar().week == curr_week:
                            if curr_week == 26:
                                #print(f'this is last peak day{day_of_peak}')
                                #print(consumption)
                                #print(len(consumption))
                                #print(len(generation))

                                df_week_peak_demand.iloc[0:len(consumption), j] = consumption
                                df_week_peak_generation.iloc[0:len(consumption), j] = generation
                                if j == 0:
                                    j_0 = curr_week
                                if j == 1:
                                    j_1 = curr_week
                                if j == 2:
                                    j_2 = curr_week
                                if j == 3:
                                    j_3 = curr_week
                                if j == 4:
                                    j_4 = curr_week
                                if j == 5:
                                    j_5 = curr_week

                            generation = list()
                            consumption = list()
                            curr_week = date_period.isocalendar().week

                        curr_date = date_period.date()

                        #### Code to print out image

                        try:
                            current_res = iteration_raw(
                                flats_consumption=typed.List(flats_consumption_list),
                                p_pool_buy=self.pool_buy_price,
                                p_pool_sell=self.pool_sell_price,
                                p_grid_buy=self.grid_buy_price,
                                p_grid_sell=self.grid_sell_price,
                                generation=float(df_generation.loc[date_period]["kWh/hh"])
                            )
                        except KeyboardInterrupt:
                            print(flats_consumption_list)


                        # print(f"{date_period}: {current_res}")
                        for char_i, res_list in enumerate(current_res):
                            for i, flat_char in enumerate(res_list):
                                res[char_i][i] += flat_char

                        #print(f'{flat_set}###{flats_consumption_list} ##### {float(df_generation.loc[date_period]["kWh/hh"])} ##### {res}')

                    csv_writer.writerow(res)

                    # Update progress in the command line
                    total_progress_number_counter += 1
                    update_progress_in_command_line(
                        percent_progess=100.0 * total_progress_number_counter / total_progress_number,
                        time_worked_in_seconds=time.time() - start_time)

                    df_peak.iloc[iteration_counter, j] = peak_to_peak
                    #print(df_peak)
            df_peak.to_csv(self.results_directory.joinpath("peak_to_peak.csv"))

        df_week_peak_demand.to_csv(self.results_directory.joinpath("week_peak_demand.csv"))
        df_week_peak_generation.to_csv(self.results_directory.joinpath("week_peak_generation.csv"))

        print(f' j = 0: {j_0}')
        print(f' j = 1: {j_1}')
        print(f' j = 2: {j_2}')
        print(f' j = 3: {j_3}')
        print(f' j = 4: {j_4}')
        print(f' j = 5: {j_5}')


        # final progress in the command line update
        update_progress_in_command_line_with_success(time.time() - start_time)
