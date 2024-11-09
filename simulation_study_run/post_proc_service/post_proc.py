from pathlib import Path
from datetime import datetime
from csv import DictReader, reader, writer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from common_functions.objects_proc import load_data_from_dict_to_objects
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import ast
import csv
import numpy as np
import numpy_financial as npf
import sys
import scipy.stats as stats
import pandas as pd


def equality_index(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return 1 - ((fair_area - area) / fair_area)


def calc_load(char_list: list[list[float]], source) -> list[float]:
    source_index = {'pool_sell': 1, 'pool_buy': 2, 'grid_sell': 3, 'pv_consume': 4, 'grid_buy': 5}

    load_list = [0.0]

    for i in range(len(char_list[0])):
        load_list[i] = sum(char_list[source_index[source]])

        return load_list


def calc_benefit_index(lst):
    """Returns the share of positive values in a list."""
    negative_count = len([num for num in lst if num > 0])
    total_count = len(lst)
    if total_count == 0:
        return 0
    else:
        return negative_count / total_count


#######################################################################################################################

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


#########################################################################################################################


class PostProcService:
    """This class calculates characteristics of results..."""

    def __init__(self, init_data_filepath: Path, results_directory: Path, post_proc_directory: Path):
        self.init_data_filepath = init_data_filepath
        self.results_directory = results_directory
        self.post_proc_directory = post_proc_directory
        self.post_proc_directory.mkdir(parents=True, exist_ok=True)
        self.discount_rate = 0.03  # local value
        self.operation_maintenance_cost_per_kW = 30
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
            self.grid_buy_price: float = 0.34
            self.coop_sell_price: float = 0.0
            self.price_per_kwp: int = 1018
            self.kwp_installed: int = 37
            self.flat_sets: list[list[str]] = list()
            self.random_seed: int = 123123
        else:
            load_data_from_dict_to_objects(obj=self, load_data_dict=self._get_load_from_init_data_file())

        # this is the same as self.simulation_results_pool_buy, self.simulation_results_pool_sell, etc.
        # but just already summed up. Not sure why I did this. I assume I did this first and then realised I need them
        # individualy and did ut again
        self.simulation_pv_consume, self.simulation_grid_sell, self.simulation_pool_sell, self.simulation_grid_buy, \
        self.simulation_pool_buy = self._proc_simulation_results()

        self.simulation_results = [
            self._read_post_proc_results(simulation_filename=f"simulation_{i + 1}.csv")
            for i in range(self.num_monte_carlo_runs)
        ]

        self.simulation_results_pv_consume = [
            self._read_results(simulation_filename=f"simulation_{i + 1}.csv", source="pv_consume")
            for i in range(self.num_monte_carlo_runs)
        ]

        self.simulation_results_pool_buy = [
            self._read_results(simulation_filename=f"simulation_{i + 1}.csv", source="pool_buy")
            for i in range(self.num_monte_carlo_runs)
        ]

        self.simulation_results_pool_sell = [
            self._read_results(simulation_filename=f"simulation_{i + 1}.csv", source="pool_sell")
            for i in range(self.num_monte_carlo_runs)
        ]

        self.simulation_results_grid_buy = [
            self._read_results(simulation_filename=f"simulation_{i + 1}.csv", source="grid_buy")
            for i in range(self.num_monte_carlo_runs)
        ]

        self.simulation_results_grid_sell = [
            self._read_results(simulation_filename=f"simulation_{i + 1}.csv", source="grid_sell")
            for i in range(self.num_monte_carlo_runs)
        ]

    def _get_load_from_init_data_file(self) -> dict:
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
        with open(self.init_data_filepath, "r") as file:
            reader = DictReader(file)
            return reader.__next__()

    def calc_benefit_p(self, char_list: list[list[float]]) -> list[float]:
        benefit_list = [0.0 for _ in range(len(char_list[0]))]

        # char_list[0] - flat_money_change
        # char_list[1] - pool_sell
        # char_list[2] - pool_buy
        # char_list[3] - grid_sell
        # char_list[4] - pv_consume
        # char_list[5] - grid_buy

        """For collective self-consumption"""
        for i in range(len(char_list[0])):
            benefit_list[i] = char_list[4][i] * (self.grid_buy_price - self.coop_sell_price) + \
                              char_list[2][i] * (self.grid_buy_price - self.coop_sell_price) + \
                              char_list[5][i] * (self.grid_buy_price - self.coop_sell_price)
        return benefit_list

        """For normal P2P trading market"""
        # for i in range(len(char_list[0])):
        #    benefit_list[i] = char_list[4][i] * (self.grid_buy_price - self.coop_sell_price) + \
        #                      char_list[2][i] * (self.grid_buy_price - self.pool_buy_price) + \
        #                      char_list[1][i] * (self.pool_sell_price - self.coop_sell_price)
        # return benefit_list

    def calc_benefit_coop(self, char_list: list[list[float]]) -> list[float]:
        benefit_list = [0.0 for _ in range(len(char_list[0]))]

        # char_list[0] - flat_money_change
        # char_list[1] - pool_sell
        # char_list[2] - pool_buy
        # char_list[3] - grid_sell
        # char_list[4] - pv_consume
        # char_list[5] - grid_buy

        for i in range(len(char_list[0])):
            benefit_list[i] = char_list[4][i] * self.coop_sell_price + \
                              char_list[5][i] * self.grid_sell_price
        return benefit_list

    def _write_post_proc_results(self, data: list[list[float]], simulation_filename: str):
        with open(self.post_proc_directory.joinpath(simulation_filename), "w+") as file:
            csv_writer = writer(file, lineterminator='\r')
            csv_writer.writerow(data)

    def _read_post_proc_results(self, simulation_filename: str) -> list[list[float]]:
        with open(self.post_proc_directory.joinpath(simulation_filename), "r+") as file:
            csv_reader = reader(file)
            simulation_data = list()
            for row in csv_reader:
                for flats_per_run in row:
                    simulation_data.append(ast.literal_eval(flats_per_run))
            return simulation_data

    def _read_results(self, simulation_filename: str, source) -> list[list[float]]:
        source_index = {'pool_sell': 1, 'pool_buy': 2, 'grid_sell': 3, 'pv_consume': 4, 'grid_buy': 5}

        with open(self.results_directory.joinpath(simulation_filename), "r+") as file:
            simulation_data = list()
            csv_reader = reader(file)
            next(csv_reader)
            for row in csv_reader:
                simulation_data.append(ast.literal_eval(row[source_index[source]]))
            return simulation_data

    def _proc_simulation_results(self):
        simulation_pv_consume_sum = list()
        simulation_grid_sell_sum = list()
        simulation_pool_sell_sum = list()
        simulation_grid_buy_sum = list()
        simulation_pool_buy_sum = list()

        for iteration_counter, simulation_flats_sets in enumerate(self.flat_sets):
            simulation_filename = f"simulation_{iteration_counter + 1}.csv"
            simulation_filename_coop = f"simulation_coop{iteration_counter + 1}.csv"
            simulation_benefit_list = list()
            simulation_pv_consume = list()
            simulation_grid_sell = list()
            simulation_pool_sell = list()
            simulation_grid_buy = list()
            simulation_pool_buy = list()

            with open(self.results_directory.joinpath(simulation_filename), "r+") as file:
                csv_reader = reader(file)
                csv_reader.__next__()
                for row, flats_set in zip(csv_reader, simulation_flats_sets):
                    char_list = []
                    for row_char in row:
                        # print(row)
                        char_list.append(ast.literal_eval(row_char))

                    # calculates financial benefit as per definition for participant
                    simulation_benefit_list.append(self.calc_benefit_p(char_list=char_list))

                    simulation_pv_consume.append(calc_load(char_list, 'pv_consume'))
                    simulation_grid_sell.append(calc_load(char_list, 'grid_sell'))
                    simulation_pool_sell.append(calc_load(char_list, 'pool_sell'))
                    simulation_grid_buy.append(calc_load(char_list, 'grid_buy'))
                    simulation_pool_buy.append(calc_load(char_list, 'pool_buy'))

            self._write_post_proc_results(data=simulation_benefit_list, simulation_filename=simulation_filename)

            # This is the list in a list with the sum of "load type" under each simulation step with a new list added
            # for each monte carlo run
            simulation_pv_consume_sum.append(simulation_pv_consume)
            simulation_grid_sell_sum.append(simulation_grid_sell)
            simulation_pool_sell_sum.append(simulation_pool_sell)
            simulation_grid_buy_sum.append(simulation_grid_buy)
            simulation_pool_buy_sum.append(simulation_pool_buy)

        return simulation_pv_consume_sum, simulation_grid_sell_sum, simulation_pool_sell_sum, simulation_grid_buy_sum, \
               simulation_pool_buy_sum

    def calc_payback_period(self, benefits: float, pv_share=1):

        installation_costs = self.price_per_kwp * self.kwp_installed
        payback_period = installation_costs / pv_share / benefits
        return payback_period

    def calc_marginal_benefit_kwh(self, payback_period, benefits, consumption):
        # benefits = annual savings calculated in post proc files
        profitable_period = self.pv_lifetime - payback_period
        annual_savings_over_lifetime = (benefits * profitable_period) / self.pv_lifetime
        marginal_benefit = annual_savings_over_lifetime / consumption
        return marginal_benefit

    def distribution_plot(self, x_values, data_set, plot_color, labels, calc_task):
        self.values = ResultGraph()
        self.values.curves = [Curve() for _ in range(11)]
        percentil_values = [100.0 * (0.0 + i * 0.1) for i in range(11)]

        for i, n_flat in enumerate(x_values):
            values_list = list()

            if calc_task == 'community':
                for simulation in data_set:
                    values_list.append(sum(simulation[i]))
            if calc_task == 'individual':
                for simulation in data_set:
                    values_list.append(sum(simulation[i]) / len(simulation[i]))
            if calc_task == 'equality':
                for simulation in data_set:
                    values_list.append(equality_index(simulation[i]))

            for j, percentile_value in enumerate(percentil_values):
                self.values.curves[j].append_point(
                    [n_flat, np.percentile(values_list, percentile_value)])

        center_index = len(percentil_values) // 2
        i = 0
        for load_curve in self.values.curves:
            if i == center_index:
                i += 1
                continue
            load_curve.color = plot_color
            load_curve.alpha = (0.5 - abs(len(percentil_values) // 2 - i) / 10.0) * 2.0
            if load_curve.alpha == 0:
                load_curve.alpha = 0.1
            i += 1
        self.values.curves[center_index].color = "black"
        self.values.curves[center_index].alpha = 1.0

        center_index = len(self.values.curves) // 2
        for i in range(len(self.values.curves)):
            if i == center_index:
                continue
            if i < center_index:
                plt.fill_between(self.values.curves[i].get_axis_values(0),
                                 self.values.curves[i].get_axis_values(1),
                                 self.values.curves[i + 1].get_axis_values(1),
                                 color=self.values.curves[i].color,
                                 alpha=self.values.curves[i].alpha)
            elif i > center_index:
                plt.fill_between(self.values.curves[i].get_axis_values(0),
                                 self.values.curves[i].get_axis_values(1),
                                 self.values.curves[i - 1].get_axis_values(1),
                                 color=self.values.curves[i].color,
                                 alpha=self.values.curves[i].alpha)
        plt.plot(self.values.curves[center_index].get_axis_values(0),
                 self.values.curves[center_index].get_axis_values(1),
                 color=self.values.curves[center_index].color,
                 alpha=self.values.curves[center_index].alpha)
        return

        ########################################################################################################################

    def plot_graph(self, aver_line_y_values=None):

        green = '#28A197'  # self-consumption
        blue = '#12436D'  # pool energy
        red = '#801650'  # grid energy
        orange = '#F46A25'
        grey = '#3D3D3D'
        purple = '#A285D1'

        p_pool = str(self.pool_buy_price).replace('.', '')
        p_coop = str(self.coop_sell_price).replace('.', '')
        p_grid_sell = str(self.grid_sell_price).replace('.', '')

        ########################################################################################################################
        # PV energy consumption by source -- COMMUNITY

        title = 'PV energy consumption by source (community) '
        x_label = 'Market size (No. of participants involved) '
        y_label = 'Electricity in kWh '

        # -------------------------------------------------------------------------------------------------------------
        # line plot
        plt.grid()

        # plt.title(title + '(line)')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        aver_line_y_values = [0.0 for _ in self.simulation_results_pv_consume[0]]

        for simulation in self.simulation_results_pv_consume:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run)
            plt.plot(x_values, y_values, color=green, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pv_consume_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=green, linewidth=3, label='Self-consumed')
        plt.legend()
        # plt.scatter(x_values, aver_line_y_values, color="r")
        aver_line_y_values = [0.0 for _ in self.simulation_results_grid_sell[0]]

        for simulation in self.simulation_results_grid_sell:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run)
            plt.plot(x_values, y_values, color=red, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        grid_sell_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=red, linewidth=3, label='Sold to grid')
        plt.legend()

        aver_line_y_values = [0.0 for _ in self.simulation_results_pool_sell[0]]

        for simulation in self.simulation_results_pool_sell:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run)
            plt.plot(x_values, y_values, color=blue, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pool_sell_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=blue, linewidth=3, label='Sold to pool')
        plt.legend(fontsize=13)
        plt.xticks(x_values, fontsize=15)
        plt.savefig(self.results_directory.joinpath(str(title) + "line.png"))
        # lt.show()
        plt.close()



        # -------------------------------------------------------------------------------------------------------------
        # stacked plot
        plt.grid()

        # plt.title(title + '(stacked)')
        plt.xlabel(x_label)
        plt.ylabel('Share in %')

        y = np.row_stack((pv_consume_aver_line, grid_sell_aver_line, pool_sell_aver_line))
        percent = y / y.sum(axis=0).astype(float) * 100

        plt.stackplot(x_values, percent, labels=('Self-consumed', 'Sold to grid', 'Sold to pool'),
                      colors=[green, red, blue])
        plt.legend(fontsize=13)
        plt.xticks(x_values, fontsize=15)

        plt.savefig(self.results_directory.joinpath(str(title) + "stacked.png"))
        plt.close()

        # -------------------------------------------------------------------------------------------------------------
        # distribution plot

        fig, ax = plt.subplots(1, figsize=(6,5))
        ax.grid()

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        self.distribution_plot(x_values, self.simulation_results_pv_consume, green, 'Self-consumed', 'community')
        self.distribution_plot(x_values, self.simulation_results_grid_sell, red, 'Sold to grid', 'community')
        self.distribution_plot(x_values, self.simulation_results_pool_sell, blue, 'Sold to pool', 'community')

        lbl_selfconsume = mpatches.Patch(color=green, label='Self-consumed')
        lbl_sold_grid = mpatches.Patch(color=red, label='Sold to grid')
        lbl_sold_pool = mpatches.Patch(color=blue, label='Sold to pool')


        plt.legend(handles=[lbl_selfconsume, lbl_sold_grid, lbl_sold_pool], fontsize=13)
        plt.xticks(x_values, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(left=0)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.tight_layout()
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        plt.savefig(self.results_directory.joinpath(str(title) + "_" + ".png"))
        # plt.show()
        plt.close()

        # ---------------------------------------------------
        # Calc-self consumption
        selfconsumption = list()
        print(pv_consume_aver_line[1])

        for pv_consume,  grid_sell, pool_sell in zip(pv_consume_aver_line, grid_sell_aver_line, pool_sell_aver_line):
            selfconsumption.append((pv_consume + pool_sell) / (pv_consume + pool_sell + grid_sell))

        print('This is the community self-consumption:')
        print(selfconsumption)

        print(f'This is pool sell {(pool_sell_aver_line[0])}')
        print(f'This is pv consume {(pv_consume_aver_line[0])}')
        print(f'This is grid sell {(grid_sell_aver_line[0])}')


        ########################################################################################################################
        # PV energy consumption by source -- INDIVIDUAL

        title = 'PV energy consumption by source (individual) '
        x_label = 'Market size (No. of participants) '
        y_label = 'Electricity in kWh '

        # -------------------------------------------------------------------------------------------------------------

        plt.grid()

        # plt.title(title + '(line)')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        aver_line_y_values = [0.0 for _ in self.simulation_results_pv_consume[0]]

        for simulation in self.simulation_results_pv_consume:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run) / len(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run) / len(flats_per_run)
            plt.plot(x_values, y_values, color=green, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pv_consume_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=green, linewidth=3, label='Self-consumed')
        plt.legend()
        # plt.scatter(x_values, aver_line_y_values, color="r")
        aver_line_y_values = [0.0 for _ in self.simulation_results_grid_sell[0]]

        for simulation in self.simulation_results_grid_sell:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run) / len(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run) / len(flats_per_run)
            plt.plot(x_values, y_values, color=red, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        grid_sell_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=red, linewidth=3, label='Sold to grid')
        plt.legend()

        aver_line_y_values = [0.0 for _ in self.simulation_results_pool_sell[0]]

        for simulation in self.simulation_results_pool_sell:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run) / len(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run) / len(flats_per_run)
            plt.plot(x_values, y_values, color=blue, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pool_sell_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=blue, linewidth=3, label='Sold to pool')
        plt.legend()

        plt.savefig(self.results_directory.joinpath(str(title) + ".png"))
        # plt.show()
        plt.close()

        # -------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots(1, figsize=(6,5))
        ax.grid()

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        self.distribution_plot(x_values, self.simulation_results_pv_consume, green, 'Self-consumed', 'individual')
        self.distribution_plot(x_values, self.simulation_results_grid_sell, red, 'Sold to grid', 'individual')
        self.distribution_plot(x_values, self.simulation_results_pool_sell, blue, 'Sold to pool', 'individual')

        lbl_selfconsume = mpatches.Patch(color=green, label='Self-consumed')
        lbl_sold_grid = mpatches.Patch(color=red, label='Sold to grid')
        lbl_sold_pool = mpatches.Patch(color=blue, label='Sold to pool')


        plt.legend(handles=[lbl_selfconsume, lbl_sold_grid, lbl_sold_pool], fontsize=13)
        plt.xticks(x_values, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(left=0)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.tight_layout()
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        plt.savefig(self.results_directory.joinpath(str(title) + "_" + ".png"))
        # plt.show()
        plt.close()

        ########################################################################################################################
        # Household energy consumption by source -- COMMUNITY

        title = 'Household_energy_consumption_by_source (community) '
        x_label = 'Market size (No. of participants)'
        y_label = 'Electricity in kWh '

        # -------------------------------------------------------------------------------------------------------------
        plt.grid()

        # plt.title(title + '(line)')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        aver_line_y_values = [0.0 for _ in self.simulation_results_pv_consume[0]]

        for simulation in self.simulation_results_pv_consume:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run)
            plt.plot(x_values, y_values, color=green, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pv_consume_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=green, linewidth=3, label='Self-consumed')
        plt.legend()
        # plt.scatter(x_values, aver_line_y_values, color="r")
        aver_line_y_values = [0.0 for _ in self.simulation_results_grid_buy[0]]

        for simulation in self.simulation_results_grid_buy:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run)
            plt.plot(x_values, y_values, color=red, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        grid_buy_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=red, linewidth=3, label='Bought from grid')
        plt.legend()

        aver_line_y_values = [0.0 for _ in self.simulation_results_pool_buy[0]]

        for simulation in self.simulation_results_pool_buy:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run)
            plt.plot(x_values, y_values, color=blue, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pool_buy_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=blue, linewidth=3, label='Bought from pool')
        plt.legend()

        plt.xticks(x_values)

        plt.savefig(self.results_directory.joinpath(str(title) + "line.png"))
        ##plt.show()
        plt.close()

        # ---------------------------------------------------
        # Calc self-sufficiency
        selfsufficiency = list()
        print(pv_consume_aver_line[1])

        for pv_consume, grid_buy, pool_buy in zip(pv_consume_aver_line, grid_buy_aver_line, pool_buy_aver_line):
            selfsufficiency.append((pv_consume + pool_buy)/(pv_consume + grid_buy + pool_buy))

        print('This is the community self-sufficiency:')
        print(selfsufficiency)

        # -------------------------------------------------------------------------------------------------------------

        plt.grid()

        # plt.title(title + '(stacked)')
        plt.xlabel(x_label)
        plt.ylabel('Share in %')

        y = np.row_stack((pv_consume_aver_line, pool_buy_aver_line, grid_buy_aver_line))
        percent = y / y.sum(axis=0).astype(float) * 100

        plt.stackplot(x_values, percent, labels=('Self-consumed', 'Bought from pool', 'Bought from grid'),
                      colors=[green, blue, red])
        plt.legend()
        plt.xticks(x_values)

        plt.savefig(self.results_directory.joinpath(str(title) + "stacked.png"))
        plt.close()

        # -------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots(1, figsize=(6, 5))
        ax.grid()

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        self.distribution_plot(x_values, self.simulation_results_pv_consume, green, 'Self-consumed', 'community')
        self.distribution_plot(x_values, self.simulation_results_grid_buy, red, 'Bought from grid', 'community')
        self.distribution_plot(x_values, self.simulation_results_pool_buy, blue, 'Bought from pool', 'community')
        plt.xticks(x_values)

        lbl_selfconsume = mpatches.Patch(color=green, label='Self-consumed')
        lbl_bought_grid = mpatches.Patch(color=red, label='Bought from grid')
        lbl_bought_pool = mpatches.Patch(color=blue, label='Bought from pool')



        plt.legend(handles=[lbl_selfconsume, lbl_bought_grid, lbl_bought_pool], fontsize=13)
        plt.xticks(x_values, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(left=0)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.tight_layout()
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        plt.savefig(self.results_directory.joinpath(str(title) + "_" + ".png"))
        # plt.show()
        plt.close()

        ########################################################################################################################
        # Household energy consumption by source -- INDIVIDUAL

        title = 'Household_energy_consumption_by_source (individual) '
        x_label = 'Market size (No. of participants)'
        y_label = 'Electricity in kWh '

        # -------------------------------------------------------------------------------------------------------------
        plt.grid()

        # plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        aver_line_y_values = [0.0 for _ in self.simulation_results_pv_consume[0]]

        for simulation in self.simulation_results_pv_consume:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run) / len(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run) / len(flats_per_run)
            plt.plot(x_values, y_values, color=green, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pv_consume_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=green, linewidth=3, label='Self-consumed')
        plt.legend()
        # plt.scatter(x_values, aver_line_y_values, color="r")
        aver_line_y_values = [0.0 for _ in self.simulation_results_grid_buy[0]]

        for simulation in self.simulation_results_grid_buy:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run) / len(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run) / len(flats_per_run)
            plt.plot(x_values, y_values, color=red, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        grid_buy_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=red, linewidth=3, label='Bought from grid')
        plt.legend()

        aver_line_y_values = [0.0 for _ in self.simulation_results_pool_buy[0]]

        for simulation in self.simulation_results_pool_buy:
            y_values = list()
            for i, flats_per_run in enumerate(simulation):
                y_values.append(sum(flats_per_run) / len(flats_per_run))
                aver_line_y_values[i] += sum(flats_per_run) / len(flats_per_run)
            plt.plot(x_values, y_values, color=blue, alpha=0.3)
        aver_line_y_values = [curr / self.num_monte_carlo_runs for curr in aver_line_y_values]
        pool_buy_aver_line = aver_line_y_values
        plt.plot(x_values, aver_line_y_values, color=blue, linewidth=3, label='Bought from pool')
        plt.legend()
        plt.xticks(x_values)

        plt.savefig(self.results_directory.joinpath(str(title) + ".png"))
        ##plt.show()
        plt.close()

        # -------------------------------------------------------------------------------------------------------------

        fig, ax = plt.subplots(1, figsize=(6, 5))
        ax.grid()

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        self.distribution_plot(x_values, self.simulation_results_pv_consume, green, 'Self-consumed', 'individual')
        self.distribution_plot(x_values, self.simulation_results_grid_buy, red, 'Bought from grid', 'individual')
        self.distribution_plot(x_values, self.simulation_results_pool_buy, blue, 'Bought from pool', 'individual')

        lbl_selfconsume = mpatches.Patch(color=green, label='Self-consumed')
        lbl_bought_grid = mpatches.Patch(color=red, label='Bought from grid')
        lbl_bought_pool = mpatches.Patch(color=blue, label='Bought from pool')

        plt.legend(handles=[lbl_selfconsume, lbl_bought_grid, lbl_bought_pool], fontsize=13)
        plt.xticks(x_values, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(left=0)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.tight_layout()
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        plt.savefig(self.results_directory.joinpath(str(title) + "_" + ".png"))
        # plt.show()
        plt.close()

        ########################################################################################################################

        title = 'Community benefits '
        x_label = 'Market size (No. of participants involved) '
        y_label = 'Annual benefits in £ '

        # -------------------------------------------------------------------------------------------------------------
        # Community benefit

        # plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        mean_value = list()
        ci_lower = list()
        ci_upper = list()
        for i, n_flat in enumerate(x_values):
            value_list = list()
            for simulation in self.simulation_results:
                # Calculate the sample mean and standard deviation
                value_list.append(sum(simulation[i]))
            mean = sum(value_list) / len(value_list)
            std_dev = stats.tstd(value_list)
            mean_value.append(mean)
            interval = stats.t.interval(0.95, len(value_list) - 1, loc=mean, scale=std_dev / len(value_list) ** 0.5)
            ci_lower.append(interval[0])
            ci_upper.append(interval[1])

        d = {'col1': mean_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
        community_benefit_pool_price_0 = pd.DataFrame(data=d)
        community_benefit_pool_price_0.to_csv(
            self.results_directory.joinpath("community_benefit_p_coop_" + p_coop + "_p_pool_" + p_pool + ".csv"),
            index=True)

        fig, ax = plt.subplots()
        x = x_values
        ax.plot(x, mean_value, color=orange, linestyle='dashed', marker='o')
        ax.fill_between(
            x, ci_lower, ci_upper, color=orange, alpha=.15)
        ax.set_ylim(ymin=0)
        ax.set_title(title)
        plt.ylabel(y_label)
        fig.legend()
        plt.grid()
        plt.xticks(x_values)
        plt.savefig(self.results_directory.joinpath(str(title) + ".png"))
        plt.close()

        ########################################################################################################################
        # Individual benefit

        title = 'Individual benefits (average)'
        plt.grid()
        # plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        mean_value = list()
        ci_lower = list()
        ci_upper = list()
        for i, n_flat in enumerate(x_values):
            value_list = list()
            for simulation in self.simulation_results:
                # Calculate the sample mean and standard deviation
                value_list.extend(simulation[i])
            mean = sum(value_list) / len(value_list)
            mean_value.append(mean)
            interval = stats.t.interval(alpha=0.95, df=len(value_list) - 1, loc=np.mean(value_list),
                                        scale=stats.sem(value_list))

            ci_lower.append(interval[0])
            ci_upper.append(interval[1])

        d = {'col1': mean_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
        individual_benefit_pool_price_0 = pd.DataFrame(data=d)
        individual_benefit_pool_price_0.to_csv(
            self.results_directory.joinpath("individual_benefit_p_coop_" + p_coop + "_p_pool_" + p_pool + ".csv"),
            index=True)

        fig, ax = plt.subplots()
        x = x_values
        ax.plot(x, mean_value, color=orange, linestyle='dashed', marker='o')
        ax.fill_between(
            x, ci_lower, ci_upper, color=orange, alpha=.15)
        ax.set_ylim(ymin=0)
        plt.xticks(x_values)
        plt.grid()
        plt.savefig(self.results_directory.joinpath(str(title) + ".png"))
        plt.close()
        ########################################################################################################################
        # Cost Calculation

        title = 'Annual cost of electricity per participant'
        x_label = 'Market size (No. of participants involved) '
        y_label = 'Annual electricity bill in £'

        # -------------------------------------------------------------------------------------------------------------
        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        bau_data = list()
        p2p_data = list()
        bill_reduction = list()

        for i, n_flat in enumerate(x_values):

            bau_bill = list()
            p2p_bill = list()
            benefit = list()
            total_load = list()
            for simulation, pv_consume, grid_buy, pool_buy in zip(self.simulation_results,
                                                                  self.simulation_results_pv_consume,
                                                                  self.simulation_results_grid_buy,
                                                                  self.simulation_results_pool_buy):
                total_load.extend([_solar + _grid + _pool for _solar, _grid, _pool in
                                   zip(pv_consume[i], grid_buy[i], pool_buy[i])])
                benefit.extend(simulation[i])

            bau_bill.extend([load * self.grid_buy_price for load in total_load])
            p2p_bill.extend([bill * self.grid_buy_price - benefits for bill, benefits in zip(total_load, benefit)])
            bill_reduction.append((1 - ((sum(p2p_bill) / len(p2p_bill)) / (sum(bau_bill) / len(bau_bill)))) * 100)

            bau_data.append(bau_bill)
            p2p_data.append(p2p_bill)

        def set_box_color(bp, color):
            plt.setp(bp['boxes'], color="0")
            plt.setp(bp['boxes'], facecolor=color)
            plt.setp(bp['whiskers'], color="0")
            plt.setp(bp['caps'], color="0")
            plt.setp(bp['medians'], color="0")

        print('This is the annual reduction in %')
        print(bill_reduction)
        plt.grid()
        plt.title(title)

        bpl = plt.boxplot(bau_data, patch_artist=True, positions=np.array(range(len(bau_data))) * 2.0 - 0.4,
                          widths=0.6, showfliers=False)
        bpr = plt.boxplot(p2p_data, patch_artist=True, positions=np.array(range(len(p2p_data))) * 2.0 + 0.4,
                          widths=0.6, showfliers=False)
        set_box_color(bpl, green)  # colors are from http://colorbrewer2.org/
        set_box_color(bpr, blue)

        # draw temporary red and blue lines and use them to create a legend
        plt.plot([], c=green, label='BAU')
        plt.plot([], c=blue, label='P2P')
        plt.legend()
        plt.ylabel(y_label)

        plt.xticks(range(0, len(x_values) * 2, 2), x_values)
        plt.tight_layout()
        plt.savefig(self.results_directory.joinpath(str(title) + ".png"))
        plt.show()
        plt.close()

        ########################################################################################################################

        title = 'Equality index (1_equal 0_unequal) '
        x_label = 'Market size (No. of participants involved) '
        y_label = 'Equality index'

        ########################################################################################################################
        # Equality index

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        mean_value = list()
        ci_lower = list()
        ci_upper = list()
        for i, n_flat in enumerate(x_values):
            value_list = list()
            for simulation in self.simulation_results:
                # Calculate the sample mean and standard deviation
                value_list.append(equality_index(simulation[i]))
            mean = sum(value_list) / len(value_list)
            std_dev = stats.tstd(value_list)
            mean_value.append(mean)
            # Calculate the 95% confidence interval

            interval = stats.t.interval(alpha=0.95, df=len(value_list) - 1, loc=np.mean(value_list),
                                        scale=stats.sem(value_list))
            ci_lower.append(interval[0])
            ci_upper.append(interval[1])

        d = {'col1': mean_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
        equality_index_pool_price_0 = pd.DataFrame(data=d)
        equality_index_pool_price_0.to_csv(
            self.results_directory.joinpath("equality_index_p_coop_" + p_coop + "_p_pool_" + p_pool + "_benefits.csv"),
            index=True)

        fig, ax = plt.subplots()
        x = x_values

        ax.plot(x, mean_value, color=purple, linestyle='dashed', marker='o')
        ax.fill_between(
            x, ci_lower, ci_upper, color=purple, alpha=.15)
        ax.set_ylim(ymin=0)
        ax.set_title(title)
        fig.autofmt_xdate(rotation=45)
        plt.xticks(x_values)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        # plt.show()
        plt.savefig(self.results_directory.joinpath(str(title) + "test.png"))
        plt.close()

        # Equality index based on savings --------------------------------------------------------------------

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        mean_value = list()
        ci_lower = list()
        ci_upper = list()
        for i, n_flat in enumerate(x_values):
            value_list = list()
            count = 1

            for simulation, pv_consume, grid_buy, pool_buy in zip(self.simulation_results,
                                                                  self.simulation_results_pv_consume,
                                                                  self.simulation_results_grid_buy,
                                                                  self.simulation_results_pool_buy):
                bill_bau = [(_solar + _grid + _pool) * self.grid_buy_price for _solar, _grid, _pool in
                            zip(pv_consume[i], grid_buy[i], pool_buy[i])]
                benefits = simulation[i]

                relative_savings = [benefit / bill for benefit, bill in zip(benefits, bill_bau)]

                value_list.append(equality_index(relative_savings))
            mean = sum(value_list) / len(value_list)
            std_dev = stats.tstd(value_list)
            mean_value.append(mean)
            # Calculate the 95% confidence interval

            interval = stats.t.interval(alpha=0.95, df=len(value_list) - 1, loc=np.mean(value_list),
                                        scale=stats.sem(value_list))
            ci_lower.append(interval[0])
            ci_upper.append(interval[1])

        d = {'col1': mean_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
        equality_index_pool_price_0 = pd.DataFrame(data=d)
        equality_index_pool_price_0.to_csv(
            self.results_directory.joinpath("equality_index_p_coop_" + p_coop + "_p_pool_" + p_pool + ".csv"),
            index=True)

        fig, ax = plt.subplots()
        x = x_values

        ax.plot(x, mean_value, color=purple, linestyle='dashed', marker='o')
        ax.fill_between(
            x, ci_lower, ci_upper, color=purple, alpha=.15)
        ax.set_ylim(ymin=0.5)
        ax.set_title(title)
        fig.autofmt_xdate(rotation=45)
        plt.xticks(x_values)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        # plt.show()
        plt.savefig(self.results_directory.joinpath(str(title) + "test.png"))
        plt.close()

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        mean_value = list()
        ci_lower = list()
        ci_upper = list()
        for i, n_flat in enumerate(x_values):
            value_list = list()
            count = 1

            for simulation, pv_consume, grid_buy, pool_buy in zip(self.simulation_results,
                                                                  self.simulation_results_pv_consume,
                                                                  self.simulation_results_grid_buy,
                                                                  self.simulation_results_pool_buy):
                bill_bau = [(_solar + _grid + _pool) * self.grid_buy_price for _solar, _grid, _pool in
                            zip(pv_consume[i], grid_buy[i], pool_buy[i])]
                benefits = simulation[i]

                relative_savings = [benefit / bill for benefit, bill in zip(benefits, bill_bau)]

                value_list.append(equality_index(bill_bau))
            mean = sum(value_list) / len(value_list)
            std_dev = stats.tstd(value_list)
            mean_value.append(mean)
            # Calculate the 95% confidence interval

            interval = stats.t.interval(alpha=0.95, df=len(value_list) - 1, loc=np.mean(value_list),
                                        scale=stats.sem(value_list))
            ci_lower.append(interval[0])
            ci_upper.append(interval[1])

        d = {'col1': mean_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
        equality_index_pool_price_0 = pd.DataFrame(data=d)
        equality_index_pool_price_0.to_csv(
            self.results_directory.joinpath(
                "equality_index_p_coop_" + p_coop + "_p_pool_" + p_pool + "energy_costs.csv"),
            index=True)

        fig, ax = plt.subplots()
        x = x_values

        ax.plot(x, mean_value, color=purple, linestyle='dashed', marker='o')
        ax.fill_between(
            x, ci_lower, ci_upper, color=purple, alpha=.15)
        ax.set_ylim(ymin=0.5)
        ax.set_title(title)
        fig.autofmt_xdate(rotation=45)
        plt.xticks(x_values)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        # plt.show()
        plt.savefig(self.results_directory.joinpath(str(title) + "test.png"))
        plt.close()

        ########################################################################################################################
        # NPV Cooperative

        title = 'Net Present Value (NPV) (cooperative)'
        x_label = 'Market size (No. of participants involved) '
        y_label = 'NPV'
        # -----------------------------------------------------------------------------------------------------

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        mean_value = list()
        ci_lower = list()
        ci_upper = list()
        for i, n_flat in enumerate(x_values):
            npv_list = list()

            for simulation, grid_sell, pv_consume, pool_sell, grid_buy in zip(self.simulation_results,
                                                                              self.simulation_results_grid_sell,
                                                                              self.simulation_results_pv_consume,
                                                                              self.simulation_results_pool_sell,
                                                                              self.simulation_results_grid_buy):

                """ NPV P2P  """

                energy_sold = (sum(grid_sell[i]) * self.grid_sell_price) + \
                              ((sum(pv_consume[i]) + sum(pool_sell[i])) * self.coop_sell_price)



                cash_flows = [energy_sold - (self.operation_maintenance_cost_per_kW * self.kwp_installed)]
                cash_flows = cash_flows * self.pv_lifetime
                # add upfront installation costs to the list
                cash_flows.insert(0, -1 * (self.price_per_kwp * self.kwp_installed))

                # np.npv(rate,value)
                net_present_value = npf.npv(self.discount_rate, cash_flows)
                npv_list.append(net_present_value)

                if i == 0:
                #     print(sum(grid_sell[i]) )
                #     print(sum(pv_consume[i]))
                #     print(energy_sold)
                #     print(cash_flows)
                    print(cash_flows)
                    print(net_present_value)
                    print((sum(pv_consume[i]) + sum(pool_sell[i])))
                    print(sum(grid_sell[i]))
                    print(energy_sold)


                """ NPV collective """

                # # if i ==4:
                # #     print(f'grid sell {sum(grid_sell[i])}')
                # #     print(f'pool sell {sum(pool_sell[i])}')
                # #     print(f'pv consume sell {sum(pv_consume[i])}')
                # #     print(f'pv grid buy {sum(grid_buy[i])}')
                #
                # energy_sold = (sum(grid_sell[i]) * self.grid_sell_price) + \
                #               ((sum(pv_consume[i]) + sum(pool_sell[i])) * self.coop_sell_price) - \
                #               (sum(grid_buy[i]) * (self.grid_buy_price - self.coop_sell_price))
                #
                # cash_flows = [energy_sold - (self.operation_maintenance_cost_per_kW * self.kwp_installed)]
                #
                # cash_flows = cash_flows * self.pv_lifetime
                # # add upfront installation costs to the list
                # cash_flows.insert(0, -1 * (self.price_per_kwp * self.kwp_installed))
                #
                # # np.npv(rate,value)
                # net_present_value = npf.npv(self.discount_rate, cash_flows)
                # npv_list.append(net_present_value)

                print('##########')

            #print(npv_list)

            mean = sum(npv_list) / len(npv_list)
            std_dev = stats.tstd(npv_list)
            mean_value.append(mean)
            interval = stats.t.interval(0.95, len(npv_list) - 1, loc=mean, scale=std_dev / len(npv_list) ** 0.5)
            ci_lower.append(interval[0])
            ci_upper.append(interval[1])

        d = {'col1': mean_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}
        cooperative_npv = pd.DataFrame(data=d)

        cooperative_npv.to_csv(
            self.results_directory.joinpath("cooperative_npv_p_coop_" + p_coop + "_p_grid_" + p_grid_sell + ".csv"),
            index=True)

        def thousands(x, pos):
            """The two arguments are the value and tick position."""
            return f'{x * 1e-3:1.1f}k'

        fig, ax = plt.subplots()
        x = x_values
        ax.plot(x, mean_value, color=purple, linestyle='dashed', marker='o')
        ax.fill_between(
            x, ci_lower, ci_upper, color=purple, alpha=.15)
        ax.set_title(title)
        # Use automatic StrMethodFormatter
        ax.yaxis.set_major_formatter(thousands)
        plt.xticks(x_values)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid()
        # plt.show()
        plt.savefig(self.results_directory.joinpath(str(title) + ".png"))
        plt.close()

        ########################################################################################################################
        # Savings by consumption group

        # Annual electricity bill BAU and P2P market

        title = 'slice at 20'

        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        x_values = [len(flats_per_run) for flats_per_run in self.simulation_results[0]]

        fig, ax = plt.subplots()  # create figure and axes

        fig.tight_layout()

        for i, n_flat in enumerate(x_values):

            annual_load = list()
            bau_costs = list()
            annual_benefits = list()

            for simulation, pv_consume, grid_buy, pool_buy in zip(self.simulation_results,
                                                                  self.simulation_results_pv_consume,
                                                                  self.simulation_results_grid_buy,
                                                                  self.simulation_results_pool_buy):
                annual_load.extend([(_solar + _grid + _pool) for _solar, _grid, _pool in
                                    zip(pv_consume[i], grid_buy[i], pool_buy[i])])
                bau_costs.extend([(_solar + _grid + _pool) * self.grid_buy_price for _solar, _grid, _pool in
                                  zip(pv_consume[i], grid_buy[i], pool_buy[i])])

                annual_benefits.extend(simulation[i])

            data_table = pd.DataFrame(
                {'annual_load': annual_load, 'bau_costs': bau_costs, 'annual_benefits': annual_benefits})
            data_table.to_csv(
                self.results_directory.joinpath("bills_" + str(n_flat) + "_coop_" + str(self.coop_sell_price)
                                                + "_pool_" + str(self.pool_buy_price) + "_.csv"))

            # def set_box_color(bp, color):
            #     plt.setp(bp['boxes'], color="0")
            #     plt.setp(bp['boxes'], facecolor=color)
            #     plt.setp(bp['whiskers'], color="0")
            #     plt.setp(bp['caps'], color="0")
            #     plt.setp(bp['medians'], color="0")
            #
            # # define the bins for energy consumption
            # bins = [0, 1900, 3100,  4600, float('inf')]
            #
            # # categorize the equivalent savings based on energy consumption bins
            # equivalent_savings_bins = pd.cut(annual_load, bins, labels=['0-1900', '1901-3100', '3101-4600', '4600+'])
            #
            # # create a dataframe from the two lists and the equivalent savings bins
            # df = pd.DataFrame({'annual_energy_consumption': annual_load,
            #                    'equivalent_savings': relative_benefits,
            #                    'equivalent_savings_bins': equivalent_savings_bins})
            #
            # # create a box plot
            # #fig, ax = plt.subplots(figsize=(10, 6))
            # box = plt.boxplot([df[df['equivalent_savings_bins'] == '0-1900']['equivalent_savings'],
            #             df[df['equivalent_savings_bins'] == '1901-3100']['equivalent_savings'],
            #             df[df['equivalent_savings_bins'] == '3101-4600']['equivalent_savings'],
            #              df[df['equivalent_savings_bins'] == '4600+']['equivalent_savings']], patch_artist=True, showfliers=False)
            #
            # set_box_color(box, green)
            #
            #
            # ax.set_xticklabels(['0-1900', '1901-3100', '3101-4600', '4600+'])
            # ax.set_xlabel('Equivalent Savings Bins')
            # ax.set_ylabel('Equivalent Savings')
            # ax.set_title('Box Plot of Equivalent Savings by Energy Consumption Bins')
            #
            #
            # plt.savefig(self.results_directory.joinpath(title + str(x) + ".png"))
            # # plt.show()
            # plt.close()

            ###############

