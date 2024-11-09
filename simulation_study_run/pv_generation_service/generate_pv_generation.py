from __future__ import annotations
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import pandas as pd
from csv import DictReader
from pathlib import Path
from datetime import datetime
from common_functions.objects_proc import write_data_from_object_to_csv_file, load_data_from_dict_to_objects
import sys
import csv

class PvGenerationService:
    """This class generates PV load profile."""

    def __init__(self, init_data_filepath: Path, results_directory: Path):
        self.init_data_filepath = init_data_filepath
        self.results_directory = results_directory
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

        self.generate_pvgeneration()

    def _get_load_from_init_data_file(self) -> dict:
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

    def generate_pvgeneration(self):
        """This function creates 30 PV generation data based on the location entered.
         Return True if result file was created"""

        # set time period
        times_in_h = pd.date_range(start=self.start_stat_period, end=self.end_stat_period, freq='H')

        # build PV system
        # Select PV module and inverter
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        cec_inverters = pvlib.pvsystem.retrieve_sam('CECInverter')

        module = sandia_modules['Schott_Solar_ASE_250_DGF_50__250___2007__E__']
        inverter = cec_inverters['ABB__PVI_3_0_OUTD_S_US__208V_']

        location = Location(latitude=self.lat, longitude=self.long, tz=self.time_zone, altitude=self.building_altitude,
                            name=self.location_name)

        temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        system = PVSystem(surface_tilt=35, surface_azimuth=150, module_parameters=module, inverter_parameters=inverter,
                          temperature_model_parameters=temperature_parameters, modules_per_string=4,
                          strings_per_inverter=1)

        # Get poa data for 2013 (366 days) half hourly

        poa_data, meta, inputs = pvlib.iotools.get_pvgis_hourly(self.lat, self.long, start=2013, end=2013,
                                                                raddatabase="PVGIS-SARAH2", components=True,
                                                                surface_tilt=self.tilt, \
                                                                surface_azimuth=self.azimuth, outputformat='json',
                                                                usehorizon=True, userhorizon=None, pvcalculation=False, \
                                                                peakpower=None, pvtechchoice='crystSi',
                                                                mountingplace='free', loss=0, trackingtype=0,
                                                                optimal_surface_tilt=False, \
                                                                optimalangles=False,
                                                                url='https://re.jrc.ec.europa.eu/api/v5_2/',
                                                                map_variables=True, timeout=30)

        poa_data["poa_diffuse"] = poa_data["poa_sky_diffuse"] + poa_data["poa_ground_diffuse"]
        poa_data["poa_global"] = poa_data["poa_diffuse"] + poa_data["poa_direct"]

        poa_data.index = times_in_h

        # interpolate poa data to half hourly
        poa_data_30min = poa_data.resample('30T').interpolate()

        # Build PV data with 30 min data

        modelchain = ModelChain(system, location)
        modelchain.run_model_from_poa(poa_data_30min)

        pv_output = pd.DataFrame(modelchain.results.ac, columns=["kWh/hh"])
        # divided by 2 to account for half hourly data and multiplied by 37 to receive 37kWp output
        # divided by 1000 to receive kWh
        pv_output["kWh/hh"] = pv_output["kWh/hh"] / 2 * self.kwp_installed / 1000
        # remove negative values
        pv_output[pv_output < 0] = 0

        pv_output.index.name = 'DateTime'
        pv_output.to_csv(self.results_directory.joinpath("pv_output_" + str(self.start_stat_period.year) + "_" +
                                                         str(self.kwp_installed) + "kWp.csv"))