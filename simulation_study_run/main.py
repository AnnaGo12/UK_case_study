from datetime import datetime
from pathlib import Path
from init_data_service.data_set_convert import dataset_convert
from init_data_service.initial_data import InitialDataService
from simulation_service.simulation import SimulationService
from post_proc_service.post_proc import PostProcService
from pv_generation_service.generate_pv_generation import PvGenerationService
from init_data_service.data_set_convert import *

def test_convert_init_dataset_function():
    dataset_convert(
        init_dataset_filepath=Path(r"C:\Projects\community\simulation_study_run\data\CC_LCL-FullData.csv"),
        directory=Path(r"C:\Projects\community\simulation_study_run\data\2013"),
        #directory=Path(r"C:\Projects\community\simulation_study_run\data\2013_unfiltered_dataset"),
        relative_error_limit=0.1,
        start_stat_period=datetime(year=2013, month=1, day=1, hour=0, minute=0),
        end_stat_period=datetime(year=2013, month=12, day=31, hour=23, minute=30),
        chunksize=100000
    )

def test_initial_data_service(filepath: Path):
    initial_data_service = InitialDataService(
        num_monte_carlo_runs=100,
        max_flats_number=4,
        num_flats_step=2,
        consumption_data_directory=Path(r"data\2013"),
        generation_data_filepath=Path(r"data\pv_output_2013_37kWp.csv"),
        start_stat_period=datetime(year=2013, month=1, day=1, hour=0, minute=0),
        end_stat_period=datetime(year=2013, month=12, day=31, hour=23, minute=0),
        pool_sell_price=0.00,
        pool_buy_price=0.00,
        grid_sell_price=0.05,
        grid_buy_price=0.34,
        coop_sell_price=0.00, #############
        price_per_kwp=1016,
        kwp_installed=37,
        lat=51.469,
        long=-0.105,
        time_zone='Europe/London',
        building_altitude=26,
        location_name='ElmoreHouse',
        tilt=35,
        azimuth=-30,
        pv_lifetime=25,
        flats_num_list=[2, 4, 8, 16, 32, 64],
    )
    initial_data_service.write_data_in_csv_file(filepath)
    initial_data_service.read_data_from_csv_file(filepath)
    return initial_data_service

def test_monte_carlo_simulation():
    init_filepath = r"data\2013_init_settings.csv"
    results_directory = r"data\2013_results_speed_test"
    initial_data_service = test_initial_data_service(Path(init_filepath))
    simulation_service = SimulationService(init_data_filepath=Path(init_filepath),
                                           results_directory=Path(results_directory))
    simulation_service.run_speed_test()

def test_post_proc():
    init_filepath = r"data\2013_init_settings.csv"
    results_directory = r"data\2013_results_speed_test"
    post_proc_directory = r"data\2013_post_proc_speed_test"
    post_proc = PostProcService(init_data_filepath=Path(init_filepath), results_directory=Path(results_directory),
                                post_proc_directory=Path(post_proc_directory))
    post_proc.plot_graph()

def generate_pv_generation_data():
    init_filepath = r"data\2013_init_settings_last_version.csv"
    results_directory = r"data"
    pv_generation = PvGenerationService(init_data_filepath=Path(init_filepath),
                                        results_directory=Path(results_directory))

def main():
    #test_convert_init_dataset_function()  # converts large load dataset in multiple files
    #test_initial_data_service(Path(r"data\2013_init_settings.csv"))  #  saves initialisation data for simulation
    #generate_pv_generation_data()  # generates half-hourly PV load profiles
    test_monte_carlo_simulation()  # runs market simulation
    test_post_proc()  # analyses results


if __name__ == "__main__":
    main()
