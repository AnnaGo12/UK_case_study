from numba import njit
import numpy as np
import time

def convert_dataset(filename: str)-> str:
    pass


def get_init_random_data(n_len, generation_for_one_flat=0.15) -> tuple:
    consumption = np.random.rand(n_len) * generation_for_one_flat * 2.0
    generation = generation_for_one_flat * n_len
    return consumption, generation


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
    for i, flat_requests in enumerate(flats_requests):
        if flat_requests < 0:
            buy_request_flats.append(flat_requests)
            buy_request_flats_ids.append(i)
        else:
            sell_request_flats.append(flat_requests)
            sell_request_flats_ids.append(i)

    print(buy_request_flats)
    print(sell_request_flats)
    print(buy_request_flats_ids)
    print(sell_request_flats_ids)

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
        for i in range(len(buy_request_flats)):
            if buy_request_flats[i] * (-1) <= every_flat_pool_max_energy_buy:
                flat_money_change[
                    buy_request_flats_ids[buy_request_flats_new_indices[i]]] = buy_request_flats[i] * p_pool_buy
                pool_sell_request += buy_request_flats[i]
                number_of_buy_request -= 1
            else:
                flat_energy_buy_on_grid = (-1) * buy_request_flats[i] - every_flat_pool_max_energy_buy
                flat_money_buy_on_grid = flat_energy_buy_on_grid * p_grid_buy
                flat_money_change[
                    buy_request_flats_ids[buy_request_flats_new_indices[i]]] = -flat_money_buy_on_grid \
                                                                               - every_flat_pool_max_money_buy
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
        for i in range(len(sell_request_flats)):
            if sell_request_flats[i] <= every_flat_pool_max_energy_sell:
                flat_money_change[
                    sell_request_flats_ids[sell_request_flats_new_indices[i]]] = sell_request_flats[i] * p_pool_sell
                pool_buy_request += sell_request_flats[i]
                number_of_sell_request -= 1
            else:
                flat_energy_sell_on_grid = sell_request_flats[i] - every_flat_pool_max_energy_sell
                flat_money_sell_on_grid = flat_energy_sell_on_grid * p_grid_sell
                flat_money_change[
                    sell_request_flats_ids[sell_request_flats_new_indices[i]]] = \
                    flat_money_sell_on_grid + every_flat_pool_max_money_sell
                pool_buy_request += every_flat_pool_max_energy_sell
                number_of_sell_request -= 1
            if number_of_sell_request != 0:
                every_flat_pool_max_energy_sell = (-1) * pool_buy_request / number_of_sell_request
                every_flat_pool_max_money_sell = every_flat_pool_max_energy_sell * p_pool_sell

    return flat_money_change


def run_tests():
    p_pool_buy = 0.2
    p_pool_sell = 0.2
    p_grid_buy = 0.3404
    p_grid_sell = 0.04
    generation = 0.6

    flat_consumption_tests = [np.array([0.143, 0.199, 0.125, 0.229]),
                              np.array([0.140, 0.110, 0.200, 0.120]),
                              np.array([0.215, 0.154, 0.095, 0.190])]
    for i, flats_consumption in enumerate(flat_consumption_tests):
        str_input = f"Test №{i}\n"
        str_input += f"p_pool_buy = {p_pool_buy}\n"
        str_input += f"p_pool_sell = {p_pool_sell}\n"
        str_input += f"p_grid_buy = {p_grid_buy}\n"
        str_input += f"p_grid_sell = {p_grid_sell}\n"
        str_input += f"sum_generation = {generation}\n"
        str_input += f"flats_consumption:\n{flats_consumption}\n"

        res = iteration_raw(flats_consumption, p_pool_buy, p_pool_sell, p_grid_buy, p_grid_sell, generation)
        str_input += f"money change results:\n{res}\n"
        print(str_input)


def run_time_test():
    n_flats_in_iteration = 62
    p_pool_buy = 0.2
    p_pool_sell = 0.2
    p_grid_buy = 0.3404
    p_grid_sell = 0.04
    flats_consumption, generation = get_init_random_data(n_len=n_flats_in_iteration)
    n_times = 17520

    start_time = time.time()
    iteration_raw(flats_consumption, p_pool_buy, p_pool_sell, p_grid_buy, p_grid_sell, generation)
    print(f"Time of work first function call {time.time() - start_time} seconds")

    start_time = time.time()
    for _ in range(n_times):
        iteration_raw(flats_consumption, p_pool_buy, p_pool_sell, p_grid_buy, p_grid_sell, generation)
    print(f"Time of work {n_times} iterations with {n_flats_in_iteration} flats - {time.time() - start_time} seconds")


def main():
    # run this function to show results of algorithm tests proc
    run_tests()

    # run this function to show time of work with random input
    # run_time_test()


if __name__ == "__main__":
    main()
