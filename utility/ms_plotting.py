"""
    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Local
from Jingle.utility.info_write_load_utils import read_experiment_info_from_files
from Jingle.utility.hierarchy_env import are_two_environments_equal
from Jingle.utility.plotting import COLOURS


NUM_INITIAL_ENTRIES_TO_IGNORE = 0

BAR_FIGSIZE = (26, 12)
BAR_FIGSIZE_WITH_LEGEND = (26, 12)
BAR_LEGEND_FONT_SIZE = 30
BAR_ANNOT_FONT_SIZE = 25
BAR_AXIS_FONT_SIZE = 30
BAR_YLABEL_FONT_SIZE = 26
CURVE_FIG_SIZE = (16, 11)
CURVE_LEGEND_FONT_SIZE = 40
CURVE_XYLABEL_FONT_SIZE = 52
CURVE_XYTICK_SIZE = 52

TIME_ELAPSED = 'time_elapsed'
ALLOC_PER_UNIT_LOAD = 'alloc_per_unit_load'
EVENT_TIME_PERIOD = 'event_time_period'
EVENT_START_TIME = 'event_start_time'
EVENT_END_TIME = 'event_end_time'
P99 = 'p99'
AVG_LATENCY = 'avg_latency'
CUM_P99 = 'cum_p99'
CUM_AVG_LATENCY = 'cum_avg_latency'
LOAD = 'load'

FIELDS_TO_PLOT = ['p99', 'avg_latency']
# FIELDS_TO_PLOT = [P99, CUM_P99]
# FIELDS_TO_PLOT = [P99, CUM_P99]


# Utilities for loading data -----------------------------------------------------------------------
def read_data_from_client_data_log(file_name):
    """ Reads data. """
    df = pd.read_csv(file_name, index_col=0)
    ret = []
    row_counter = 0
    for _, row in df.iterrows():
        row_counter += 1
        if row_counter >= NUM_INITIAL_ENTRIES_TO_IGNORE:
            curr_data = {EVENT_TIME_PERIOD: row['event_end_time'] - row['event_start_time'],
                         P99: row['p99'],
                         AVG_LATENCY: row['avg_latency'],
                         EVENT_START_TIME: row['event_start_time'],
                         EVENT_END_TIME: row['event_end_time'],
                         LOAD: row['load'],
                        }
            ret.append(curr_data)
    ret.sort(key= lambda elem: elem['event_start_time'])
    exp_start_time = ret[0]['event_start_time']
    cum_p99_sum = 0
    cum_avg_lat_sum = 0
    for idx, elem in enumerate(ret):
        elem[TIME_ELAPSED] = elem[EVENT_START_TIME] - exp_start_time
        cum_p99_sum += elem[P99]
        cum_avg_lat_sum += elem[AVG_LATENCY]
        elem[CUM_P99] = cum_p99_sum / (idx + 1)
        elem[CUM_AVG_LATENCY] = cum_avg_lat_sum / (idx + 1)
    print('Retrieved %d (ignored %d) rows from %s'%(
          row_counter, NUM_INITIAL_ENTRIES_TO_IGNORE, file_name))
    return ret

def process_data_log_files_for_each_method(meth_data_logs_names):
    """ Processes all log files in the directory. """
    meth_data_logs = {}
    for meth, file_names in meth_data_logs_names.items():
        # only reading first file for now.
        meth_data_logs[meth] = read_data_from_client_data_log(file_names[0])
    return meth_data_logs

def get_data_logs_file_names(results_dir, method_order, env_descr):
    """ Loads the results from the results directory. """
    first_env = (None, None)
    first_info = None
    meth_data_logs_names = {}
    # Internal function to verify that the environments are equal and return the client data log.
    def _process_files_in_curr_meth_results_dir(meth_dir_name, meth):
        """ Processes the current file. """
        nonlocal first_env
        nonlocal first_info
        # Preliminaries, create key if not present and load environment ----------------------------
        curr_env, curr_info = read_experiment_info_from_files(meth_dir_name)
        if first_env[0]:
            if not are_two_environments_equal(first_env[1], curr_env):
                report_str = 'Environments not equal:\n%s:\n%s\n\n%s:\n%s'%(
                    first_env[0], first_env[1].write_to_file(None),
                    meth_dir_name, curr_env.write_to_file(None))
                raise ValueError(report_str)
            if not curr_info == first_info:
                report_str = 'Experimental info are not equal:\n%s: %s\n%s: %s'%(
                    first_env[0], first_info, meth_dir_name, curr_info)
                raise ValueError(report_str)
        else:
            first_env = (meth_dir_name, curr_env)
            first_info = curr_info
        # Next load the data logs ------------------------------------------------------------------
        if not (meth in meth_data_logs_names):
            meth_data_logs_names[meth] = []
        meth_data_logs_names[meth].append(os.path.join(meth_dir_name, 'ca-client.csv'))
    # Go over each method --------------------------------------------------------------------------
    dir_list = [os.path.join(results_dir, elem) for elem in os.listdir(results_dir)
                if env_descr in elem]
    for meth_dir_name in dir_list:
        for meth in method_order:
            if meth in meth_dir_name:
                _process_files_in_curr_meth_results_dir(meth_dir_name, meth)
    env = first_env[1]
    experiment_info = first_info
    return env, experiment_info, meth_data_logs_names

def load_data_from_results_dir(results_dir, method_order, env_descr):
    """ Loads data from the results dir. """
    env, experiment_info, meth_data_logs_names = get_data_logs_file_names(results_dir, method_order,
                                                                          env_descr)
    meth_data_logs = process_data_log_files_for_each_method(meth_data_logs_names)
    return env, experiment_info, meth_data_logs

# Utility for generating curves --------------------------------------------------------------------
def _get_grid_pt_vals_for_method(data_log_for_meth, fields_to_plot, grid_pts):
    """ Generate grid pt vals for a single method. """
    ret = {}
    for fld in fields_to_plot:
        y_vals = [elem[fld] for elem in data_log_for_meth]
        x_vals = [elem[TIME_ELAPSED] for elem in data_log_for_meth]
        ret[fld] = np.interp(grid_pts, x_vals, y_vals)
    return ret

def _gen_curves_for_field(field_name, fld_results_meth_dict, grid_pts, method_order,
                          method_legend_colour_marker_dict, to_plot_legend, options):
    """ Generate curves for the field. """
    # Set plot type  -----------------------------------------------------------------------------
    if options['plot_type'][field_name] == 'plot':
        plot_func = plt.plot
    elif options['plot_type'][field_name] == 'loglog':
        plot_func = plt.loglog
    elif options['plot_type'][field_name] == 'semilogy':
        plot_func = plt.semilogy
    elif options['plot_type'][field_name] == 'semilogx':
        plot_func = plt.semilogx
    elif options['plot_type'][field_name] == 'scatter':
        plot_func = plt.scatter
    else:
        raise ValueError('Unknown plot function %s.'%(options['plot_type'][field_name]))
    plt.rc('xtick', labelsize=CURVE_XYTICK_SIZE)
    plt.rc('ytick', labelsize=CURVE_XYTICK_SIZE)
    fig = plt.figure(figsize=CURVE_FIG_SIZE)
    for method in method_order:
        grid_pts_in_mins = [elem/60 for elem in grid_pts]
        if field_name == CUM_P99:
            plt.plot(grid_pts_in_mins, fld_results_meth_dict[method], marker=',',
                      color=COLOURS[
                          method_legend_colour_marker_dict[method]['colour']],
                      linestyle=method_legend_colour_marker_dict[method][
                          'linestyle'],
                      linewidth=options['line_width'],
                      label=method_legend_colour_marker_dict[method]['legend'],
                      )
            plt.yscale('log')
            plt.ylabel('Avg P99 Latency (ms)', fontsize=CURVE_XYLABEL_FONT_SIZE)
            # plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
            plt.ylim([300,10000])
            lgd = plt.legend(bbox_to_anchor=(-0.15, 1.07, 1.15, 0.2), loc="lower left",
                            mode="expand", borderaxespad=0, ncol=4, fontsize=CURVE_LEGEND_FONT_SIZE,
                            handletextpad=0.2, handlelength=1.5)
        elif field_name == P99:
            plt.plot(grid_pts_in_mins, fld_results_meth_dict[method],
                        alpha=0.2,
                        color=COLOURS[method_legend_colour_marker_dict[method]['colour']],
                        linestyle=method_legend_colour_marker_dict[method]['linestyle'],
                        linewidth= 4,#options['line_width'],
                      )
            plt.scatter(grid_pts_in_mins, fld_results_meth_dict[method],
                        s=600,
                        alpha=0.7,
                        marker=method_legend_colour_marker_dict[method]['marker'],
                        color=COLOURS[method_legend_colour_marker_dict[method]['colour']],
                        linestyle=method_legend_colour_marker_dict[method]['linestyle'],
                        linewidth=options['line_width'],
                        label=method_legend_colour_marker_dict[method]['legend'],
                      )
            plt.yscale('log')
            plt.ylabel('P99 Latency (ms)', fontsize=CURVE_XYLABEL_FONT_SIZE)
            lgd = plt.legend(bbox_to_anchor=(-0.15, 1.07, 1.15, 0.2), loc="lower left",
                            mode="expand", borderaxespad=0, ncol=4, fontsize=CURVE_LEGEND_FONT_SIZE,
                            handletextpad=0)
            for handle in lgd.legendHandles:
                handle.set_sizes([1000])
        else:
            plot_func(grid_pts_in_mins, fld_results_meth_dict[method], marker=',',
                      color=COLOURS[method_legend_colour_marker_dict[method]['colour']],
                      linestyle=method_legend_colour_marker_dict[method]['linestyle'],
                      linewidth=options['line_width'],
                      label=method_legend_colour_marker_dict[method]['legend'],
                      )
    if to_plot_legend:
        pass
        #plt.legend(loc=options['legend_location'], fontsize=60)
    plt.gca().tick_params(which='major', width=5, length=10)
    plt.gca().tick_params(which='minor', width=4, length=7)
    # plt.title(options['fld_to_title'][field_name], fontsize=70)
    plt.xlabel('Time (minutes)', fontsize=CURVE_XYLABEL_FONT_SIZE)
    fig.tight_layout()
    # Save file --------------------------------------------------------
    if options['plot_save_dir']:
        save_file_name = os.path.join(options['plot_save_dir'], field_name + '.pdf')
        plt.savefig(save_file_name)

def gen_curves(meth_data_logs, method_order, fields_to_plot, x_bounds,
               method_legend_colour_marker_dict, to_plot_legend=True,
               options=None):
    """ Generate curves. """
    grid_pts = np.linspace(x_bounds[0], x_bounds[1], num=options['num_grid_pts'])
    grid_vals_for_each_method = \
        {meth: _get_grid_pt_vals_for_method(meth_data_logs[meth], fields_to_plot, grid_pts)
            for meth in method_order}
    metrics = {meth:{} for meth in method_order}
    for fld in fields_to_plot:
        curr_fld_results = {meth: grid_vals_for_each_method[meth][fld] for meth in method_order}
        _gen_curves_for_field(fld, curr_fld_results, grid_pts, method_order,
                              method_legend_colour_marker_dict,
                              to_plot_legend, options)
        for meth in method_order:
            num_data_for_meth = len(meth_data_logs[meth])
            curr_fld_meth_data = grid_vals_for_each_method[meth][fld]
            metrics[meth][fld] = (np.mean(curr_fld_meth_data), np.std(curr_fld_meth_data),
                                  np.std(curr_fld_meth_data) / np.sqrt(num_data_for_meth))
    return metrics


# Main utility for plotting ----------------------------------------------------------------------
def get_plot_options(options, **kwargs):
    """ Obtain plot options. """
    if options is None:
        options = {}
    def _check_for_value_and_set(key, default_value):
        """ Checks for value. """
        if key in options:
            return options[key]
        elif key in kwargs:
            return kwargs[key]
        else:
            return default_value
    ret = {}
    ret['num_grid_pts'] = _check_for_value_and_set('num_grid_pts', 100)
    ret['err_bar_type'] = _check_for_value_and_set('err_bar_type', 'fill_between')
    ret['line_width'] = _check_for_value_and_set('line_width', 4)
    ret['legend_location'] = _check_for_value_and_set('legend_location', 'upper left')
    ret['legend_font_size'] = _check_for_value_and_set('legend_font_size', 30)
    ret['to_plot_bar_legend'] = _check_for_value_and_set('to_plot_bar_legend', True)
    ret['title_font_size'] = _check_for_value_and_set('title_font_size', 30)
    ret['axis_font_size'] = _check_for_value_and_set('axis_font_size', 30)
    ret['xtick_font_size'] = _check_for_value_and_set('xtick_font_size', 20)
    ret['ytick_font_size'] = _check_for_value_and_set('ytick_font_size', 20)
    ret['marker_size'] = _check_for_value_and_set('marker_size', 20 * (4**3))
    ret['util_compute_grid_size'] = _check_for_value_and_set('util_compute_grid_size', 1000)
    ret['fld_to_title'] = _check_for_value_and_set('fld_to_title',
        {P99: 'P99 Latency', AVG_LATENCY: 'Average Latency',
         CUM_P99: 'Time-Avg P99 Latency', CUM_AVG_LATENCY: 'Average Latency (Cum)'})
    ret['plot_save_dir'] = _check_for_value_and_set('plot_save_dir', None)
    ret['plot_type'] = _check_for_value_and_set('plot_type',
        {P99: 'scatter', AVG_LATENCY: 'plot',
         CUM_P99: 'semilogy', CUM_AVG_LATENCY: 'plot'})
    ret['plot_save_dir'] = _check_for_value_and_set('plot_save_dir', None)
    return ret

def plot_ms_results(results_dir, method_order, env_descr, x_bounds,
                    method_legend_colour_marker_dict,
                    to_plot_legend=True, options=None):
    """ Plot results. """
    options = get_plot_options(options)
    _, _, meth_data_logs = load_data_from_results_dir(results_dir, method_order, env_descr)
    metrics = gen_curves(meth_data_logs, method_order, FIELDS_TO_PLOT, x_bounds,
                         method_legend_colour_marker_dict, to_plot_legend, options)
    for meth in method_order:
        report_str = '%s::  '%(meth)
        for fld in FIELDS_TO_PLOT:
            report_str += '%s= %0.3f +/- %0.4f,    '%(
                fld, metrics[meth][fld][0], metrics[meth][fld][1])
        print(report_str)
    plt.show()

def plot_qps():
    load_variation = [1, 1, 1, 2, 2, 3, 3, 6, 7, 6, 4, 4, 3, 3, 2, 1, 1, 1, 1]
    base_qps = 25
    time_interval = 60
    round = len(load_variation)
    flat = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18]
    flat2 = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16,
            16, 17, 17, 18, 18, 18]

    x = [time_interval * i for i in flat]
    y = [base_qps*load_variation[i] for i in flat2]

    print(x)
    print(y)
    plt.plot(x, y, color = 'r', linewidth= 2, linestyle = "dashed", label = 'qps')
    plt.xlabel("time",  fontsize=20)
    plt.ylabel("qps",  fontsize=20)
    plt.title("Load",  fontsize=20)
    plt.rc('xtick', labelsize=CURVE_XYTICK_SIZE)
    plt.rc('ytick', labelsize=CURVE_XYTICK_SIZE)
    plt.legend()

    plt.xticks(np.arange(min(x), max(x) + 1, 120.0))
    plt.yticks(np.arange(min(y), max(y) + 1, 50.0))

    plt.savefig("qps.png", dpi=300)
    plt.show()


def plot_allocs():
    time_interval = 60

    x = [time_interval * i for i in range(17)]

    # allocations
    k8s = [1 for i in range(17)]
    aslearn = [1, 1.146, 2, 3.846, 4 ,4 ,4 , 3.3, 2, 2, 1.866, 1, 1, 1, 1, 1, 1]

    plt.plot(x, k8s, color='y', linewidth=2, label = 'k8s-AS')
    plt.plot(x, aslearn, color='r', linewidth=2, label = 'Jingle-AS')
    plt.xlabel("time", fontsize=10)
    plt.ylabel("# allocations", fontsize=10)
    plt.legend()

    plt.yticks(np.arange(0, max(aslearn) + 1))

    plt.savefig("allocs.png", dpi=300)
    plt.show()

def plot_latency():
    time_interval = 60

    x = [time_interval * i for i in range(17)]

    # allocations
    k8s =     [15.82, 18.01, 25.92, 27.09, 37.04, 51.47, 231.04, 37.25, 48.80, 24.47, 12.89, 19.42, 21.92, 23.01, 15.46, 10.62, 14.89]
    aslearn = [14.71, 22.71, 21.33, 16.16, 15.84, 14.13, 11.27, 9.9,    21.77, 13.76, 14.4,   16.4, 11.44, 10.4,  10.99, 11.65, 14.16]

    SLO = [20 for i in range(17)]

    plt.plot(x, k8s, color='y', linewidth=2, label='k8s-AS')
    plt.plot(x, aslearn, color='r', linewidth=2, label='Jingle-AS')
    plt.plot(x, SLO, color='g', linewidth=1, label='SLO Rquirement', linestyle = "dashed")
    plt.xlabel("time", fontsize=10)
    plt.ylabel("P99 Latency (ms)", fontsize=10)
    plt.legend()

    plt.savefig("lat.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # plot_qps()
    # plot_allocs()
    plot_latency()