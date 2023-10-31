import time
def files(args):
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments
    :type start_idx: int
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = args.methods
    # results_files.extend([args.dataset, args.net, str(int(time.time()))])
    file_name = '_'.join(results_files)

    for i in range(args.get_num_exp()):
        idx_str = file_name + '_' + str(i)

        log_files.append("logs/" + idx_str + "1030.log")
        # results_files.append(idx + "_results.csv")
        # models_folders.append(idx + "_models")

    return log_files #, results_files, models_folders
