def files(start_idx_str, num_exp):
    """
    Generate the filenames for all experiment IDs.

    :param start_idx: start index for experiments
    :type start_idx: int
    :param num_exp: number of experiments to run
    :type num_exp: int
    """
    log_files = []
    results_files = []
    models_folders = []

    for i in range(num_exp):
        idx_str = start_idx_str + '_' + str(i)

        log_files.append("logs/" + idx_str + ".log")
        # results_files.append(idx + "_results.csv")
        # models_folders.append(idx + "_models")

    return log_files #, results_files, models_folders
