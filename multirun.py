import numpy as np
import subprocess
import pathlib, json

if __name__ == "__main__":

    script = 'run_pathint.py' 
    cases = ['ED', 'EDr1', 'gridcells', 'EDd1', 'gridcellsd1', 'MSlow', 'MSmed', 'MShigh', \
             'sensorymed', 'sensoryhigh', 'MSlowd1', 'MSmedd1', 'MShighd1', 'sensorymedd1', 'sensoryhighd1', \
             'nullrotation', 'nullrotationd1']
    dir_loc = './data/testparam/'
    # cases = ['EDd1', 'gridcellsd1', 'MSlowd1', 'MSmedd1', 'MShighd1', 'sensorymedd1', 'sensoryhighd1', \
    #          'nullrotationd1']

    cases = ['ED_redsweep']

    compute = False
    analyse = False
    plot = True

    for case in cases:
        print("##### CASE " + case + " ######")
        experiment = 'remapping' if case not in {'nullrotation', 'nullrotation_sparsity', 'nullrotationd1', 'nullrotationd1_sparsity'} else 'recruitment'
        output = case
        path = pathlib.Path(dir_loc+ output)
        path.mkdir(parents=True, exist_ok=True)

        num_envs = None
        envs_list = None
        num_seeds = None
        seeds_list = None
        ## ENCODING - DECODING
        if case == 'ED':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [128], [-1], 'randclosed-load-polyae', 'M', 'rotation', 10, ''
        elif case == 'ED_redsweep':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, [1, 2, 4, 8, 16, 32, 64], [16, 32, 64], [-1], 'identity', 'M', 'rotation', 30, ''
        elif case == 'ED_nrooms':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, [2, 4, 8, 16, 32, 64, 128, 512], [16], [-1], 'randclosed-load-polyae', 'M', 'rotation', 10, ''
        elif case == 'EDr1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 1, [128], [-1], 'identity', 'M', 'rotation', 10, ''
        elif case == 'gridcells':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 8, [12], [-1], 'randclosed-load-polyae', 'C', 'gridcells', 10, ''
        elif case == 'EDd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [3], [-1], 'randclosed-load-polyae', 'M', 'rotation', np.array([1,9]), '--save_input'
        elif case == 'gridcellsd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'C', 'gridcells', np.array([0,3]), '--save_input'

        ## MIXED-SELECTIVE
        if case == 'MSlow':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'CM', 'parallel', 10, '--input_sepnorm'
        if case == 'MSlow_variances':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [8, 32, 64], [0, 0.5, 1], 'randclosed-load-polyae', 'CM', 'parallel', 10, '--input_sepnorm'
        elif case == 'MSmed':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'CM', 'flexibleGP', 10, '--input_sepnorm'
        elif case == 'MSmed_variances':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, [1, 2, 4, 8, 16], [8, 16, 32, 64], [0.1, 0.2, 0.5, -1], 'randclosed-load-polyae', 'CM', 'flexibleGP', 30, '--input_sepnorm'
        elif case == 'MShigh':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'CM', 'flexible', 10, '--input_sepnorm'
        elif case == 'sensorymed':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'M', 'sensoryGP', 10, ''
        elif case == 'sensoryhigh':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'M', 'sensory', 10, ''
        elif case == 'MSlowd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'CM', 'parallel', np.arange(3), '--input_sepnorm --save_input'
        elif case == 'MSmedd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'CM', 'flexibleGP', np.array([0,1,5]), '--input_sepnorm --save_input'
        elif case == 'MShighd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [4], [4], 'randclosed-load-polyae', 'CM', 'flexible', np.arange(3), '--input_sepnorm --save_input'
        elif case == 'MSrd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 4, [4], [-1], 'randclosed-load-polyae', 'CM', 'flexibler', np.array([0,2]), '--input_sepnorm --save_input'
        elif case == 'MSrd1M':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 4, [4], [-1], 'randclosed-load-polyae', 'Mex', 'flexibler', np.array([0,2]), '--input_sepnorm --save_input'
        elif case == 'sensorymedd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'M', 'sensoryGP', np.array([8,23,10]), '--save_input'
        elif case == 'sensoryhighd1':
            dim_pcs, red, dims, variances, model, conj, encoding, envs_list, extra = \
            1, 8, [4], [4], 'randclosed-load-polyae', 'M', 'sensory', np.arange(3), '--save_input'

        ## NULLSPACE REMAPPING
        elif case == 'nullrotation':
            dim_pcs, red, dims, variances, model, conj, encoding, num_seeds, extra = \
            2, 32, [128], [-1], 'randclosed-load-polyae', 'M', 'rotation', 1, ''
            tagging_sparse = [1]
            tagging_currents = [-10]
        elif case == 'nullrotation_sparsity':
            dim_pcs, red, dims, variances, model, conj, encoding, num_seeds, extra = \
            2, 16, [16], [-1], 'randclosed-load-polyae', 'M', 'rotation', 5, ''
            tagging_sparse = [0.1, 0.25, 0.5, 0.75, 0.9]
            tagging_currents = [-10]
        elif case == 'nullrotationd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_seeds, extra = \
            1, 16, [3], [-1], 'randclosed-load-polyae', 'M', 'rotation', 1, '--save_input'
            tagging_sparse = [1]
            tagging_currents = [-10]
        elif case == 'nullrotationd1_sparsity':
            dim_pcs, red, dims, variances, model, conj, encoding, num_seeds, extra = \
            1, 16, [3], [-1], 'randclosed-load-polyae', 'M', 'rotation', 1, '--save_input'
            tagging_sparse = [0.25, 0.5, 0.75, 1]
            tagging_currents = [-10]
            
        case_scripts = "--dim_pcs " + str(dim_pcs) + " --model " + model + " --model_conj " + \
            conj + " --encoding " + encoding + " " + extra

        simulate = 'minimization'
        script_args = ("--path_type grid --simulate " + str(simulate) + " " + case_scripts + \
                    " --load_id 0 --save --decoder_amp 1 --thresh_amp 1")
        script_args += " --dir ./data/testparam/" + output + "/"

        
        if experiment == "remapping":
            if envs_list is None: envs_list = np.arange(num_envs)
            if compute:
                print("## RUNNING SIMULATIONS ##")
                # Run trials in different environments with different embedding variance and dimensionality
                for dim in dims:
                    print("Dimensions:" + str(dim))
                    if 'red' in locals():
                        neur = list(np.array(red) * dim) if isinstance(red, list) else red * dim
                    for n in (neur if isinstance(neur, list) else [neur]):
                        print("Redundancy:" + str(n/dim) + "/ Neurons:" + str(n))
                        script_args += " --nb_neurons " + str(n) 
                        for sigma in variances:
                            print("Variance:" + str(sigma))
                            for env in envs_list+50:
                                print("Environment:" + str(env))
                                script_args += " --dim_bbox " + str(dim) + " --env " + str(env) + " --embedding_sigma " + str(sigma)
                                command = "python " + script + " " + script_args
                                command_list = command.split()
                                subprocess.run(command)

            if analyse:
                print("## ANALYZING DATA ##")
                # Do analysis
                analyse_script = "python analyse_stats.py --dir_loc " + dir_loc + " --dir " + output
                for s in ['database', 'ratemaps_pfs', 'remapping']:
                    print("Analyse " + s)
                    command = analyse_script + " --compute " + s
                    command_list = command.split()
                    subprocess.run(command)

            if plot:
                print("## PLOTTING RESULTS ##")
                # Do plots
                plot_script = "python plot_analysis.py --dir_loc " + dir_loc + " --dir " + output
                if case == 'ED_redsweep':
                    array = ['redundancy_remap', 'placefields']
                    plot_cases = [['ED_overlap', 'ED_spatialcorr', 'ED_perpcs', 'ED_meanperpfsizes'], ['64,64' , '64,4096']]
                    # array = ['placefields']
                    # plot_cases = [['64,4096']]
                elif case == 'ED_nrooms':
                    array = ['nrooms']
                    plot_cases = [['']]
                elif case == 'MSmed_variances':
                    array = ['dims_remap', 'redundancy_remap']
                    plot_cases = [['MSmed_overlap', 'MSmed_sparsecorr'], ['MSmed_overlap', 'MSmed_sparsecorr']]
                elif case == 'MSlow_variances':
                    array = ['dims_remap']
                    plot_cases = [['MSlow_overlap', 'MSlow_sparsecorr']]
                elif dim_pcs == 2:
                    array = ['measures', 'remap_vec', 'placefields']
                    plot_cases = [['']]
                else:
                    array = ['placefields', 'vis']
                    plot_cases = [['']]
                idx = 0
                for s in array:
                    print("Plot " + s)
                    for plot_case in plot_cases[idx]:
                        command = plot_script + " --plot " + s
                        if plot_case != '':
                            command += " --plot_case " + plot_case
                        command_list = command.split()
                        subprocess.run(command)
                    idx += 1

        elif experiment == "recruitment":
            if seeds_list is None: seeds_list = np.arange(num_seeds)
            if compute:
                print("## RUNNING SIMULATIONS ##")
                # Run standard trial (no inhibition experiment)
                dim = dims[0]
                print("Dimensions:" + str(dim))
                n = red*dim
                script_args += " --nb_neurons " + str(n) 
                sigma = variances[0]
                print("Variance:" + str(sigma))
                env = 0+50
                print("Environment:" + str(env))
                min_args = "--compute_fr" if simulate == 'minimization' else ""
                print("Baseline experiment")
                script_args += " --dim_bbox " + str(dim) + " --env " + str(env) + " --embedding_sigma " + str(sigma) + " " + min_args
                command = "python " + script + " " + script_args
                command_list = command.split()
                subprocess.run(command)

                # Read tagged neurons from standard trial
                patt = "*.json"
                basepath = dir_loc + output
                path = pathlib.Path(basepath)
                standard_file = next(path.rglob(patt))
                with open(standard_file) as res_file:
                    f = json.load(res_file)
                if simulate != 'minimization':
                    active_idx_total = np.array(f['activeidx'])
                else: 
                    maxfr = np.array(f['maxfr'])
                    active_idx_total = np.argwhere(maxfr > 1e-3)[:,0]

                num_active_total = active_idx_total.shape[0]

                for seed in seeds_list:
                    print("Seed = " + str(seed))
                    np.random.seed(seed)
                    active_idx = active_idx_total.copy()
                    num_active = num_active_total
                    tagged_idx = []
                    # incremental inhibition
                    for S in tagging_sparse:
                        # Tagging
                        print("Sparseness = " + str(S))
                        num_active_now = num_active_total*(1-S)
                        num_inhibit = int(num_active - num_active_now)

                        opsin = np.zeros(num_active, dtype=int)
                        opsin[np.random.choice(num_active, num_inhibit, replace=False)] = 1
                        tagged_idx += active_idx[np.where(opsin)].tolist()

                        active_idx = active_idx[np.where(opsin == 0)]
                        num_active = active_idx.shape[0]

                        tagged_idx_str = ' '.join(map(str, tagged_idx))
                        tagged_args = "--tagging_sparse " + str(S)
                        tagged_args += (" --tagged_idx " + tagged_idx_str) if len(tagged_idx) > 0 else ''
                        tagged_args += " --tagging_seed " + str(seed)
                        # Execute inhibition trials on tagged_idx neurons
                        for C in tagging_currents:
                            print("Current = " + str(C))
                            inhibition_args = tagged_args + " --current_amp " + str(C)
                            script_args += " --dim_bbox " + str(dim) + " --env " + str(env) + " --embedding_sigma " + str(sigma) + " " + inhibition_args
                            command = "python " + script + " " + script_args
                            subprocess.run(command)

            if analyse:
                print("## ANALYZING DATA ##")
                # Do analysis
                analyse_script = "python analyse_stats.py --null --dir_loc " + dir_loc + " --dir " + output
                for s in ['database', 'ratemaps_pfs', 'remapping']:
                    command = analyse_script + " --compute " + s
                    command_list = command.split()
                    subprocess.run(command)

            if plot:
                print("## PLOTTING RESULTS ##")
                # Do plots
                plot_script = "python plot_analysis.py --dir_loc " + dir_loc + " --dir " + output
                if case == 'nullrotation_sparsity':
                    array = ['sparsecanon_remap']
                    plot_cases = [['nullrotation_sparsity_overlap', 'nullrotation_sparsity_spatialcorr']]
                elif dim_pcs == 2:
                    array = ['remap_vec']
                    plot_cases = [['']]
                else:
                    array = ['pca']
                    plot_cases = [['']]
                idx = 0
                for s in array:
                    print("Plot " + s)
                    for plot_case in plot_cases[idx]:
                        command = plot_script + " --plot " + s
                        if plot_case != '':
                            command += " --plot_case " + plot_case
                        command_list = command.split()
                        subprocess.run(command)
                    idx += 1
