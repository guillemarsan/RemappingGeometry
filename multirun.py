import numpy as np
import subprocess
import pathlib, json

if __name__ == "__main__":

    script = 'run_pathint.py' 
    # cases = ['ED', 'EDr1', 'gridcells', 'EDd1', 'gridcellsd1', 'MSlow', 'MSmed', 'MShigh', \
    #          'sensorymed', 'sensoryhigh', 'MSlowd1', 'MSmedd1', 'MShighd1', 'sensorymedd1', 'sensoryhighd1', \
    #          'nullrotation', 'nullrotationd1']
    dir_loc = './data/v1/'
    # cases = ['EDd1', 'gridcellsd1', 'MSlowd1', 'MSmedd1', 'MShighd1', 'sensorymedd1', 'sensoryhighd1', \
    #          'nullrotationd1']

    cases = ['gridcellsd18']
    compute = True
    analyse = True
    plot = True

    for case in cases:
        print("##### CASE " + case + " ######")
        experiment = 'remapping' if case not in {'nullrotation', 'nullrotationd1'} else 'recruitment'
        output = case
        path = pathlib.Path(dir_loc+ output)
        path.mkdir(parents=True, exist_ok=True)

        ## ENCODING - DECODING
        if case == 'ED':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [128], [-1], 'randclosed-load-polyae', 'M', 'rotation', 10, ''
        elif case == 'EDr1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 1, [128], [-1], 'identity', 'M', 'rotation', 10, ''
        elif case == 'gridcells':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 8, [12], [-1], 'randclosed-load-polyae', 'C', 'gridcells', 10, ''
        elif case == 'EDd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [3], [-1], 'randclosed-load-polyae', 'SM', 'rotation', 3, '--save_input'
        elif case == 'gridcellsd18':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'C', 'gridcells', 3, '--save_input'

        ## MIXED-SELECTIVE
        if case == 'MSlow':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'CM', 'parallel', 10, '--input_sepnorm'
        elif case == 'MSmed':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [64], [-1], 'randclosed-load-polyae', 'CM', 'flexibleGP', 10, '--input_sepnorm'
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
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [4], [-1], 'randclosed-load-polyae', 'CM', 'parallel', 3, '--input_sepnorm --save_input'
        elif case == 'MSmedd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [4], [-1], 'randclosed-load-polyae', 'CM', 'flexibleGP', 3, '--input_sepnorm --save_input'
        elif case == 'MShighd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [4], [4], 'randclosed-load-polyae', 'CM', 'flexible', 3, '--input_sepnorm --save_input'
        elif case == 'MSrd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'CM', 'flexibler', 3, '--input_sepnorm --save_input'
        elif case == 'MSrd1M':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 8, [4], [-1], 'randclosed-load-polyae', 'M', 'flexibler', 3, '--input_sepnorm --save_input'
        elif case == 'sensorymedd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [4], [-1], 'randclosed-load-polyae', 'M', 'sensoryGP', 3, '--save_input'
        elif case == 'sensoryhighd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [4], [4], 'randclosed-load-polyae', 'M', 'sensory', 3, '--save_input'

        ## NULLSPACE REMAPPING
        elif case == 'nullrotation':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            2, 16, [128], [-1], 'randclosed-load-polyae', 'M', 'rotation', None, ''
            tagging_sparse = [1]
            tagging_currents = [-10]
        elif case == 'nullrotationd1':
            dim_pcs, red, dims, variances, model, conj, encoding, num_envs, extra = \
            1, 256, [3], [-1], 'randclosed-load-polyae', 'M', 'rotation', None, '--save_input'
            tagging_sparse = [1]
            tagging_currents = [-10]
            
        case_scripts = "--dim_pcs " + str(dim_pcs) + " --model " + model + " --model_conj " + \
            conj + " --encoding " + encoding + " " + extra

        simulate = 'minimization'
        script_args = ("--path_type grid --simulate " + str(simulate) + " " + case_scripts + \
                    " --load_id 0 --save --decoder_amp 1 --thresh_amp 1")
        script_args += " --dir ./data/v1/" + output + "/"

        if experiment == "remapping":

            if compute:
                print("## RUNNING SIMULATIONS ##")
                # Run trials in different environments with different embedding variance and dimensionality
                for dim in dims:
                    print("Dimensions:" + str(dim))
                    n = red*dim
                    script_args += " --nb_neurons " + str(n) 
                    for sigma in variances:
                        print("Variance:" + str(sigma))
                        for env in np.arange(num_envs)+50:
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
                array = ['measures', 'remap_vec', 'placefields'] if dim_pcs == 2 else ['placefields', 'vis']
                for s in array:
                    print("Plot " + s)
                    command = plot_script + " --plot " + s
                    command_list = command.split()
                    subprocess.run(command)

        elif experiment == "recruitment":
            
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
                    active_idx = np.array(f['activeidx'])
                else: 
                    maxfr = np.array(f['maxfr'])
                    active_idx = np.argwhere(maxfr > 1e-3)[:,0]

                num_active = active_idx.shape[0]
                
                for S in tagging_sparse:
                    # Tagging
                    print("Sparseness = " + str(S))
                    seed = 0 
                    np.random.seed(seed)
                    opsin = np.random.binomial(n=1, p=S, size=num_active)
                    tagged_idx = active_idx[np.where(opsin)].tolist()

                    tagged_idx_str = ' '.join(map(str, tagged_idx))
                    tagged_args = "--tagging_sparse " + str(S)
                    tagged_args += (" --tagged_idx " + tagged_idx_str) if len(tagged_idx) > 0 else ''
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
                array = ['measures', 'placefields'] if dim_pcs == 2 else ['pca', 'remap_vec']
                for s in array:
                    command = plot_script + " --plot " + s
                    command_list = command.split()
                    subprocess.run(command)
