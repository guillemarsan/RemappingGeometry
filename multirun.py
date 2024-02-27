import argparse
import numpy as np
import subprocess
import pathlib, json

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of space")
    parser.add_argument("--nb_neurons", type=int, default=-16,
                        help="Number of neurons")
    parser.add_argument("--experiment", type=str, default="remapping",
                        help="Type of experiment: remapping or recruitment")
    parser.add_argument("--script", type=str, default="run_pathint.py",
                        help="Script to run")
    parser.add_argument("--encoding", type=str, default="flexible",
                        help="Type of encoding: rotation, flexible or parallel")
    parser.add_argument("--variances", nargs='+', type=float, default=[-1],
                        help="In remapping, variances to test between 0 and 1")
    parser.add_argument("--dimensions", nargs='+', type=float, default=[8, 16, 32, 64],
                        help="In remapping, dimensions to test between minimum and N")
    parser.add_argument("--num_environments", type=int, default=10,
                        help="In remapping, number of environments to test")
    parser.add_argument("--tagging_sparse", nargs='+', type=float, default=[0, 0.25, 0.5, 0.75, 0.9, 1],
                        help="In recruitment, probability of tagging")
    parser.add_argument("--tagging_currents", nargs='+', type=float, default=[0, -100, -500],
                        help="In recruitment, number of currents to test between 0 and -100")
    parser.add_argument("--output", type=str, default="./data/PosterFlexiblenn/",
                        help="Output folder")
    args = parser.parse_args()

    script_args = ("--path_type grid --simulate minimization" + 
                    " --model randclosed-load-polyae --input_sepnorm --model_sepnorm --load_id 0 --save --decoder_amp 1 --thresh_amp 1")
    script_args += " --dim_pcs " + str(args.dim_pcs) 
    script_args += " --encoding " + args.encoding
    script_args += " --dir " + args.output

    if args.experiment == "remapping":

        # Run trials in different environments with different embedding variance and dimensionality
        for dim in args.dimensions:
            print("Dimensions:" + str(dim))
            n = args.nb_neurons if args.nb_neurons > 0 else -dim*args.nb_neurons
            script_args += " --nb_neurons " + str(n) 
            for sigma in args.variances:
                print("Variance:" + str(sigma))
                for env in np.arange(args.num_environments):
                    print("Environment:" + str(env))
                    script_args += " --dim_bbox " + str(dim) + " --env " + str(env) + " --embedding_sigma " + str(sigma)
                    command = "python " + args.script + " " + script_args
                    command_list = command.split()
                    subprocess.run(command)

    elif args.experiment == "recruitment":
        
        sigma = 1
        env = 0
        variance_args = "--env " + str(env) + " --embedding_sigma " + str(sigma)

        # Run standard trial (no inhibition experiment)
        seed = 0
        print("Standard experiment")
        command = "python " + args.script + " " + encoding_arg + " " + \
                            script_args + " " + folder_arg + " " + variance_args
        command_list = command.split()
        subprocess.run(command)

        # Read tagged neurons from standard trial
        patt = "*.json"
        basepath = args.output
        path = pathlib.Path(basepath)
        standard_file = next(path.rglob(patt))
        with open(standard_file) as res_file:
            f = json.load(res_file)
        active_idx = np.array(f['activeidx'])
        num_active = active_idx.shape[0]
        
        for S in args.tagging_sparse:

            # Tagging
            print("Sparseness = " + str(S))
            np.random.seed(seed)
            opsin = np.random.binomial(n=1, p=S, size=num_active)
            tagged_idx = active_idx[np.where(opsin)].tolist()

            tagged_idx_str = ' '.join(map(str, tagged_idx))
            tagged_args = "--tagging_sparse " + str(S)
            tagged_args += (" --tagged_idx " + tagged_idx_str) if len(tagged_idx) > 0 else ''
            # Execute inhibition trials on tagged_idx neurons
            for C in args.tagging_currents:
                print("Current = " + str(C))
                inhibition_args = tagged_args + " --current_amp " + str(C)
                command = "python " + args.script + " " + encoding_arg + " " + \
                                script_args + " " + folder_arg + " " + variance_args + " "  + inhibition_args
                command_list = command.split()
                subprocess.run(command)