import argparse
import numpy as np
import subprocess
import pathlib, json

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--experiment", type=str, default="recruitment",
                        help="Type of experiment: remapping or recruitment")
    parser.add_argument("--script", type=str, default="run_pathint.py",
                        help="Script to run")
    parser.add_argument("--encoding", type=str, default="rotation",
                        help="Type of encoding: rotation, flexible or parallel")
    parser.add_argument("--num_variances", type=int, default=1,
                        help="In remapping, number of variances to test between 0 and 1")
    parser.add_argument("--num_environments", type=int, default=1,
                        help="In remapping, number of environments to test")
    parser.add_argument("--tagging_sparse", nargs='+', type=float, default=[0, 0.25, 0.5, 0.75, 0.9, 1],
                        help="In recruitment, probability of tagging")
    parser.add_argument("--tagging_currents", nargs='+', type=float, default=[0, -100, -500],
                        help="In recruitment, number of currents to test between 0 and -100")
    parser.add_argument("--output", type=str, default="./data/TestRecruitmentTotal/",
                        help="Output folder")
    args = parser.parse_args()

    encoding_arg = "--encoding " + args.encoding
    script_args = ("--dim_pcs 2 --nb_neurons 512 --dim_bbox 16" + 
                    " --model randclosed-load-polyae --save --decoder_amp 0.3 --thresh_amp 1.2")
    folder_arg = "--dir " + args.output

    if args.experiment == "remapping":

        # Run trials in different environments with different embedding variance
        for i in np.arange(args.num_variances):
            sigma = 1/args.num_variances * i
            print("Variance:" + str(sigma))
            for env in np.arange(args.num_environments):
                print("Environment:" + str(env))
                variance_args = "--env " + str(env) + " --embedding_sigma " + str(sigma)
                command = "python " + args.script + " " + encoding_arg + " " + \
                            script_args + " " + folder_arg + " " + variance_args
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