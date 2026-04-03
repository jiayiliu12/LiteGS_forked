from argparse import ArgumentParser
import sys

import litegs
import litegs.config
import wandb

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp_cdo,op_cdo,pp_cdo,dp_cdo=litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo,parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo,parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo,parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo,parser)
    
    parser.add_argument("--test_epochs", nargs="+", type=int, default=list(range(10,210,10)))
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    lp=litegs.arguments.ModelParams.extract(args)
    op=litegs.arguments.OptimizationParams.extract(args)
    pp=litegs.arguments.PipelineParams.extract(args)
    dp=litegs.arguments.DensifyParams.extract(args)
    
    # litegs.training.start(lp,op,pp,dp,args.test_epochs,args.save_epochs,args.checkpoint_epochs,args.start_checkpoint)

    # 2: Define the search space
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "sweep/objective_Testset"},
        "parameters": {
            "mass_threshold": {"max": 0.95, "min": 0.4},
            "lambda_s": {"max": 1.0, "min": 0.0},
        },
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="LiteGS-merged-Speedy")

    def wandb_start():
        run = wandb.init(project="LiteGS-merged-Speedy", dir="/capstor/scratch/cscs/ljiayi/LiteGS_wandb")
        dp.mass_threshold = run.config.mass_threshold
        dp.lambda_s = run.config.lambda_s

        litegs.training.start(lp,op,pp,dp,args.test_epochs,args.save_epochs,args.checkpoint_epochs,args.start_checkpoint)

    wandb.agent(sweep_id, function=wandb_start, count=20)