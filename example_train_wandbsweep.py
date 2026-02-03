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
        "metric": {"goal": "maximize", "name": "test/psnr_Testset"},
        "parameters": {
            "soft_prune_ratio": {"max": 0.4, "min":0.0},
            "hard_prune_ratio": {"max": 0.5, "min":0.1},
            "soft_prune_epoch_interval": {"max":20, "min":5},
            "hard_prune_epoch_interval": {"max":10, "min":5},
            # "": {"max":, "min":},
            # "": {"max":, "min":},
        },
    }

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="LiteGS")

    def wandb_start():
        run = wandb.init(project="LiteGS")
        dp.soft_prune_ratio = run.config.soft_prune_ratio
        dp.hard_prune_ratio = run.config.hard_prune_ratio
        dp.soft_prune_ratio = run.config.soft_prune_ratio
        dp.soft_prune_ratio = run.config.soft_prune_ratio

        litegs.training.start(lp,op,pp,dp,args.test_epochs,args.save_epochs,args.checkpoint_epochs,args.start_checkpoint)

    wandb.agent(sweep_id, function=wandb_start, count=25)