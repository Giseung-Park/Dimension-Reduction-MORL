import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.evaluation import seed_everything

### sumo_rl add
from sumo_rl.environment.env import SumoEnvironment
import argparse
import pdb
import wandb

def main():
    # import torch, random
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    #
    # if torch.cuda.is_available() and (args.device != 'cpu'):
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    seed_everything(args.seed)

    def make_env():
        env = mo_gym.make("mo-lunar-lander-v2")
        env = MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=args.main_learning_rate, # 3e-4,
        gamma=0.99,
        batch_size=32,
        net_arch=[128, 128], # activation is nn.ReLU
        buffer_size=args.buffer_size,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=args.epsilon_decay_steps,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=args.homotopy_decay_steps,
        learning_starts=200,
        envelope=args.envelope,
        gradient_updates=1,
        target_net_update_freq=args.target_update_interval,  # 1000,  # 500 reduce by gradient updates
        tau=args.tau, # 1
        log=True, # True
        project_name="Lunar_High_Dim",
        experiment_name="Envelope_L",
        seed=args.seed,
        reduced_reward_dim=args.reduced_reward_dim,
        # update_freq=args.update_freq,
        device=args.device, # auto
        r_learning_rate = args.r_learning_rate,
        r_learn_interval = args.r_learn_interval,
        evaluation_form = args.evaluation_form,
        reduce_baselines = args.baselines,
        relax_positive = args.relax_positive,
        n_coef = args.npca_coef,
        drp = args.dropout_rate,
        dehidnum = args.decoder_hidden_variables,
    )

    assert args.num_eval_weights_for_front > 1

    agent.train(
        total_timesteps=args.total_timesteps,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        # ref_point=[-150*np.ones(5), -100*np.ones(5), -50*np.ones(5)], ## For Lunar
        ref_point= [np.array([0, -150, -150, -150, -150]), np.array([0, -100, -100, -100, -100]), np.array([-100, -100, -100, -100, -100]),],
        # known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
        num_eval_weights_for_front=args.num_eval_weights_for_front,
        num_eval_episodes_for_front=args.num_eval_episodes_for_front,
        eval_freq=args.eval_freq_factor,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )


if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Envelope-Q-Learning for LunarLander""")
    ## 0. Device
    prs.add_argument("-dev", dest="device", type=str, choices=['auto', 'cuda', 'cpu'], default='auto', help="Option for device\n")

    ## 1. Environmental Variables
    prs.add_argument("-rd", dest="reduced_reward_dim", type=int, default=5, help="reduced_reward_dim\n")  # 4 for reduced

    ## Envelope or not
    prs.add_argument("-env", dest="envelope", action="store_true", help="Run envelope algo or not.\n") ##

    prs.add_argument("-tt", dest="total_timesteps", type=int, default=2000000, ##
                     help="Total Timesteps. We have total episodes of total_timesteps/(num_seconds/delta_time).\n")

    prs.add_argument("-bf", dest="buffer_size", type=int, default=1000000, help="Buffer size\n") ##
    prs.add_argument("-se", dest="seed", type=int, default=0, help="Random seed\n") ##

    ## 2. Metric-related variables
    prs.add_argument("-nevep", dest="num_eval_episodes_for_front", type=int, default=5, help="num_eval_episodes_for_front\n") ## Traffic gives deterministic result.

    prs.add_argument("-nevw", dest="num_eval_weights_for_front", type=int, default=50, help="num_eval_weights_for_front\n") ## N samples
    prs.add_argument("-efq", dest="eval_freq_factor", type=int, default=200000, help="Eval period in terms of # of episodes\n") ##

    ## 3. Learning hyperparameters
    ## Hard update: default
    prs.add_argument("-tgit", dest="target_update_interval", type=int, default=500, help="Target_update_interval\n")  ##
    prs.add_argument("-tau", dest="tau", type=float, default=1, help="Target update ratio\n")  ##

    ## Main Q learning rate
    prs.add_argument("-mlr", dest="main_learning_rate", type=float, default=0.001, help="Learning Rate of Main Q function\n") ##
    ## decay
    prs.add_argument("-epdec", dest="epsilon_decay_steps", type=int, default=200000, help="epsilon_decay_steps\n") ##
    prs.add_argument("-hodec", dest="homotopy_decay_steps", type=int, default=50000, help="homotopy_decay_steps\n") ##

    ## eval parameters
    prs.add_argument("-evf", dest="evaluation_form", type=str, choices=['random', 'equal'], default='random', help="Option for evaluation form\n")  ## We require this value no less than reduced_reward_dim

    ## 4. Ours hyperparameters
    prs.add_argument("-base", dest="baselines", type=str, choices=['ours', 'ae', 'pca', 'ours_nodec', 'npca', 'random'], default='ours', help="Option for evaluation form\n")  ## We require this value no less than reduced_reward_dim

    prs.add_argument("-dehid", dest="decoder_hidden_variables", type=int, default=32, help="decoder_hidden_variables\n")  ##

    prs.add_argument("-rlr", dest="r_learning_rate", type=float, default=0.001, help="r_learning_rate\n")  ##
    prs.add_argument("-rint", dest="r_learn_interval", type=int, default=5, help="r_learn_interval\n") ##

    prs.add_argument("-drp", dest="dropout_rate", type=float, default=0, help="dropout_rate\n")  ## Apply in decoder

    prs.add_argument("-rp", dest="relax_positive", action="store_true", help="Run our algo or not.\n") ## Default False

    ## 5. NPCA parameter
    prs.add_argument("-ncoef", dest="npca_coef", type=float, default=50000, help="npca_coef\n")

    args = prs.parse_args()

    wandb.init(project="Reward_Dim_Reduction_Lunar_Rev3", group= 'Envel_' + str(args.envelope)
                                                + '_rd=' + str(args.reduced_reward_dim)
                                                + '_base=' + str(args.baselines)
                                                 + '_tt=' + str(args.total_timesteps)
                                               + '_nevw=' + str(args.num_eval_weights_for_front)
                                               + '_efq=' + str(args.eval_freq_factor)
                                               + '_mlr=' + str(args.main_learning_rate)
                                                   # + '_hodec=' + str(args.homotopy_decay_steps)
                                                + '_evf=' + str(args.evaluation_form) + '_rlr=' + str(args.r_learning_rate) + '_rint=' + str(args.r_learn_interval)
                                                + '_drp=' + str(args.dropout_rate),
                                                # + '_ncoef_new=' + str(args.npca_coef),
                                                # + '_rp=' + str(args.relax_positive) + '_affineenc', # + '_sum1' + '_lindec' + '_sum1_lindec_bias'
                                                job_type="train")

    wandb.run.name = "seed=" + str(args.seed)

    main()

    wandb.finish()
