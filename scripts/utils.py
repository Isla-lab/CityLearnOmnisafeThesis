# CityLearn utils
from citylearn.data import DataSet
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.citylearn import CityLearnEnv

# Logging
import wandb
from stable_baselines3.common.callbacks import BaseCallback

# Utils
import argparse
from collections import deque
from typing import Any, List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Configurations for Constrained RL on CityLearn")
        self._add_args()
        self.args = self.parser.parse_args()

    def _add_args(self):
        # Seed
        self.parser.add_argument('--seed', type=int, default=1, help="Experiment seed")

        # CityLearn config
        self.parser.add_argument('--data', type=str, default='citylearn_challenge_2023_phase_1', help="CityLearn dataset")
        self.parser.add_argument('--custom', action='store_true', help="Flag for CityLearn dataset customization")

        # RL args
        self.parser.add_argument('--wrapper', type=str, choices=['omnisafe', 'sb3'], default='omnisafe', help="CityLearn wrapper to use")
        self.parser.add_argument('--algo', type=str, default='PPO', help="RL algorithm to use")
        self.parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes to rollout")

        # Logging
        self.parser.add_argument('--wandb', action='store_true', help="Flag for logging on wandb")
        self.parser.add_argument('--entity', type=str, default='luca0', help="Wandb entity")
        self.parser.add_argument('--project', type=str, default='citylearn_omnisafe', help="Wandb project name")
        self.parser.add_argument('--tag', type=str, default='comfort_reward', help="Wandb tag")


class CityLearnKPIWrapper(StableBaselines3Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            # Get KPIs
            kpis = self.env.evaluate()

            # Filter district level KPIs
            kpis = kpis[kpis['level'] == 'district']
            discomfort = kpis[(kpis['cost_function'] == 'discomfort_cold_delta_average')]['value'].item()
            carbon_emissions = kpis[(kpis['cost_function'] == 'carbon_emissions_total')]['value'].item()
            net_consumption = kpis[(kpis['cost_function'] == 'electricity_consumption_total')]['value'].item()

            # Populate info dict
            info['discomfort'] = discomfort
            info['carbon_emissions'] = carbon_emissions
            info['net_consumption'] = net_consumption

        return obs, reward, terminated, truncated, info


class CityLearnWandbCallback(BaseCallback):
    def __init__(self, verbose: int=0, window_len: int=100):
        super().__init__(verbose)
        
        # Episodic info
        self.ep_count = 0
        self.ep_rewards = deque(maxlen=window_len)
        self.ep_lengths = deque(maxlen=window_len)

        # KPIs info
        self.discomfort_h = deque(maxlen=window_len)
        self.carbon_emissions_h = deque(maxlen=window_len)
        self.net_consumption_h = deque(maxlen=window_len)

        # Log fn
        self._log_fn = lambda x: sum(x)/len(x)

    def _on_step(self) -> bool:
        
        info = self.locals['infos'][0]
        if self.locals['dones'][0]:
            # Update episode count
            self.ep_count += 1

            # Episodic info
            ep_info = info['episode']
            self.ep_rewards.append(ep_info['r'])
            self.ep_lengths.append(ep_info['l'])

            # KPIs
            self.discomfort_h.append(info['discomfort'])
            self.carbon_emissions_h.append(info['carbon_emissions'])
            self.net_consumption_h.append(info['net_consumption'])
                
            # Log episodic return and length to wandb
            wandb.log(
                {
                    'TotalEnvSteps': self.num_timesteps,
                    'Metrics/Discomfort': self._log_fn(self.discomfort_h),
                    'Metrics/CO2_Emissions': self._log_fn(self.carbon_emissions_h),
                    'Metrics/Electricity_Consumption': self._log_fn(self.net_consumption_h),
                    'Metrics/EpRet': self._log_fn(self.ep_rewards),
                    'Metrics/EpLen': self._log_fn(self.ep_lengths),
                    'Metrics/EpCost': 0.0
                },
                step=self.ep_count
            )

            print(
                f"{'*'*30}\nEPISODE {self.ep_count}"                          +
                f'\n- Discomfort:              {self.discomfort_h[-1]}'       +
                f'\n- CO2 Emissions:           {self.carbon_emissions_h[-1]}' +
                f'\n- Electricity Consumption: {self.net_consumption_h[-1]}'  +
                f'\n- Reward:                  {self.ep_rewards[-1]}'         +
                f'\n- Length:                  {self.ep_lengths[-1]}'
            )
        
        return True
    

def _select_items(schema: Dict[str, Any], key: str, available_items: List[str]=None) -> Tuple[dict, List[str]]:
    assert key in ['buildings', 'observations', 'actions'], f'Unknown schema key {key}.'
    
    # Init
    flag_key = 'include' if key == 'buildings' else 'active'
    pool = available_items if available_items is not None else list(schema[key].keys())

    print(f"Available {key}:")
    for idx, item in enumerate(pool):
        print(f"- {idx+1}. {item}")

    # Item selection
    user_input = input(f"\nSelect {item} by entering their numbers separated by commas (e.g., 1,3,5): ")
    selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip().isdigit() and 0 < int(i.strip()) <= len(pool)]
    selected_items = [pool[i] for i in selected_indices]

    print(f"Selected items: {selected_items}\n\n")

    # Filter CityLearn items based on user selection
    for item in schema[key].keys():
        schema[key][item][flag_key] = (item in selected_items) 

    return schema, selected_items
    

def select_env_config(dataset: str) -> Dict[str, Any]:

    print("="*40)
    print("CityLearnOmnisafe")
    print("="*40)
    print(f"Dataset: {dataset}")

    # Filter schema's available observations/actions 
    schema = DataSet().get_schema(dataset)
    available_obs = [obs for obs in schema['observations'].keys() if schema['observations'][obs]['active']]
    available_act = [act for act in schema['actions'].keys() if schema['actions'][act]['active']]

    # User's building selection
    schema, _ = _select_items(schema=schema, key='buildings')
    # User's observation selection
    schema, selected_obs = _select_items(schema=schema, key='observations', available_items=available_obs)
    # User's action selection
    schema, selected_act = _select_items(schema=schema, key='actions', available_items=available_act)

    for action in selected_act:
        device = action.split('_')[0]
        
        if not any(device in obs for obs in selected_obs):
            print(f"[WARN] You selected to control '{action}', but no observation related to '{device}' is selected. Please select again.")

    return schema


def default_env_config(dataset: int) -> Dict[str, Any]:
    # Get schema from CityLearn dataset
    schema = DataSet().get_schema(dataset)

    # Our default configuration considers only `Building_1`
    for build in schema['buildings'].keys():
        schema['buildings'][build]['include'] = (build == 'Building_1')

    return schema

def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment instance.

    Returns
    -------
    kpis: pd.DataFrame
        KPI table.
    """

    kpis = env.unwrapped.evaluate()

    # names of KPIs to retrieve from evaluate function
    kpi_names = {
        'cost_total': 'Cost',
        'carbon_emissions_total': 'Emissions',
        'daily_peak_average': 'Avg. daily peak',
        'ramping_average': 'Ramping',
        'monthly_one_minus_load_factor_average': '1 - load factor',
        'discomfort_proportion': 'Discomfort'
    }
    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()
    kpis['cost_function'] = kpis['cost_function'].map(lambda x: kpi_names[x])

    # round up the values to 2 decimal places for readability
    kpis['value'] = kpis['value'].round(2)

    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})

    return kpis

def plot_district_kpis(envs: dict[str, CityLearnEnv], base_path: str) -> None:
    """Plots electricity consumption, cost, carbon emissions,
    average daily peak, ramping and (1 - load factor) at the
    district-level for different control agents in a bar chart.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    base_path: str
            Path to save the figure.
    Returns
    -------
    None
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='district'].copy()
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names= kpis['kpi'].unique()
    column_count_limit = 3
    row_count = math.ceil(len(kpi_names)/column_count_limit)
    column_count = min(column_count_limit, len(kpi_names))

    # Calculate figure size - adjusted for better proportions
    figsize = (6*column_count, 3.5*row_count)
    
    fig, _ = plt.subplots(
        row_count, column_count, figsize=figsize
    )

    # Add figure title
    fig.suptitle('District KPIs', fontsize=16, fontweight='bold')

    for i, (ax, (k, k_data)) in enumerate(zip(fig.axes, kpis.groupby('kpi'))):
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)

        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_yticklabels([])  # Remove y-axis tick labels
        ax.set_title(k)

        for j in range(len(ax.containers)):
            ax.bar_label(ax.containers[j], fmt='%.2f')

        # Add dashed vertical line at x=1
        ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, label='No Control')

        if i == len(kpi_names) - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

        for s in ['right','top']:
            ax.spines[s].set_visible(False)

    plt.tight_layout()
    os.makedirs(base_path, exist_ok=True)
    plt.savefig(f'{base_path}/district_kpis.png', bbox_inches='tight')
    plt.close()