import pandas as pd
import json
import requests
import requests
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import uuid
import os

from scipy.stats import zscore
from dune_client.client import DuneClient
import datetime
from pytz import utc
base_path = '/mnt/d/Code/tokenGPT/llm_data_dir'



def get_uniswap_top_pools(arguments):
    n_pools = int(arguments['N'])
    gpt_memory = arguments['gpt_memory']
    url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

    # GraphQL query to fetch the pool id, token0, token1 name and symbols, fee tier, tvlUSD and daily volume for the top n_pools
    query = """
    {{
      pools(first: {0}, orderBy: volumeUSD, orderDirection: desc) {{
        
        feeTier
        id
        liquidity
        feesUSD
        totalValueLockedUSD
        txCount
        volumeUSD
        token1Price
        totalValueLockedETH
        totalValueLockedToken0
        totalValueLockedToken1
        volumeToken0
        volumeToken1
        volumeUSD
        
      }}
    }}
    """.format(n_pools)

    # Make the request
    response = requests.post(url, json={'query': query})

    # Get the JSON data from the response
    data = json.loads(response.text)
    pools = data['data']['pools']

    # Convert to DataFrame
    df = pd.json_normalize(pools)
   
    os.makedirs(base_path, exist_ok=True)
    # Create a unique timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate the unique file path
    file_path = os.path.join(base_path, f"top_{n_pools}_uniswap_pools_data_at_{timestamp}.csv")
    
    df.to_csv(file_path, index=False)

    # Get the pool IDs from the data
    pool_ids = [pool['id'] for pool in data['data']['pools']]
    messsage = f"""Here is the data of top uniswap V3 pools {str(data)}"""
    print(messsage)
    gpt_memory(None, messsage)
    return messsage


def get_uniswap_pool_day_data(arguments):
    pool_id = arguments['pool_id']
    number_of_days = int(arguments['number_of_days'])
    gpt_memory = arguments['gpt_memory']
    url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
    query = f"""
    {{
        poolDayDatas(where: {{ pool: "{pool_id}" }}, first: {number_of_days}, orderBy: date, orderDirection: desc) {{
            date
            tick
            sqrtPrice
            liquidity
            volumeUSD
            volumeToken0
            volumeToken1
            tvlUSD
            feesUSD
            close
            open
            low
            high
        }}
    }}
    """
    response = requests.post(url, json={'query': query})
    data = response.json()
    data_df = pd.DataFrame(data['data']['poolDayDatas'])

    data_df['volumeUSD'] = data_df['volumeUSD'].astype(float)
    data_df['volumeToken0'] = data_df['volumeToken0'].astype(float)
    data_df['volumeToken1'] = data_df['volumeToken1'].astype(float)
    data_df['sqrtPrice'] = (data_df['sqrtPrice'].astype(float))
    data_df['liquidity'] = data_df['liquidity'].astype(float)
    data_df['tvlUSD'] = data_df['tvlUSD'].astype(float)
    data_df['feesUSD'] = data_df['feesUSD'].astype(float)
    data_df['close'] = data_df['close'].astype(float)
    data_df['open'] = data_df['open'].astype(float)
    data_df['low'] = data_df['low'].astype(float)
    data_df['high'] = data_df['high'].astype(float)
    data_df['date'] = pd.to_datetime(data_df['date'], unit='s')
  
    os.makedirs(base_path, exist_ok=True)
    # Create a unique timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate the unique file path
    file_path = os.path.join(base_path, f"daily_{number_of_days}_days_{pool_id}_pool__data_at{timestamp}.csv")
    data_df.to_csv(file_path, index=False)


    image_paths = []

    def save_plot_and_return_path(plot_func, filename):
        img_path = os.path.join(base_path, filename)
        plot_func()
        plt.savefig(img_path)
        plt.close()
        image_paths.append(img_path)
        return img_path

    # Plot and save images
    save_plot_and_return_path(lambda: data_df.plot(x='date', y='sqrtPrice'), 'sqrtPrice_plot.png')
    save_plot_and_return_path(lambda: data_df.plot(x='date', y='feesUSD'), 'feesUSD_plot.png')
    save_plot_and_return_path(lambda: data_df.plot(x='date', y='liquidity'), 'liquidity_plot.png')

    ax = data_df.plot(x='date', y='liquidity', color='blue', label='liquidity')
    data_df.plot(x='date', y='sqrtPrice', color='red', secondary_y=True, ax=ax, label='sqrtPrice')
    ax.set_ylabel('liquidity')
    ax.right_ax.set_ylabel('sqrtPrice')
    plt.title('liquidity and sqrtPrice Over Time')
    save_plot_and_return_path(lambda: None, 'liquidity_sqrtPrice_plot.png')

    ax = data_df.plot(x='date', y='liquidity', color='blue', label='liquidity')
    data_df.plot(x='date', y='feesUSD', color='red', secondary_y=True, ax=ax, label='Fees')
    ax.set_ylabel('liquidity')
    ax.right_ax.set_ylabel('Fees')
    plt.title('liquidity and Fees Over Time')
    save_plot_and_return_path(lambda: None, 'liquidity_feesUSD_plot.png')

    ax = data_df.plot(x='date', y='volumeToken0', color='blue', label='volumeToken0')
    data_df.plot(x='date', y='volumeToken1', color='red', secondary_y=True, ax=ax, label='volumeToken1')
    ax.set_ylabel('volumeToken0')
    ax.right_ax.set_ylabel('volumeToken1')
    plt.title('Volume Token0 and Token1 Over Time')
    save_plot_and_return_path(lambda: None, 'volumeToken0_volumeToken1_plot.png')

    # EDA plots
    save_plot_and_return_path(lambda: plt.hist(data_df['volumeUSD']), 'volumeUSD_histogram.png')
    save_plot_and_return_path(lambda: sns.boxplot(x=data_df['volumeUSD']), 'volumeUSD_boxplot.png')
    save_plot_and_return_path(lambda: plt.scatter(x=data_df['date'], y=data_df['volumeUSD']), 'volumeUSD_scatter.png')
    save_plot_and_return_path(lambda: sns.countplot(data_df['volumeUSD']), 'volumeUSD_countplot.png')
    save_plot_and_return_path(lambda: sns.heatmap(data_df.corr()), 'heatmap.png')

    corr_matrix = data_df.corr()
    save_plot_and_return_path(lambda: sns.heatmap(corr_matrix, annot=True, cmap='coolwarm'), 'corr_heatmap.png')

    messsage = f"""DataFrame saved at {file_path} in csv format, Here is {number_of_days} days daily data of {pool_id} pool {str(data)}"""
    print(messsage)
    gpt_memory(None, messsage)
    return messsage,image_paths



def price_impact_analysis(arguments):
    weeks = 52
    initial_weekly_transactions = int(arguments['initial_weekly_transactions'])
    initial_circulating_supply = int(arguments['initial_circulating_supply'])
    initial_price = float(arguments['initial_price'])
    growth_rate = 0.05
    growth_rate_decay = 0.01
    buy_transactions = 0.1
    sell_transactions = 0.05
    D0 = 10000
    kc = 0.1
    kl = 0.05
    Delta_Dc = 500
    Delta_Dl = 300
    epsilon_d = 0.8
    epsilon_s = 0.5
    # Initialize data arrays
    weekly_transactions = np.zeros(weeks)
    tokens_locked = np.zeros(weeks)
    tokens_released = np.zeros(weeks)
    net_supply_change = np.zeros(weeks)
    circulating_supply = np.zeros(weeks)
    token_price = np.zeros(weeks)
    net_demand_change = np.zeros(weeks)
    price_change = np.zeros(weeks)

    # Initial conditions
    weekly_transactions[0] = initial_weekly_transactions
    circulating_supply[0] = initial_circulating_supply
    token_price[0] = initial_price
    net_demand_change[0] = D0 + kc * Delta_Dc + kl * Delta_Dl
    price_change[0] = 0

    # Model each week
    for week in range(1, weeks):
        weekly_transactions[week] = weekly_transactions[week - 1] * (1 + growth_rate)
        tokens_locked[week] = weekly_transactions[week] * buy_transactions
        tokens_released[week] = weekly_transactions[week] * sell_transactions
        net_supply_change[week] = tokens_released[week] - tokens_locked[week]
        circulating_supply[week] = circulating_supply[week - 1] + net_supply_change[week]

        # Recalculate demand increase for the week
        Delta_Dc *= (1 + growth_rate)
        Delta_Dl *= (1 + growth_rate)
        net_demand_change[week] = D0 + kc * Delta_Dc + kl * Delta_Dl

        # Price adjustment based on the change in demand and circulating supply

        delta_D = net_demand_change[week] - net_demand_change[week - 1]
        delta_S = net_supply_change[week]

        # Price adjustment using price elasticities
        price_change_due_to_demand = (delta_D / (net_demand_change[week - 1] * epsilon_d))
        price_change_due_to_supply = (delta_S / ((circulating_supply[week - 1] + delta_S) * epsilon_s))

        # Net price change and new price calculation
        price_change[week] = price_change_due_to_demand + price_change_due_to_supply
        token_price[week] = token_price[week - 1] * (1 + price_change[week])

        # Growth rate decay on weekly basis
        growth_rate = growth_rate * (1 - week * growth_rate_decay)

    # Update results DataFrame with new calculations
    updated_results = pd.DataFrame({
        "Week": np.arange(weeks),
        "Transactions": weekly_transactions,
        "Tokens Locked": tokens_locked,
        "Tokens Released": tokens_released,
        "Weekly Supply Change": net_supply_change,
        "Weekly Demand Change": net_demand_change,
        "Circulating Supply": circulating_supply,
        "Percentage change Price": price_change,
        "Token Price": token_price,
    })

    os.makedirs(base_path, exist_ok=True)
    # Create a unique timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate the unique file path
    file_path = os.path.join(base_path, f"price_impact_analysis_at_{timestamp}.csv")
    updated_results.to_csv(file_path, index=False)

    # Plotting the results with the revised price adjustment
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    # Plot for Circulating Supply
    axs[0].plot(updated_results['Week'], updated_results['Circulating Supply'], label='Circulating Supply',
                color='blue')
    axs[0].set_title('Effective Circulating Supply Over Time')
    axs[0].set_xlabel('Week')
    axs[0].set_ylabel('Effective Circulating Supply')
    axs[0].grid(True)

    # Plot for Token Price
    axs[1].plot(updated_results['Week'], updated_results['Token Price'], label='Token Price', color='red')
    axs[1].set_title('Token Price Over Time')
    axs[1].set_xlabel('Week')
    axs[1].set_ylabel('Token Price (USD)')
    axs[1].grid(True)

    # Plot for Demand Increase
    axs[2].plot(updated_results['Week'], updated_results['Weekly Demand Change'], label='Weekly Demand Change',
                color='green')
    axs[2].set_title('Weekly Demand Change Over Time')
    axs[2].set_xlabel('Week')
    axs[2].set_ylabel('Demand Change')
    axs[2].grid(True)

    # Plot for Tokens Locked
    axs[3].plot(updated_results['Week'], updated_results['Tokens Locked'], label='Tokens Locked', color='green')
    axs[3].set_title('Token Locked in Sinks Over Time')
    axs[3].set_xlabel('Week')
    axs[3].set_ylabel('Tokens Locked')
    axs[3].grid(True)

    def save_plot_and_return_path(plot_func, filename):
        img_path = os.path.join(base_path, filename)
        plot_func()
        plt.savefig(img_path)
        plt.close()
        image_paths.append(img_path)
        return img_path

    plt.tight_layout()
    unique_id = uuid.uuid4()
    image_path = f"{base_path}/{unique_id}.png"
    plt.savefig(image_path)
    # plt.show()

    return  updated_results.to_string(), [image_path]

def fetch_dune_client_query_data(arguments):
    query_id = int(arguments['query_id'])
    gpt_memory = arguments['gpt_memory']
    dune_api_key = "gqBRKclPQMlU9Jm009ikKIrYV4gWhtuq"
    dune = DuneClient(dune_api_key)
    #curl -H "X-Dune-API-Key:" gqBRKclPQMlU9Jm009ikKIrYV4gWhtuq"https://api.dune.com/api/v1/query/3467177/results?limit=1000"
    # Fetch the latest query result
    query_result = dune.get_latest_result(query_id)
    messsage = f"""Here are the results for query_id: {query_id}: {str(query_result['result'])}"""
    print(messsage)
    gpt_memory(None, messsage)
    return messsage
    


import requests
import json

def initialize_script(base_path, reset_env_var=True):
    url = 'http://127.0.0.1:8000/initialize_script/'
    headers = {'Content-Type': 'application/json'}
    data = {
        "base_path": base_path,
        "reset_env_var": reset_env_var
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def train_ddpg(max_steps=2, n_episodes=2, model_name="model_storage/ddpg/ddpg_fazool",
               alpha=0.001, beta=0.001, tau=0.8, batch_size=50, training=True,
               agent_budget_usd=10000, use_running_statistics=False):
    url = 'http://127.0.0.1:8000/train_ddpg/'
    headers = {'Content-Type': 'application/json'}
    data = {
        "max_steps": max_steps,
        "n_episodes": n_episodes,
        "model_name": model_name,
        "alpha": alpha,
        "beta": beta,
        "tau": tau,
        "batch_size": batch_size,
        "training": training,
        "agent_budget_usd": agent_budget_usd,
        "use_running_statistics": use_running_statistics
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def evaluate_ddpg(eval_steps=2, eval_episodes=2, model_name="model_storage/ddpg/ddpg_fazool",
                  percentage_range=0.6, agent_budget_usd=10000, use_running_statistics=False):
    url = 'http://127.0.0.1:8000/evaluate_ddpg/'
    headers = {'Content-Type': 'application/json'}
    data = {
        "eval_steps": eval_steps,
        "eval_episodes": eval_episodes,
        "model_name": model_name,
        "percentage_range": percentage_range,
        "agent_budget_usd": agent_budget_usd,
        "use_running_statistics": use_running_statistics
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def train_ppo(max_steps=2, n_episodes=2, model_name="model_storage/ppo/ppo2_fazool22",
              buffer_size=5, n_epochs=20, gamma=0.5, alpha=0.001, gae_lambda=0.75,
              policy_clip=0.6, max_grad_norm=0.6, agent_budget_usd=10000, use_running_statistics=False,
              action_transform="linear"):
    url = 'http://127.0.0.1:8000/train_ppo/'
    headers = {'Content-Type': 'application/json'}
    data = {
        "max_steps": max_steps,
        "n_episodes": n_episodes,
        "model_name": model_name,
        "buffer_size": buffer_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "alpha": alpha,
        "gae_lambda": gae_lambda,
        "policy_clip": policy_clip,
        "max_grad_norm": max_grad_norm,
        "agent_budget_usd": agent_budget_usd,
        "use_running_statistics": use_running_statistics,
        "action_transform": action_transform
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def evaluate_ppo(eval_steps=2, eval_episodes=2, model_name="model_storage/ppo/ppo2_fazool",
                 percentage_range=0.5, agent_budget_usd=10000, use_running_statistics=False,
                 action_transform="linear"):
    url = 'http://127.0.0.1:8000/evaluate_ppo/'
    headers = {'Content-Type': 'application/json'}
    data = {
        "eval_steps": eval_steps,
        "eval_episodes": eval_episodes,
        "model_name": model_name,
        "percentage_range": percentage_range,
        "agent_budget_usd": agent_budget_usd,
        "use_running_statistics": use_running_statistics,
        "action_transform": action_transform
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def inference(pool_state, user_preferences, pool_id="0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
              ddpg_agent_path="model_storage/ddpg/ddpg_1",
              ppo_agent_path="model_storage/ppo/lstm_actor_critic_batch_norm"):
    
    pool_state = {
    "current_profit": 500,
    "price_out_of_range": False,
    "time_since_last_adjustment": 40000,
    "pool_volatility": 0.2
    }
    user_preferences = {
    "risk_tolerance": {"profit_taking": 50, "stop_loss": -500},
    "investment_horizon": 7,
    "liquidity_preference": {"adjust_on_price_out_of_range": True},
    "risk_aversion_threshold": 0.1,
    "user_status": "new_user"
    }
    url = 'http://127.0.0.1:8000/inference/'
    headers = {'Content-Type': 'application/json'}
    data = {
        "pool_state": pool_state,
        "user_preferences": user_preferences,
        "pool_id": pool_id,
        "ddpg_agent_path": ddpg_agent_path,
        "ppo_agent_path": ppo_agent_path
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

