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

from scipy.stats import zscore
from dune_client.client import DuneClient
import datetime
from pytz import utc



def get_uniswap_top_pools(arguments):
    n_pools = int(arguments['N'])
    gpt_memory = arguments['gpt_memory']
    url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

    # GraphQL query to fetch the pool id and daily volume for the top n_pools
    query = """
    {{
      pools(first: {0}, orderBy: volumeUSD, orderDirection: desc) {{
        id
        volumeUSD
      }}
    }}
    """.format(n_pools)

    # Make the request
    response = requests.post(url, json={'query': query})

    # Get the JSON data from the response
    data = json.loads(response.text)

    # Get the pool IDs from the data
    pool_ids = [pool['id'] for pool in data['data']['pools']]
    messsage = f"""Here the ids that i got according your queries {str(pool_ids)}"""
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
    df = pd.DataFrame(data['data']['poolDayDatas'])

    return data


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

    plt.tight_layout()
    unique_id = uuid.uuid4()
    image_path = f"{unique_id}.png"
    plt.savefig(image_path)
    # plt.show()

    return image_path, updated_results.to_string()