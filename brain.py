from openai import OpenAI
import json
from uniswap import get_uniswap_top_pools, get_uniswap_pool_day_data, price_impact_analysis,fetch_dune_client_query_data
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()


class TokenGPT:
    def __init__(self):
        self.initial_text = "My name is Token GPT and I am here to help you on tokennomics"
        self.gpt_memory = [{"role": "assistant", "content": self.initial_text}]
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    def memory_manage(self, human=None, tokengpt=None):
        if human is not None:
            self.gpt_memory.append({"role": "user", "content": human})

        if tokengpt is not None:
            self.gpt_memory.append({"role": "assistant", "content": tokengpt})

    def conversation(self, text):
        system_message = f"""I am your AI token engineering Assistant which will help you to navigate in complex landscape of 
        design, verification and optimization of decentralized token based economies.
        Select the best function that describes the user query and call it and provide assitance
        If something isn't clear and arguments is not in the data you can ask the proactive question from the user to fill it out"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_uniswap_top_pools",
                    "description": "The function aims to assist users in getting the data of top uniswap pools, user provides number of pools we wants to get data and this function will return the data like id, feetier, feesUSD, volumUSD, tvlUSD, symbol of token0 =, symbol of token1 etc. of each of top n pools, n is given by users based on how many top pool's data he wants to get "
                                   "ids"
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "N": {"type": "string",
                                  "description": "Its is number of pools user wwants to get data, i.e if user sets N=10, the function will return data of top 10 uniswap v3 pools"}, },
                        'required': ['N']
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_uniswap_pool_day_data",
                    "description": "Given the pool_id and number_of_days this function gets day data of that uniswap pool for number of days sepcifed by user, i.e user asks to get the data of pool_id= 0x2f5e87C9312fa29aed5c179E456625D79015299c and number_of_days=10 then theis function will return daily data of this pool the data will include daily price, volumeUSD, feesUSD, etc of each of the 10 days"
                                   "ids"
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pool_id": {"type": "string",
                                        "description": "Its the id of Uniswap v3 pool for which user wants to get the daily data"},
                            "number_of_days": {"type": "string",
                                               "description": "Its the number of days to get the data of the desired pool id from uniswap, i.e if user speicifies 10 days then this function will get 10 days daily data of speciified pool id"}
                        },
                        'required': ['N', 'number_of_days']
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "price_impact_analysis",
                    "description": "The function aims to assist users in analyzing the price impact of differnt token sinks(staking, locks, utility allocation, hodling etc). This function gets initial weekly transaction volume of token, initial token price, and initial circulating supply of token from user an nd based on these parameters it calculates the impact of differnt token sinks in the protocol on price of token , "
                                   
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "initial_weekly_transactions": {"type": "string",
                                                            "description": "Initial volume of weekly transactions"},
                            "initial_circulating_supply": {"type": "string",
                                                           "description": "Initial circulating supply of the token"},
                            "initial_price": {
                                "type": "string",
                                "description": "Initial price of the token"
                            }
                        },
                        'required': ["initial_weekly_transactions", "initial_circulating_supply", "initial_price"]
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_dune_client_query_data",
                    "description": "This function gets the data from dune analytics dashboards given the query, user inputs the query id and and this functions returns the data given by this query id "
                                   
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_id": {"type": "string",
                                                            "description": "Id of dune analytics query"},
                            
                        },
                        'required': ["query_id"]
                    },
                },
            },

        ]

        system = [{"role": "system", "content": system_message}]
        user = [{"role": "user", "content": text}]
        history = self.gpt_memory
        messages = system + history + user
        self.memory_manage(text)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        print(response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls is None:
            print(response_message.content)
            self.memory_manage(None, response_message.content)
            return response_message.content
        else:
            if tool_calls:
                function_name = tool_calls[0].function.name
                json_response = json.loads(tool_calls[0].function.arguments)
                json_response['gpt_memory'] = self.memory_manage

                # Step 3: call the function
                # Note: the JSON response may not always be valid; be sure to handle errors
                available_functions = {
                    "get_uniswap_top_pools": get_uniswap_top_pools,
                    # "answer_general_questions": self.general.rag_response,
                    "get_uniswap_pool_day_data": get_uniswap_pool_day_data,
                    "price_impact_analysis": price_impact_analysis,
                    "fetch_dune_client_query_data":fetch_dune_client_query_data,
                }
                function_to_call = available_functions[function_name]

                if function_name == 'get_uniswap_pool_day_data':
                    #function_response = function_to_call(json_response)
                    function_response, image_paths = function_to_call(json_response)
                    #data = f"Data saved as CSV and plots generated. Here are the image paths: {image_paths}"
                    #messages = [{"role": "user", "content": data}]
                    data = f"""Convert this data into  a table format also tell user that this data is saved in csv format in llm_data_dir folder of this project, whihc they can use anywhere they want  {json.dumps(function_response)}"""
                    messages = [{"role": "user", "content": data}]
                    response = self.client.chat.completions.create(
                         model="gpt-4o",
                        messages=messages,
                    )
                    response_message = response.choices[0].message.content
                    self.memory_manage(None, response_message)
                    print(f"image_paths  {image_paths}")
                    return response_message,image_paths
                
                elif function_name == 'get_uniswap_top_pools':
                    function_response = function_to_call(json_response)
                    data = f"""Convert this data into  a table format, also tell user that this data is saved in csv format in llm_data_dir folder of this project, whihc they can use anywhere they want {json.dumps(function_response)}"""
                    messages = [{"role": "user", "content": data}]
                    response = self.client.chat.completions.create(
                         model="gpt-4o",
                        messages=messages,
                    )
                    response_message = response.choices[0].message.content
                    self.memory_manage(None, response_message)
                    return response_message  


                elif function_name == "price_impact_analysis":
                    function_response, image_path = function_to_call(json_response)

                    data = f"""Convert this data into  a table format, also tell user that this data is saved in csv format in llm_data_dir folder of this project {json.dumps(function_response)}"""
                    messages = [{"role": "user", "content": data}]
                    response = self.client.chat.completions.create(
                         model="gpt-4o",
                        messages=messages,
                    )
                    response_message = response.choices[0].message.content
                    self.memory_manage(None, response_message)
                    print(f"image_paths  {image_path}")
                    return response_message, image_path
                else:
                    function_response = function_to_call(json_response)
                    return function_response
