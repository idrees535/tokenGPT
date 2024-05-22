from openai import OpenAI
import json
from uniswap import get_uniswap_top_pools, get_uniswap_pool_day_data, price_impact_analysis
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage


class TokenGPT:
    def __init__(self):
        self.initial_text = "My name is Token GPT and I am here to help you on tokennomics"
        self.gpt_memory = [{"role": "assistant", "content": self.initial_text}]
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key="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"
        )

    def memory_manage(self, human=None, tokengpt=None):
        if human is not None:
            self.gpt_memory.append({"role": "user", "content": human})

        if tokengpt is not None:
            self.gpt_memory.append({"role": "assistant", "content": tokengpt})

    def conversation(self, text):
        system_message = f"""Your AI token engineering Assistant which will help you to navigate in complex landscape of 
        design, verification and optimization of decentralized token based economies.
        Select the best function that describes the user query and call it and provide assitance
        If something isn't clear and arguments is not in the data you can ask the proactive question from the user to fill it out"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_uniswap_top_pools",
                    "description": "The function aims to assist users in getting the top pools from uniswap and presents its "
                                   "ids"
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "N": {"type": "string",
                                  "description": "Its just the number of ids to get the pools from uniswap"}, },
                        'required': ['N']
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_uniswap_pool_day_data",
                    "description": "The function aims to assist users in getting the top pools from uniswap and presents its "
                                   "ids"
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pool_id": {"type": "string",
                                        "description": "Its the id of pool to get the data from the uniswap"},
                            "number_of_days": {"type": "string",
                                               "description": "Its the number of days to get the data of the desired pool id from uniswap"}
                        },
                        'required': ['N', 'number_of_days']
                    },
                },
            },

            {
                "type": "function",
                "function": {
                    "name": "price_impact_analysis",
                    "description": "The function aims to assist users in analyzing the price impact over a specified number "
                                   "of weeks based on various transaction and market parameters."
                    ,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "initial_weekly_transactions": {"type": "string",
                                                            "description": "Initial number of weekly transactions"},
                            "initial_circulating_supply": {"type": "string",
                                                           "description": "Initial circulating supply of the asset"},
                            "initial_price": {
                                "type": "string",
                                "description": "Initial price of the asset"
                            }
                        },
                        'required': ["initial_weekly_transactions", "initial_circulating_supply", "initial_price"]
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
                    "price_impact_analysis": price_impact_analysis
                }
                function_to_call = available_functions[function_name]
                if function_name == 'get_uniswap_pool_day_data':
                    function_response = function_to_call(json_response)
                    data = f"""Convert this data into insightful information for user in humanly textual message {json.dumps(function_response)}"""
                    messages = [{"role": "user", "content": data}]
                    response = self.client.chat.completions.create(
                         model="gpt-4o",
                        messages=messages,
                    )
                    response_message = response.choices[0].message.content
                    self.memory_manage(None, response_message)
                    return response_message
                elif function_name == "price_impact_analysis":
                    image, df_data = function_to_call(json_response)
                    self.memory_manage(None, df_data)
                    return image
                else:
                    function_response = function_to_call(json_response)
                    return function_response
