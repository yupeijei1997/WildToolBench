from .gpt_handle import GPTMultiTurnMessages, GPTAZUREMultiTurnMessages


agent_handle_map = {
    "gpt4o": GPTAZUREMultiTurnMessages,
    "gpt4-turbo": GPTAZUREMultiTurnMessages,
    "o1": GPTAZUREMultiTurnMessages,
    "claude3.5": GPTAZUREMultiTurnMessages
}
