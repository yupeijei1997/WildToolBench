from tool_class.tool_ace import ToolACE
from tool_class.xlam import Xlam
from tool_class.xlam2 import Xlam2
from tool_class.gorilla import Gorilla
from tool_class.llama import Llama
from tool_class.qwen import Qwen
from tool_class.deepseek import DeepSeek
from tool_class.chatglm import ChatGLM
from tool_class.watt import Watt
from tool_class.fc_medium import FC_Medium
from tool_class.hammer import Hammer


tool_model_map = {
    "toolace": ToolACE,
    "toolace2": ToolACE,
    "xlam": Xlam,
    "xlam2-70b": Xlam2,
    "xlam2-32b": Xlam2,
    "xlam2-8b": Xlam2,
    "xlam2-3b": Xlam2,
    "xlam2-1b": Xlam2,
    "gorilla": Gorilla,
    "deepseek": DeepSeek,
    "chatglm": ChatGLM,
    "fcm3.1": FC_Medium,
    ## Watt
    "watt70b": Watt,
    "watt8b": Watt,
    ## Hammer
    "hammer7b": Hammer,
    "hammer3b": Hammer,
    "hammer1.5b": Hammer,
    "hammer0.5b": Hammer,
    ## LLAMA
    "llama70b": Llama,
    "llama8b": Llama,
    "llama3b": Llama,
    "llama1b": Llama,
    ## Qwen
    "qwen72b": Qwen,
    "qwen32b": Qwen,
    "qwen14b": Qwen,
    "qwen7b": Qwen,
    "qwen3b": Qwen,
    "qwen1.5b": Qwen,
    "qwen0.5b": Qwen
}

tool_model_path_map = {
    # toolace
    "toolace": "/xxx/model/ToolACE-8B",
    "toolace2": "/xxx/ToolACE-2-Llama-3.1-8B",
    # xlam
    "xlam": "/xxx/model/xLAM-7b-fc-r",
    "xlam2-70b": "/xxx/Llama-xLAM-2-70b-fc-r",
    "xlam2-32b": "/xxx/xLAM-2-32b-fc-r",
    "xlam2-8b": "/xxx/Llama-xLAM-2-8b-fc-r",
    "xlam2-3b": "/xxx/xLAM-2-3b-fc-r",
    "xlam2-1b": "/xxx/xLAM-2-1b-fc-r",
    # Watt
    "watt70b": "/xxx/model/watt-tool-70B",
    "watt8b": "/xxx/model/watt-tool-8B",
    # Hammer2.1
    "hammer7b": "/xxx/model/Hammer2.1-7b",
    "hammer3b": "/xxx/model/Hammer2.1-3b",
    "hammer1.5b": "/xxx/model/Hammer2.1-1.5b",
    "hammer0.5b": "/xxx/model/Hammer2.1-0.5b",
    # other
    "gorilla": "/xxx/model/gorilla-openfunctions-v2",
    "commandr": "/xxx/model/c4ai-command-r-v01",
    "deepseek": "/xxx/model/DeepSeek-Coder-V2-Lite-Instruct",
    "chatglm": "/xxx/model/glm-4-9b-chat-hf",
    "fcm3.1": "/xxx/model/functionary-medium-v3.1",
    # LLAMA3.3
    "llama70b": "/xxx/model/Llama-3.3-70B-Instruct",
    "llama8b": "/xxx/model/Meta-Llama-3.1-8B-Instruct",
    "llama3b": "/xxx/model/Llama-3.2-3B-Instruct",
    "llama1b": "/xxx/model/Llama-3.2-1B-Instruct",
    # QWEN2.5
    "qwen72b": "/xxx/model/Qwen2.5-72B-Instruct",
    "qwen32b": "/xxx/model/Qwen2.5-32B-Instruct",
    "qwen14b": "/xxx/model/Qwen2.5-14B-Instruct",
    "qwen7b": "/xxx/model/Qwen2.5-7B-Instruct",
    "qwen3b": "/xxx/model/Qwen2.5-3B-Instruct",
    "qwen1.5b": "/xxx/model/Qwen2.5-1.5B-Instruct",
    "qwen0.5b": "/xxx/model/Qwen2.5-0.5B-Instruct",
    # QWEN3
    "qwen3-235b-a22b": "/xxx/model/Qwen3-235B-A22B",
    "qwen3-30b-a3b": "/xxx/model/Qwen3-30B-A3B",
    "qwen3-32b": "/xxx/model/Qwen3-32B",
    "qwen3-14b": "/xxx/model/Qwen3-14B",
    "qwen3-8b": "/xxx/model/Qwen3-8B",
    "qwen3-4b": "/xxx/model/Qwen3-4B",
    "qwen3-1.7b": "/xxx/model/Qwen3-1.7B",
    "qwen3-0.6b": "/xxx/model/Qwen3-0.6B",
}
