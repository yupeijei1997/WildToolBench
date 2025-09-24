from .toolace_handle import ToolACEMultiTurnMessages
from .xlam_handle import XLAMMultiTurnMessages
from .xlam2_handle import XLAM2MultiTurnMessages
from .gorilla_handle import GorillaMultiTurnMessages
from .gpt_handle import GPTMultiTurnMessages, GPTAZUREMultiTurnMessages
from .llama_handle import LlamaMultiTurnMessages
from .qwen_handle import QwenMultiTurnMessages
from .qwq_handle import QwQMultiTurnMessages
from .dsr1_handle import DSR1MultiTurnMessages
from .dsv3_handle import DSV3MultiTurnMessages
from .chatglm_handle import ChatGLMMultiTurnMessages
from .hammer_handle import HammerMultiTurnMessages
from .watt_handle import WattMultiTurnMessages
from .fcm_handle import FCMMultiTurnMessages


tool_handle_map = {
    # toolace
    "toolace": (ToolACEMultiTurnMessages, False),
    "toolace2": (ToolACEMultiTurnMessages, False),
    # xlam
    "xlam": (XLAMMultiTurnMessages, False),
    "xlam2-70b": (XLAM2MultiTurnMessages, False),
    "xlam2-32b": (XLAM2MultiTurnMessages, False),
    "xlam2-8b": (XLAM2MultiTurnMessages, False),
    "xlam2-3b": (XLAM2MultiTurnMessages, False),
    "xlam2-1b": (XLAM2MultiTurnMessages, False),
    # other
    "gorilla": (GorillaMultiTurnMessages, False),
    "chatglm": (ChatGLMMultiTurnMessages, False),
    "gpt4o": (GPTAZUREMultiTurnMessages, True),
    "gemini": (GPTMultiTurnMessages, True),
    "claude": (GPTMultiTurnMessages, True),
    "mistral": (GPTMultiTurnMessages, True),
    "fcm3.1": (FCMMultiTurnMessages, True),
    # Watt
    "watt70b": (WattMultiTurnMessages, True),
    "watt8b": (WattMultiTurnMessages, True),
    # Hammer
    "hammer7b": (HammerMultiTurnMessages, False),
    "hammer3b": (HammerMultiTurnMessages, False),
    "hammer1.5b": (HammerMultiTurnMessages, False),
    "hammer0.5b": (HammerMultiTurnMessages, False),
    # LLAMA
    "llama70b": (LlamaMultiTurnMessages, True),
    "llama8b": (LlamaMultiTurnMessages, True),
    "llama3b": (LlamaMultiTurnMessages, True),
    "llama1b": (LlamaMultiTurnMessages, True),
    # QWEN
    "qwen72b": (QwenMultiTurnMessages, True),
    "qwen32b": (QwenMultiTurnMessages, True),
    "qwen14b": (QwenMultiTurnMessages, True),
    "qwen7b": (QwenMultiTurnMessages, True),
    "qwen3b": (QwenMultiTurnMessages, True),
    "qwen1.5b": (QwenMultiTurnMessages, True),
    "qwen0.5b": (QwenMultiTurnMessages, True),
    "qwq32b": (QwQMultiTurnMessages, False),
}
