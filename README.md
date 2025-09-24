# Benchmarking LLM Tool-Use in the Wild


<p align="center">
    📖 <a>English</a> •
    <a href="README_ZH.md">中文</a>
</p>


![Example](./picture/benchmark_comparison.png)


## 📖 Overview

Fulfilling user needs through Large Language Model multi-turn, multi-step tool-use
is rarely a straightforward process. Real user interactions are inherently wild, being
unpredictable, messy, and flexible. We identify three key challenges from user
behaviour: compositional tasks that demand orchestration of tool-call topologies,
implicit intent spread across dialogue turns requiring contextual inference, and
instruction transition that mix task queries, clarifications, and casual conversation,
forcing LLMs to adjust their policies on the fly. Existing benchmarks overlook these
behaviors, causing the progress of LLMs observed on tool-use to be spurious. To
address this, we introduce WildToolBench, a LLM tool-use benchmark grounded in
real-world user behavior patterns. Comprehensive evaluations of 57 LLMs reveal
that no model achieves an accuracy of more than 15%, indicating a substantial
gap in the robustness of LLMs’ agentic ability. Controlled experiments and in-
depth analyses further indicate that the real challenge for LLM tool-use lies not in
artificially complex tasks, but in the wild nature of user behavior, emphasizing the
need to reconsider the interactions among LLMs, users, and tools.

## 😊 Key Materials

- Test data location: wildtoolbench/data/WildToolBench.jsonl
- Location of the prediction results for the 57 models reported in the paper: wildtoolbench/bench_test/result
- More detailed information about the WildToolBench can be found below

## ⚡️ Quickstart

### Basic Installation

```bash
# Create a new Conda environment with Python 3.10
conda create -n WildToolBench python=3.10
conda activate WildToolBench

# Change directory to the `wildtoolbench`
cd wildtoolbench/

# Install the package
pip install -r requirements.txt
```

## ⏳ Inference

### 🤖 API Models
This project supports multiple API models, including: GPT-5、GPT-4o、o1, etc.

Taking GPT-4o as an example, set the following key in the environment variables

```bash
export OPENAI_MODEL=xxxxxxxxx
export OPENAI_API_KEY=xxxxxxxxx
export OPENAI_BASE_URL=xxxxxxxxx
```

If using AZURE, set the following key:

```bash
export AZURE_OPENAI_DEPLOYMENT=xxxxxxxxx
export AZURE_OPENAI_ENDPOINT=xxxxxxxxx
export AZURE_OPENAI_API_KEY=xxxxxxxxx
export AZURE_OPENAI_API_VERSION=xxxxxxxxx
```

Afterwards, use the following code to request model results, setting the model to gpt4o. If the test is unexpectedly interrupted midway, you can modify the continue_file to continue the test. This will ensure that the already predicted results will not be predicted again.

```bash
cd wildtoolbench/bench_test

python3 request_pipeline.py \
    --model=gpt4o \
    --data_path=./data/WildToolBench.jsonl \
    --output_path=./result \
    --language=en \
    --continue_file=empty.jsonl \
    --remove_role=True \
    --contain_context=True
```

### 🤗 HuggingFace Models

This project also supports a variety of open-source specialized models and open-source general models, as follows:

Open-source specialized models include: xLAM2 series、watt-tool series、ToolACE2-8B、ToolACE-8B、Hammer2.1 series, etc.

Open-source general models include: Qwen3 series, Qwen2.5 series, Llama-3.3 series、DeepSeek-R1、DeepSeek-V3, etc.

Taking [Qwen2.5-7B-Instruct](https://qwen.readthedocs.io/en/latest/framework/function_call.html) as an example:  

First, you need to download the model to a specific directory, and then replace the directory path into the `tool_model_path_map` variable in `wildtoolbench/tool_calls/tool_model_map.py`.  

After that, you can deploy the model using the following code.

```bash
python3 web_server.py qwen7b
```

Afterward, use the following code to request the model results, set the model to qwen7b, and set the model_url to the IP and port number of your deployment machine, for example: http://111.111.111.111:12345. If the test stops unexpectedly, you can modify the continue_file to continue testing.

```bash
python3 request_pipeline.py \
    --model=qwen7b \
    --data_path=./data/WildToolBench.jsonl \
    --output_path=./result \
    --language=en \
    --model_url=MODEL_URL \
    --continue_file=empty.jsonl \
    --remove_role=True \
    --contain_context=True
```


Finally, in wildtoolbench/bench_test/handle/handles.py, we have enumerated the 10 types of Handles that we have implemented. If you want to test other models, you can refer to this file to get the settings for the model parameters. Additionally, if you want to add your own implemented Handle, you can also do so in this file.

## 💫 Evaluation

Use the following code to evaluate the model's prediction results. Fill in PREDICT_DATA_FILE with the corresponding prediction file from the previous step's ./result directory. The evaluation results include: matrix accuracy for action type and layer, individual accuracy for action type and layer, multi-tool invocation result analysis, error type analysis, true/false multi-turn accuracy, true multi-turn subtype accuracy, and parameter error type analysis.

Detailed results will be output to data_with_details.csv.

```bash
cd wildtoolbench/bench_test

python3 analysis_result.py \
    --data_file PREDICT_DATA_FILE \
    --output_csv_flag=True \
    --output_csv_path=./data_with_details.csv
```

## 🧠 Controllable Multi Agent Data Generation Framework

### ⚡️ Quickstart

Taking the example where all agents use AZURE GPT-4o as the base model, and generate data in English. First, set the following key in the environment variables.

```bash
export AZURE_OPENAI_DEPLOYMENT=xxxxxxxxx
export AZURE_OPENAI_ENDPOINT=xxxxxxxxx
export AZURE_OPENAI_API_KEY=xxxxxxxxx
export AZURE_OPENAI_API_VERSION=xxxxxxxxx
export LANGUAGE=en
```

The core innovation of this paper lies in the fact that our proposed WildToolBench is capable of covering all possible action spaces for any number of tasks, and except for the first round, all are true multi-turn tasks. Therefore, our framework can support the generation of data for any number of tasks. Here we take the generation of four tasks as an example, with the reference code as follows:
```bash
cd multi_agent

python3 generate.py \
    --layer_num_total 4 \
    --user_model ["gpt4o"] \
    --planner_model "gpt4o" \
    --tool_model "gpt4o" \
    --agent_model "gpt4o" \
    --checker_model "gpt4o"
```
In particular, we have preserved data in three formats, including: raw generated data, streamlined data (with checker messages and role-switch messages removed), and captured data from each agent, available for researchers in need.

Finally, let's showcase a portion of the results generated for an actual case. It can be observed that the Continue Question User Agent has generated a true multi-turn task of implicit type, "Who is the uploader of the third wallpaper?" This implies that in order to solve this task, the model needs to refer to the preceding context to obtain the ID 778899 of the third wallpaper, and then continue to invoke the tool getWaifuDetails to retrieve detailed information about the wallpaper.

```
User: I need the latest 5 anime wallpapers.

Planner：getWaifuImages(limit=5, tags=["anime wallpapers"])

Tool：...

Planner：prepare_to_answer(answer_type=tool)

AI:

Here are the latest 5 anime wallpapers:
1. Wallpaper ID: 112233
 - [View Wallpaper](https://wallhaven.cc/w/112233)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/112233.jpg)
 - Resolution: 1920x1080
2. Wallpaper ID: 445566
 - [View Wallpaper](https://wallhaven.cc/w/445566)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/445566.jpg)
 - Resolution: 2560x1440
3. Wallpaper ID: 778899
 - [View Wallpaper](https://wallhaven.cc/w/778899)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/778899.jpg)
 - Resolution: 3840x2160
4. Wallpaper ID: 101112
 - [View Wallpaper](https://wallhaven.cc/w/101112)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/101112.jpg)
 - Resolution: 1920x1080
5. Wallpaper ID: 131415
 - [View Wallpaper](https://wallhaven.cc/w/131415)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/131415.jpg)
 - Resolution: 2560x1440

You can click on the links to view the full wallpapers. Hope you enjoy these wallpapers! If you need more help, please let me know.

Continue Question User Agent Generate:
User: Who is the uploader of the third wallpaper?

Planner：getWaifuDetails(image_id=778899)
```
