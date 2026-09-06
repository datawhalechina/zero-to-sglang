# Part 0 — Before you learn

## 本地 GPU 部署 SGLang

本课在你自己的 GPU 上用 SGLang 跑起 Qwen3-0.6B（Qwen3 系列最小的模型，权重约 1.5 GB，4 GB 显存可跑，作为教学模型），并发出第一个请求。后续课程中，我们会教大家如何build自己的mini-sglang，并且serve起来大模型。
（也可以让自己的coding agent看本课的内容然后自动化跑一跑）

### 环境要求

- NVIDIA GPU，显存 ≥ 4 GB。RTX 30/40/50 系可直接跑；RTX 20 系 / T4 等老卡需加参数，见文末常见问题。
- Linux 或 WSL2。不支持 Windows 原生。不支持 macOS，没有 N 卡的见文末。
- NVIDIA 驱动已安装。验证命令：

```bash
nvidia-smi
```

能显示显卡信息且右上角 CUDA Version ≥ 12.x 即为正常。这个版本号决定后文安装 SGLang 时选哪组命令。CUDA Toolkit 无需单独安装，SGLang 的依赖自带 CUDA 运行时。WSL2 环境下驱动安装在 Windows 侧，WSL2 内无需再装。

### Python 环境

要求 Python ≥ 3.10，建议在独立环境中安装：

```bash
conda create -n sglang python=3.12 -y
conda activate sglang
```

没有 conda 时用 venv 效果相同：

```bash
python3 -m venv ~/sglang-env
source ~/sglang-env/bin/activate
```

后续所有命令都在此环境中执行；新开的终端需要重新 activate。

### 安装 SGLang

SGLang 默认按 CUDA 13 构建，安装命令按 `nvidia-smi` 右上角的 CUDA Version 区分。

**CUDA Version ≥ 13.0**：

```bash
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow sglang
```

**CUDA Version 12.x**：

```bash
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow sglang
uv pip install --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
```

后三条把 torch 和 kernel 换成 CUDA 12 构建。也可以把驱动升级到 ≥ 580 后直接走 CUDA 13 命令。

验证：

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

能打印版本号即安装成功。安装方式随版本更新，报错时以[官方安装文档](https://docs.sglang.io/get_started/install.html)为准。

### 模型下载加速（国内用户）

模型权重在 Hugging Face 上，首次启动 server 时自动下载。国内直连很慢，以下两种办法任选其一。

用 hf-mirror 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或从 ModelScope 下载：

```bash
uv pip install modelscope
export SGLANG_USE_MODELSCOPE=true
```

`export` 只对当前终端生效，启动 server 的终端必须设置过，否则不起作用。永久生效的写法：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

### 启动 server

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000
```

首次运行会先下载约 1.5 GB 权重。等日志出现下面这行即启动成功：

```
The server is fired up and ready to roll!
```

这个终端关闭后服务即停止，需保持运行；后续操作在新终端进行。

Qwen3 全系列在各种硬件上的部署参数见 [SGLang Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3)，本课用默认参数即可。

### 发请求

确认服务存活：

```bash
curl http://localhost:30000/health
```

发第一个对话请求：

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用一句话介绍一下你自己。"}],
    "max_tokens": 512
  }'
```

返回的 JSON 里有模型回答即成功。

SGLang 的接口与 OpenAI API 兼容，Python 调用需要 openai 库（`uv pip install openai`）：

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "列举 3 个国家和它们的首都。"}],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

Qwen3 默认会先输出一段 `<think>...</think>` 思考过程，不是 bug。不需要思考过程时，可在请求里关闭：

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "列举 3 个国家和它们的首都。"}],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

或者启动 server 时加 `--reasoning-parser qwen3`，思考内容会被分离到响应的 `reasoning_content` 字段。

### 常见问题

**OOM（out of memory）**：显存小或显卡同时在跑桌面。降低 SGLang 的显存占比并调短上下文：

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --mem-fraction-static 0.6 --context-length 8192 --port 30000
```

**RTX 20 系 / T4 等老卡启动报错**：默认 attention backend 需要较新架构，换 triton：

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --attention-backend triton --port 30000
```

**启动报 CUDA 相关错误（如 "CUDA driver version is insufficient" / "no kernel image is available"）**：驱动是 CUDA 12.x 但装了默认的 CUDA 13 依赖。解决办法是补跑"安装 SGLang"一节 CUDA 12 的三条 force-reinstall 命令。

**下载卡住**：常见原因是 `HF_ENDPOINT` / `SGLANG_USE_MODELSCOPE` 没有设置在启动 server 的那个终端里。

**address already in use**：端口被占用。换用其他端口（如 `--port 30001`）即可，请求命令里的端口同步修改。

**WSL2 里找不到 nvidia-smi**：驱动需要装在 Windows 侧（NVIDIA 官网下载），装好后在 PowerShell 执行 `wsl --shutdown` 重启 WSL 即可。

**其他问题**：可以把完整命令和完整报错（文本形式，非截图）发到课程群提问，附上 `nvidia-smi` 输出和 SGLang 版本号能大幅加快定位。

### 没有 NVIDIA 显卡

课程实验以 NVIDIA 环境为准。SGLang 也支持 [AMD Instinct](https://docs.sglang.io/docs/hardware-platforms/amd_gpu)、[Intel Xeon CPU](https://docs.sglang.io/docs/hardware-platforms/cpu_server)、[Apple Silicon（实验性）](https://docs.sglang.io/docs/hardware-platforms/apple_metal)等平台，但后续课程不保证适用，不建议用于本课程。

Mac 和无 N 卡的同学可以在 AutoDL 等平台按小时租一张入门级 GPU（跑 Qwen3-0.6B 最便宜的卡即可），租到的机器就是现成的 Linux 环境，从"Python 环境"一节开始操作即可。课程如提供统一算力，会在课程群另行通知。
