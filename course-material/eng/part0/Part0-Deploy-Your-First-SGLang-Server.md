# Part 0 — Before you learn

## Deploying SGLang on a local GPU

In this lesson you will use SGLang to run Qwen3-0.6B on your own GPU and send your first request. Qwen3-0.6B is the smallest model in the Qwen3 family: about 1.5 GB of weights, runs in 4 GB of VRAM, and serves as our teaching model. Later in the course we will show you how to build your own mini-sglang and serve a model with it.
(You can also point your coding agent at this lesson and let it run through the steps automatically.)

### Requirements

- NVIDIA GPU with ≥ 4 GB of VRAM. RTX 30/40/50 series cards work out of the box; older cards such as the RTX 20 series or T4 need an extra flag, see the FAQ at the end.
- Linux or WSL2. Native Windows is not supported. macOS is not supported; if you have no NVIDIA card, see the end of this lesson.
- NVIDIA driver installed. Verify with:

```bash
nvidia-smi
```

If it shows your GPU and the CUDA Version in the top-right corner is ≥ 12.x, you are good. That version number decides which set of install commands to use below. You do not need to install the CUDA Toolkit separately: SGLang's dependencies ship their own CUDA runtime. On WSL2 the driver is installed on the Windows side; nothing extra is needed inside WSL2.

### Python environment

Python ≥ 3.10 is required. We recommend a dedicated environment:

```bash
conda create -n sglang python=3.12 -y
conda activate sglang
```

Without conda, venv works the same:

```bash
python3 -m venv ~/sglang-env
source ~/sglang-env/bin/activate
```

Run every command that follows inside this environment. A new terminal needs to activate it again.

### Installing SGLang

SGLang is built against CUDA 13 by default. Pick the install commands by the CUDA Version shown in the top-right corner of `nvidia-smi`.

**CUDA Version ≥ 13.0**:

```bash
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow sglang
```

**CUDA Version 12.x**:

```bash
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow sglang
uv pip install --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
```

The last three commands swap torch and the kernels for CUDA 12 builds. Alternatively, upgrade the driver to ≥ 580 and use the CUDA 13 commands directly.

Verify:

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

If it prints a version number, the install succeeded. Install steps change between releases; if you hit errors, the [official installation docs](https://docs.sglang.io/get_started/install.html) are the source of truth.

### Faster model downloads (users in mainland China)

Model weights are hosted on Hugging Face and download automatically the first time the server starts. Direct downloads from mainland China are slow; pick either of the following.

Use the hf-mirror mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Or download from ModelScope:

```bash
uv pip install modelscope
export SGLANG_USE_MODELSCOPE=true
```

`export` only affects the current terminal. The terminal that launches the server must have it set, or it has no effect. To make it permanent:

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

### Launching the server

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000
```

The first run downloads about 1.5 GB of weights. The server is up once this line appears in the log:

```
The server is fired up and ready to roll!
```

Closing this terminal stops the server, so keep it running. Do the following steps in a new terminal.

Deployment parameters for the whole Qwen3 family across different hardware are in the [SGLang Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3). The defaults are fine for this lesson.

### Sending requests

Check that the server is alive:

```bash
curl http://localhost:30000/health
```

Send your first chat request:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 512
  }'
```

If the returned JSON contains the model's answer, it worked.

SGLang's API is OpenAI-compatible. Calling it from Python needs the openai library (`uv pip install openai`):

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

Qwen3 emits a `<think>...</think>` reasoning block before its answer by default. This is not a bug. If you do not need the reasoning, disable it in the request:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Or launch the server with `--reasoning-parser qwen3`, which separates the reasoning into the response's `reasoning_content` field.

### FAQ

**OOM (out of memory)**: little VRAM, or the GPU is also driving a desktop. Lower SGLang's share of memory and shorten the context:

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --mem-fraction-static 0.6 --context-length 8192 --port 30000
```

**RTX 20 series / T4 and other older cards fail to start**: the default attention backend needs a newer architecture. Switch to triton:

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --attention-backend triton --port 30000
```

**CUDA errors at startup (such as "CUDA driver version is insufficient" / "no kernel image is available")**: the driver is CUDA 12.x but the default CUDA 13 dependencies were installed. Fix: run the three force-reinstall commands from the CUDA 12 part of "Installing SGLang".

**Download hangs**: the usual cause is that `HF_ENDPOINT` / `SGLANG_USE_MODELSCOPE` was not set in the terminal that launched the server.

**address already in use**: the port is taken. Use another port (such as `--port 30001`) and change the port in your request commands to match.

**nvidia-smi not found in WSL2**: the driver has to be installed on the Windows side (download it from NVIDIA's website). After installing, run `wsl --shutdown` in PowerShell to restart WSL.

**Anything else**: post the full command and the full error (as text, not a screenshot) in the course group. Attaching your `nvidia-smi` output and SGLang version speeds up diagnosis a lot.

### No NVIDIA GPU

The course experiments assume an NVIDIA environment. SGLang also supports [AMD Instinct](https://docs.sglang.io/docs/hardware-platforms/amd_gpu), [Intel Xeon CPU](https://docs.sglang.io/docs/hardware-platforms/cpu_server), [Apple Silicon (experimental)](https://docs.sglang.io/docs/hardware-platforms/apple_metal) and other platforms, but later lessons are not guaranteed to work on them, so we do not recommend them for this course.

If you are on a Mac or have no NVIDIA card, rent an entry-level GPU by the hour on a cloud GPU platform such as AutoDL (the cheapest card that runs Qwen3-0.6B is enough). The rented machine is a ready-made Linux environment; start from the "Python environment" section. If the course provides shared compute, we will announce it in the course group.
