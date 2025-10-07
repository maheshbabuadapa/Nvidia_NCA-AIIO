# 🧠 NVIDIA GPU Stack Integration on OpenShift (Kubernetes) — with MIG

## 1️⃣ Overview

This guide explains how **NVIDIA’s AI inference stack** — including **NGC**, **Dynamo**, **Triton**, **CUDA**, and **MIG** — fits together inside a **Kubernetes or OpenShift cluster** to run GPU-accelerated workloads such as Large Language Models (LLMs).

---

## 2️⃣ Key Components

### 🗂️ NVIDIA NGC (NVIDIA GPU Cloud)
- NVIDIA’s official container registry and model hub.  
- Provides pre-built, optimized Docker images, pretrained models, and deployment resources.  
- Examples: `nvcr.io/nvidia/tritonserver:<tag>`, `nvcr.io/nvidia/dynamo:<tag>`.  
- Analogy: *Docker Hub + Hugging Face + App Store* for NVIDIA software.

### ⚙️ NVIDIA DYNAMO
- Distributed inference orchestrator for large LLMs.  
- Manages how workloads are split and scheduled across many GPU nodes.  
- Key features: multi-GPU scheduling, prefill/decode separation, load balancing.  
- Analogy: **Kubernetes control plane** for AI inference.

### 🧩 NVIDIA TRITON INFERENCE SERVER
- Model-serving runtime running inside each GPU node.  
- Executes inference on models using TensorRT-LLM or PyTorch.  
- Capabilities: multi-framework support, batching, HTTP/gRPC inference APIs.  
- Analogy: **kubelet + Docker Engine** for GPU workloads.

### ⚡ CUDA / TensorRT / Drivers
- Low-level GPU runtime stack communicating with hardware.  
- Components: CUDA, cuDNN, NCCL, TensorRT.  
- Installed via NVIDIA GPU Operator in OpenShift.

---

## 3️⃣ Integration Layer Diagram

```
App (Chatbot / RAG / API)
          │
          ▼
NVIDIA DYNAMO (Orchestrator)
          │
          ▼
NVIDIA TRITON (Model Server)
          │
          ▼
CUDA / TensorRT / Drivers
          │
          ▼
GPU Hardware (A100 / H100)
          ▲
          │
NVIDIA NGC Registry (source for containers & models)
```

---

## 4️⃣ OpenShift Integration Flow

1. Install **NVIDIA GPU Operator** (adds driver, CUDA, device plugin).  
2. Pull NVIDIA containers from **NGC**.  
3. Deploy **Triton** pods on GPU nodes.  
4. Deploy **Dynamo** as orchestrator.  
5. Expose Dynamo via OpenShift Route.  
6. Application → Dynamo → Triton → CUDA → GPU.

---

## 5️⃣ Analogy Table

| Component | Role | Analogy |
|------------|------|---------|
| NGC | Distribution for containers/models | Docker Hub |
| Dynamo | Orchestrator | Kubernetes master |
| Triton | Local executor | kubelet + Docker |
| CUDA / TensorRT | Compute runtime | CPU instruction set |
| GPU Operator | Device setup | Cloud-init for GPUs |

---

## 6️⃣ Simplified Cluster Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ OpenShift / Kubernetes Cluster                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ NVIDIA DYNAMO (acts like K8s Scheduler)                │ │
│ └─────────────────────────────────────────────────────────┘ │
│    │                         │                               │
│    ▼                         ▼                               │
│ ┌──────────────┐     ┌──────────────┐                        │
│ │ TRITON Node1 │     │ TRITON Node2 │                        │
│ │ CUDA/TensorRT│     │ CUDA/TensorRT│                        │
│ └──────▲───────┘     └──────▲───────┘                        │
│        │ Drivers & GPU Operator                               │
│ ┌──────┴─────────────┐ ┌──────┴──────────────┐               │
│ │ Physical GPU A100  │ │ Physical GPU H100   │               │
│ └────────────────────┘ └─────────────────────┘               │
│ [Containers pulled from NGC]                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 7️⃣ Summary Checklist

| Step | Task | Tool |
|------|------|------|
| 1 | Install GPU Operator | OpenShift OperatorHub |
| 2 | Pull Containers | NGC (nvcr.io) |
| 3 | Deploy Triton Pods | OpenShift Deployment |
| 4 | Deploy Dynamo | StatefulSet |
| 5 | Expose Services | Route/Ingress |
| 6 | Send Requests | App → Dynamo → Triton → CUDA → GPU |

---

## 8️⃣ Multi‑Instance GPU (MIG)

### 🔹 What is MIG
**MIG (Multi‑Instance GPU)** is available on NVIDIA A100, H100, and newer GPUs.  
It splits one physical GPU into multiple smaller, isolated “virtual GPUs,” each with dedicated compute and memory.

Each instance behaves as an independent GPU with guaranteed resources.

### ⚙️ How MIG Works

| MIG Profile | Meaning | Memory | Instances per GPU | Use Case |
|--------------|----------|---------|-------------------|----------|
| 1g.5gb | 1 GPU slice + 5 GB memory | 5 GB | up to 7 | Small inference jobs |
| 2g.10gb | 2 GPU slices + 10 GB memory | 10 GB | up to 3 | Medium workloads |
| 3g.20gb | 3 GPU slices + 20 GB memory | 20 GB | up to 2 | Large models |
| 7g.40gb | Full GPU | 40 GB | 1 | Full LLM training/inference |

🧩 **Format:** `<g>g.<m>gb` → “g” = compute portion, “gb” = memory size.

**Example:** `1g.5gb` → one-seventh of the GPU with 5 GB HBM memory.

### 🍕 Pizza Analogy
Imagine your GPU as a pizza:  
- `1g.5gb` → 7 small slices  
- `2g.10gb` → 3 medium slices  
- `7g.40gb` → whole pizza.  
Each slice is separate — no one eats another’s share!

### 🖥️ MIG in OpenShift Stack

```
App → Dynamo → Triton → CUDA → MIG Instances → Physical GPU
```

Each MIG instance appears to CUDA and Triton as a separate GPU device.

### 🧩 Steps to Enable MIG

```bash
sudo nvidia-smi -i 0 -mig 1                 # Enable MIG mode
sudo nvidia-smi mig -cgi 1g.5gb,2g.10gb -C  # Create instances
```

Kubernetes (via the GPU Operator) detects each MIG slice as:
```
nvidia.com/mig-1g.5gb
nvidia.com/mig-2g.10gb
```

Pods can request MIG resources:
```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1
```

### 🚀 Benefits of MIG

| Benefit | Description |
|----------|-------------|
| Better GPU utilization | Share one GPU across workloads |
| Isolation | Each instance has its own memory and compute |
| Predictable performance | Workloads don’t interfere |
| Cost efficiency | Multi-tenant friendly |
| Scalability | Many small pods on one GPU |

### ✅ Summary
> **MIG = GPU Virtualization.**  
> It divides a GPU into isolated slices, each seen as a full GPU by CUDA, Triton, and Dynamo.  
> In the stack, MIG sits **between CUDA and the physical GPU**, giving OpenShift fine-grained GPU scheduling.
