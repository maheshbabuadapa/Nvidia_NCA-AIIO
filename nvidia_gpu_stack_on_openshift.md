# 🧠 NVIDIA GPU Stack Integration on OpenShift (Kubernetes)

## 1️⃣ Overview

This guide explains how **NVIDIA’s AI inference stack** — including **NGC**, **Dynamo**, **Triton**, and **CUDA** — fits together inside a **Kubernetes or OpenShift cluster** to run GPU-accelerated workloads such as Large Language Models (LLMs).

---

## 2️⃣ Key Components

### 🗂️ **NVIDIA NGC (NVIDIA GPU Cloud)**
- **What it is:** NVIDIA’s official container registry and model hub.  
- **Purpose:** Provides pre-built, optimized Docker images, pretrained models, and deployment resources.  
- **Examples of images:**  
  - `nvcr.io/nvidia/tritonserver:<tag>` → Triton Inference Server  
  - `nvcr.io/nvidia/dynamo:<tag>` → Dynamo orchestrator  
  - `nvcr.io/nvidia/pytorch`, `nvcr.io/nvidia/tensorrt`, etc.  
- **Analogy:** Like *Docker Hub + Hugging Face + App Store* for NVIDIA software.

---

### ⚙️ **NVIDIA DYNAMO**
- **What it is:** A **distributed inference orchestrator** for large LLMs.  
- **Role:** Manages how inference workloads are split and scheduled across many GPU nodes.  
- **Key Features:**
  - Multi-GPU and multi-node scheduling  
  - Prefill / decode phase separation  
  - Cluster-level load balancing  
  - Works closely with TensorRT-LLM and Triton  
- **Analogy:** Acts like **Kubernetes control plane** for AI inference — a *scheduler/brain* for GPU clusters.

---

### 🧩 **NVIDIA TRITON INFERENCE SERVER**
- **What it is:** A **model-serving runtime** that runs inside each GPU node.  
- **Role:** Executes actual inference on models using TensorRT-LLM or PyTorch.  
- **Capabilities:**
  - Supports many frameworks (TensorRT, PyTorch, ONNX, TF, etc.)  
  - Provides HTTP/gRPC APIs for inference requests  
  - Handles batching, concurrency, and model versioning  
- **Analogy:** Like **kubelet + Docker Engine** — it runs workloads locally under Dynamo’s orchestration.

---

### ⚡ **CUDA / TensorRT / Drivers**
- **Purpose:** Low-level GPU runtime stack that directly communicates with hardware.  
- **Components:**
  - **CUDA** – core compute API for GPUs  
  - **cuDNN / NCCL** – deep learning & communication libraries  
  - **TensorRT / TensorRT-LLM** – optimized inference runtime  
- **Installed via:** NVIDIA GPU Operator in OpenShift (automates driver & toolkit setup).

---

## 3️⃣ How They Fit Together

### 🧱 Layered Architecture

```
+-------------------------------------------------------------+
| 🧠 Application Layer (Chatbot / RAG / API Gateway)           |
+-------------------------------------------------------------+
                             │
                             ▼
+-------------------------------------------------------------+
| ⚙️ NVIDIA DYNAMO (Orchestrator)                             |
|  - Distributes inference jobs across GPU nodes               |
|  - Communicates with Triton servers via gRPC/IPC             |
+-------------------------------------------------------------+
                             │
                             ▼
+-------------------------------------------------------------+
| 🧩 NVIDIA TRITON INFERENCE SERVER (on each GPU node)         |
|  - Runs models using TensorRT-LLM / PyTorch                  |
|  - Handles batching, versioning, concurrency                 |
+-------------------------------------------------------------+
                             │
                             ▼
+-------------------------------------------------------------+
| ⚡ CUDA / cuDNN / TensorRT / Drivers                         |
|  - Installed by GPU Operator                                 |
|  - Talks directly to GPU hardware                            |
+-------------------------------------------------------------+
                             │
                             ▼
+-------------------------------------------------------------+
| 🖥️ Physical GPUs (A100 / H100) + Linux OS                    |
+-------------------------------------------------------------+
                             ▲
                             │
+-------------------------------------------------------------+
| 🗂️ NVIDIA NGC (Registry & Model Hub)                         |
|  - Source of containers, models, and SDKs                    |
+-------------------------------------------------------------+
```

---

## 4️⃣ Integration with OpenShift (Kubernetes)

1. **Install GPU Operator**  
   - Deploys drivers, CUDA runtime, DCGM exporter, and device plugin.  
   - Makes GPUs available as schedulable resources.

2. **Pull NVIDIA Containers from NGC**  
   - Example:  
     ```bash
     oc new-app nvcr.io/nvidia/tritonserver:24.09-py3
     ```
   - Or use Helm charts from NGC for Triton/Dynamo.

3. **Deploy Triton Pods on GPU Nodes**  
   - Triton runs as a container on GPU nodes.  
   - Each pod mounts model artifacts (e.g., from S3, PVC, or ConfigMap).

4. **Deploy NVIDIA Dynamo**  
   - Runs as a cluster service (Deployment or StatefulSet).  
   - Schedules and orchestrates inference requests across Triton pods.

5. **Expose an API or Route**  
   - Use OpenShift Routes / Ingress to expose Dynamo’s endpoint to applications.

6. **Connect Applications**  
   - Apps send inference requests → Dynamo → Triton Pods → CUDA → GPU.

---

## 5️⃣ Analogy Summary

| Concept | Role | Analogy |
|----------|------|----------|
| **NVIDIA NGC** | Distribution platform for containers and models | Docker Hub + Model Zoo |
| **NVIDIA Dynamo** | Cluster-wide scheduler/orchestrator | Kubernetes control plane |
| **NVIDIA Triton** | Node-level inference runtime | kubelet + Docker |
| **CUDA / TensorRT** | GPU compute engine | CPU instruction set / kernel |
| **GPU Operator** | Sets up GPU drivers & device plugin | Cloud-Init for GPUs |

---

## 6️⃣ Simplified Cluster Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                OpenShift / Kubernetes Cluster               │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 NVIDIA DYNAMO Service                 │  │
│  │  (acts like K8s Scheduler for inference)              │  │
│  └───────────────────────────────────────────────────────┘  │
│               │                         │                   │
│               ▼                         ▼                   │
│  ┌──────────────────┐        ┌──────────────────┐            │
│  │  TRITON Server   │        │  TRITON Server   │            │
│  │  (GPU Node 1)    │        │  (GPU Node 2)    │            │
│  │  CUDA / TensorRT  │        │  CUDA / TensorRT │            │
│  └───────▲──────────┘        └───────▲──────────┘            │
│          │ Drivers & GPU Operator    │                        │
│  ┌───────┴──────────────┐   ┌────────┴──────────────┐        │
│  │ Physical GPU (A100)   │   │ Physical GPU (H100)   │        │
│  └───────────────────────┘   └───────────────────────┘        │
│                                                             │
│  [Containers pulled from NVIDIA NGC registry]                │
└─────────────────────────────────────────────────────────────┘
```

---

## 7️⃣ Summary Checklist

| Step | Task | Tool |
|------|------|------|
| 1 | Install NVIDIA GPU Operator | OpenShift OperatorHub |
| 2 | Pull required containers | NGC (`nvcr.io`) |
| 3 | Deploy Triton pods | OpenShift Deployments |
| 4 | Deploy Dynamo orchestrator | StatefulSet or Deployment |
| 5 | Connect via Route / Service | OpenShift Route |
| 6 | Send inference requests | App → Dynamo → Triton → CUDA → GPU |

---

✅ **Key takeaway:**  
> **NGC** gives you the containers,  
> **Dynamo** orchestrates inference across nodes,  
> **Triton** executes workloads on GPUs,  
> **CUDA + Drivers** enable the hardware,  
> and **OpenShift** ties everything together as your deployment platform.
