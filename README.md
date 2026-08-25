# BreastCare-AI

## Overview

BreastCare-AI is an **end-to-end breast ultrasound analysis platform** that combines deep-learning lesion segmentation, quantitative lesion analysis, segmentation-quality assessment, and a bounded agentic workflow.

The system segments breast lesions from ultrasound images, extracts quantitative geometric features, performs configurable quality checks, and generates a structured analysis report through a tool-based `BreastCareAgent`.

> **Important:** BreastCare-AI is a research/engineering prototype and is not a clinical diagnostic system.

---

## Project Objectives

1. Build a deep-learning model for breast ultrasound lesion segmentation
2. Orchestrate a complete analysis workflow from validation to reporting
3. Extract quantitative lesion measurements such as area, perimeter, circularity, and aspect ratio
4. Provide automated segmentation sanity and quality checks
5. Provide workflow-level observability through agent execution traces
6. Expose the analysis pipeline through a REST API and interactive web interface

---

## What We Built

### Machine Learning Model

- **Model:** HybridViTGABVSSMUNet
- **Architecture:** Hybrid segmentation architecture combining Vision Transformer components, Gated Attention Blocks, and Vision State-Space Model components
- **Inference:** PyTorch with CUDA acceleration when available
- **Input:** Preprocessed 256×256 model input
- **Output:** Binary lesion segmentation mask

### Fold 1 Validation Performance

| Metric | Result |
|---|---:|
| Mean Dice | **0.935** |
| Mean IoU | **0.880** |
| Mean Precision | **0.903** |
| Mean Recall | **0.972** |
| Mean Inference Time | **~0.27 s** |

> These metrics are from the evaluated Fold 1 validation set and should not be interpreted as independent test-set or clinical performance.

---

## BreastCareAgent

`BreastCareAgent` is a **bounded, tool-based workflow orchestration agent** designed to coordinate the image-analysis process.

The agent currently performs a **9-step workflow:**

1. **Validate image** — Decode and preprocess the ultrasound image
2. **Decide segmentation** — Check input contract and image sanity conditions
3. **Run segmentation** — Execute the HybridViTGABVSSMUNet model
4. **Check mask usability** — Verify the predicted mask is valid
5. **Extract features** — Calculate quantitative lesion measurements
6. **Quality check** — Perform segmentation sanity checks
7. **Analyze geometry** — Generate geometry analysis
8. **Save outputs** — Persist segmentation artifacts
9. **Generate report** — Produce structured analysis report

Each step records its status and decision metadata in an agent execution trace, providing **workflow-level observability and traceability**.

### Why a bounded agent?

The system intentionally uses deterministic tool-based orchestration rather than an unrestricted LLM-driven workflow.

This provides:

- Reproducible execution
- Explicit decision points
- Controlled tool usage
- Workflow traceability
- Predictable failure handling
- Easier debugging and evaluation

The agent is a **tool-based workflow orchestration layer**, not an autonomous clinical decision-maker.

---

## Lesion Feature Extraction

The predicted segmentation mask is used to calculate quantitative geometric features including:

- Lesion area in pixels
- Lesion area percentage
- Bounding-box coordinates
- Bounding-box width and height
- Aspect ratio
- Perimeter
- Circularity
- Centroid position
- Connected components

Example output:

| Feature | Example |
|---|---:|
| Lesion area | 15,921 pixels |
| Lesion area percentage | 24.29% |
| Bounding-box width | 142 px |
| Bounding-box height | 168 px |
| Aspect ratio | 0.845 |
| Circularity | 0.540 |
| Connected components | 1 |

---

## Segmentation Quality Assessment

The system performs configurable sanity checks on the predicted segmentation mask.

Current checks include conditions such as:

- Invalid mask structure
- Empty foreground
- Very small segmented regions
- Very large segmented regions
- Multiple connected components

The result is exposed separately from the geometric analysis.

Example:

```json
{
  "status": "passed",
  "outcome": "accepted",
  "reason": "Mask passed configured segmentation sanity checks.",
  "foreground_pixels": 15921,
  "flags": []
}
```

These checks are **computational quality signals** and are not clinical assessments.

---

## Geometry Analysis

The geometry-analysis module converts extracted mask features into structured, non-diagnostic descriptions.

Examples include:

- Lesion extent
- Approximate width/height balance
- Geometric compactness
- Configured geometry-based review flags

Geometry-derived measurements should not be interpreted as evidence of malignancy or benignity.

---

## System Architecture

```
                    Breast Ultrasound Image
                              |
                              v
                     React Frontend
                              |
                         POST /analyze
                              |
                              v
                      FastAPI Backend
                              |
                              v
                     BreastCareAgent
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    Input Validation      Segmentation       Quality Checks
                              |
                              v
                    HybridViTGABVSSMUNet
                              |
                              v
                     Binary Mask
                              |
               +--------------+--------------+
               |                             |
               v                             v
       Feature Extraction              Mask Quality
               |                             |
               +--------------+--------------+
                              |
                              v
                      Geometry Analysis
                              |
                              v
                    Structured Report
                              |
                              v
                     Agent Execution Trace
                              |
                              v
                       React Results UI
```

---

## End-to-End Workflow

1. User uploads an ultrasound image through the React frontend
2. FastAPI receives the multipart image upload
3. The agent validates the image and preprocessing contract
4. The agent decides whether segmentation should proceed
5. The segmentation model generates a binary lesion mask
6. The agent checks whether the mask is usable
7. Lesion features are extracted from the mask
8. Segmentation-quality checks are performed
9. Geometry analysis is generated
10. Segmentation artifacts are saved
11. A structured report and agent execution trace are returned
12. The frontend visualizes the results

---

## Backend

The backend is implemented using:

- Python
- FastAPI
- Uvicorn
- PyTorch
- CUDA
- OpenCV
- NumPy
- SciPy

### API Endpoints

**GET /health**

Returns model/backend readiness.

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

**GET /agent/status**

Returns agent readiness.

```json
{
  "status": "ok",
  "agent_ready": true,
  "model_loaded": true,
  "device": "cuda"
}
```

**POST /analyze**

Accepts an image using multipart form upload and executes the complete BreastCareAgent workflow.

```bash
curl.exe -X POST \
  -F "image=@C:\path\to\ultrasound.png" \
  http://127.0.0.1:8000/analyze
```

The response includes:

- Image metadata
- Segmentation results
- Binary mask path
- Overlay path
- Lesion measurements
- Geometry analysis
- Segmentation quality
- Model information
- Inference timing
- Agent execution trace
- Workflow outcome

---

## Frontend

The frontend is implemented using:

- React
- Vite
- JavaScript
- CSS

### Features

- Drag-and-drop image upload
- Image preview
- Interactive analysis flow
- Original image visualization
- Binary segmentation mask
- Segmentation overlay
- Lesion measurements
- Geometry analysis
- Segmentation quality status
- Agent workflow trace
- Assisted-review information
- Limitations and medical disclaimer

---

## Technology Stack

**Machine Learning**
- Python
- PyTorch
- CUDA
- OpenCV
- NumPy
- SciPy
- scikit-learn
- Albumentations

**Model Architecture**
- Vision Transformer components
- Gated Attention Blocks
- Vision State-Space Model components
- U-Net-style segmentation architecture

**Agent Layer**
- Python
- Tool-based orchestration
- Structured schemas
- Workflow state management
- Execution tracing
- Deterministic quality routing

**Backend**
- FastAPI
- Uvicorn
- REST API
- Multipart image uploads
- CORS

**Frontend**
- React
- Vite
- JavaScript
- CSS

---

## Project Structure

```
BreastCare-AI/
│
├── agent/
│   ├── __init__.py
│   ├── breastcare_agent.py
│   ├── schemas.py
│   └── tools.py
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AgentWorkflow.jsx
│   │   │   ├── AssistedReview.jsx
│   │   │   ├── ImageResults.jsx
│   │   │   ├── Measurements.jsx
│   │   │   └── UploadPanel.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
│
├── checkpoints/
├── evaluation_results/
├── inference_results/
├── logs/
│
├── analysis_engine.py
├── benchmarking.py
├── evaluate_fold.py
├── evaluate_single.py
├── fusion.py
├── GAB.py
├── hybridsegnet.py
├── inference.py
├── lesion_features.py
├── lightweight_vit.py
├── loss_criterion.py
├── losses.py
├── SS2D.py
├── summary.py
├── testing.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated files, datasets, virtual environments, and model artifacts are excluded from source control where appropriate.

---

## Setup & Running

### Prerequisites

- Python 3.x
- Node.js and npm
- NVIDIA GPU recommended for accelerated inference
- CUDA-capable NVIDIA drivers

### Backend

```bash
cd source_code

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl.exe http://127.0.0.1:8000/health

curl.exe http://127.0.0.1:8000/agent/status
```

### Frontend

Open a second terminal:

```bash
cd source_code\frontend

npm install
npm run dev
```

Then open the URL shown by Vite, typically:

```
http://localhost:5173
```

---

## Example Results

For the tested ultrasound image 000018.png:

| Metric | Result |
|---|---:|
| Image size | 451 × 440 |
| Predicted lesion area | 24.29% |
| Lesion area | 15,921 pixels |
| Aspect ratio | 0.845 |
| Circularity | 0.540 |
| Connected components | 1 |
| Segmentation quality | Passed |
| Workflow outcome | Accepted |
| Representative inference time | ~1.46 s |

The system generates:

- Binary segmentation mask
- Segmentation overlay
- Quantitative lesion measurements
- Geometry analysis
- Segmentation-quality assessment
- Agent execution trace
- Structured report

---

## Model Evaluation

The evaluated Fold 1 validation set contained 497 images.

| Metric | Result |
|---|---:|
| Mean Dice | 0.935133 |
| Std Dice | 0.036449 |
| Median Dice | 0.944185 |
| Mean IoU | 0.880148 |
| Std IoU | 0.058256 |
| Mean Precision | 0.903313 |
| Mean Recall | 0.972393 |
| Mean Inference Time | 0.270 s |
| Median Inference Time | 0.262 s |

A representative single-image evaluation also produced a Dice score of approximately 0.955, but single-image metrics should not be interpreted as overall model performance.

---

## Important Limitations

**1. Validation rather than independent testing**

The primary reported metrics are from a validation evaluation and do not establish independent test-set performance.

**2. Dataset dependence**

Performance may vary across hospitals, ultrasound machines, acquisition protocols, and patient populations.

**3. Segmentation dependence**

All downstream geometric measurements depend on the quality of the predicted segmentation mask.

**4. Limited input-domain validation**

The current pipeline uses image-level input sanity checks rather than a dedicated medical-image modality classifier. It should therefore be used with the intended breast-ultrasound input domain.

**5. Rule-based quality assessment**

Current segmentation-quality checks are configured computational sanity checks and are not comprehensive image-quality or clinical-quality assessments.

**6. Geometry is not diagnosis**

Geometric measurements cannot independently determine malignancy or benignity.

**7. No clinical validation**

The system has not been established as a regulated clinical medical device.

---

## Medical Disclaimer

**BreastCare-AI is NOT a medical diagnostic system.**

The model predictions, segmentation masks, measurements, quality assessments, and reports are computational outputs intended for research, educational, and engineering purposes.

They must not be used independently to diagnose, treat, or make clinical decisions.

Clinical interpretation requires an appropriately qualified healthcare professional and appropriate clinical context.

---

## What We Achieved

✅ Implemented a hybrid deep-learning segmentation architecture (Vision Transformer + Gated Attention + State-Space Models)

✅ Built GPU-accelerated inference pipeline

✅ Built a bounded 9-step agentic workflow

✅ Added tool-based orchestration layer

✅ Added structured agent execution tracing for observability

✅ Implemented quantitative lesion feature extraction (9 metrics)

✅ Implemented segmentation sanity and quality checks

✅ Implemented geometry analysis module

✅ Built a FastAPI REST backend with 3 endpoints

✅ Built an interactive React/Vite frontend

✅ Integrated frontend, backend, agent, and model end-to-end

✅ Achieved 0.935 mean validation Dice on Fold 1 validation set

✅ Added model/inference timing measurements

✅ Added explicit limitations and medical disclaimer

✅ Documented the complete system architecture and workflow

---

## Future Improvements

- Independent test-set evaluation
- External dataset validation
- Dedicated ultrasound modality classification
- Improved image-quality assessment
- Uncertainty estimation
- Segmentation confidence estimation
- Stronger robustness testing
- Authentication and access control
- Secure deployment and image handling
- Clinical workflow integration
- Longitudinal case management
- Additional model/version tracking

---

## Repository

**GitHub:** [github.com/Miltonbottle/breastcare-ai](https://github.com/Miltonbottle/breastcare-ai)

The repository is intended for research, educational, and engineering purposes.

---
