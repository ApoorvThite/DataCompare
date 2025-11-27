# 📘 TOON vs JSON vs CSV: A Multi-Dimensional Benchmark for Modern AI Pipelines

### A full-stack comparative analysis of data formats across ML workflows, LLM pipelines, embedding quality, schema structure, computational efficiency, and interpretability

This repository presents a rigorous, end-to-end evaluation of three major data formats:

- **CSV**  
- **JSON**  
- **TOON** (Token-Oriented Object Notation)

The project quantifies how data representation influences **machine learning runtime**, **LLM tokenization cost**, **semantic structure**, **embedding quality**, **hallucination rate**, and the overall efficiency of AI workflows.

This is not just a formatting comparison.  
It is a **deep, multi-layer systems evaluation** of the entire AI stack: storage → schema → preprocessing → ML → embeddings → LLM behavior.

---

## 🎯 Motivation

AI pipelines depend on structured and semi-structured data, yet almost no research evaluates how the *choice of format* affects:

- preprocessing overhead  
- training time and inference latency  
- token usage during LLM tasks  
- semantic density  
- embedding quality  
- parsing accuracy  
- hallucination resistance  
- schema richness and path complexity  
- storage and compression efficiency  

This project fills that gap with **uniform experiments** on CSV, JSON, and TOON representations of the same dataset.

---

## 🧩 What This Project Explores

### ✔ ML Runtime Impact
- preprocessing time  
- model training time  
- prediction latency  

### ✔ LLM Tokenization & Parsing
- tokens per record  
- token cost  
- hallucination rate  
- parsing accuracy  

### ✔ Embedding Quality
- silhouette score  
- cluster separability  
- chunk utilization  

### ✔ Schema & Structural Complexity
- nested depth  
- number of paths  
- explicit schema support  
- type annotations  
- constraints  
- flattening needs  

### ✔ Storage-Level Characteristics
- file size  
- gzip-compressed size  
- compression ratio  
- average record size  

Each metric reveals how format choice shapes performance across the entire AI stack.

---

## 📊 Dashboards Summary

The Tableau dashboards included in this project present a consolidated, visual narrative.  
Here is the high-level summary of each one.

---

### **Dashboard 1 — How CSV, JSON, and TOON Differ in Schema & Structure**
This dashboard analyzes:

- nested structure  
- number of key paths  
- schema capabilities  
- schema richness score  
- file size characteristics  

**Key Insight:**  
TOON provides the strongest schema richness and type-level guarantees while staying more compact than JSON.

---

### **Dashboard 2 — Structural & Computational Efficiency**
This dashboard covers:

- processing overhead matrix  
- compression ratio  
- average record size  
- feature engineering difficulty  
- key path complexity  

**Key Insight:**  
CSV is cheapest to process, TOON offers rich structure with lower overhead than JSON, and JSON is the heaviest structurally.

---

### **Dashboard 3 — ML Runtime Efficiency**
This dashboard includes only three critical runtime metrics:

- preprocessing time  
- training time  
- prediction latency  

**Key Insight:**  
CSV consistently trains and infers the fastest.  
TOON slightly beats JSON due to reduced noise and lower token expansion in preprocessing.

---

### **Dashboard 4 — LLM & Embedding Behavior**
This dashboard compares:

- semantic density  
- tokens per record  
- token cost  
- chunk utilization  
- parsing accuracy  
- hallucination rate  
- embedding quality  

**Key Insight:**  
TOON yields the **highest embedding quality**, **lowest hallucination rate**, and **best semantic density**, making it the strongest choice for LLM-intensive workflows.

---

## 📂 Repository Structure
```
data-formats-ml-tableau/
│
├── data_raw/
│ └── global_ai_ml_data_salaries.csv
│
├── data_canonical/
│ └── salaries_canonical.csv
│
├── data_formatted/
│ ├── salaries_flat.csv
│ ├── salaries_nested.json
│ ├── salaries_toon.toon
│
├── metrics/
│ ├── format_structure_metrics.csv
│ ├── llm_density_metrics.csv
│ ├── llm_token_cost.csv
│ ├── schema_metrics.csv
│ ├── ml_metrics.csv
│ ├── format_metrics.csv
│
├── tableau/
│ ├── dashboard_1_schema_structure.twbx
│ ├── dashboard_2_efficiency.twbx
│ ├── dashboard_3_ml_runtime.twbx
│ ├── dashboard_4_llm_metrics.twbx
│
├── scripts/
│ ├── preprocess_formats.py
│ ├── compute_schema_metrics.py
│ ├── compute_ml_metrics.py
│ ├── compute_llm_density.py
│ ├── compute_embedding_quality.py
│
└── README.md
```


---

## ⚙️ Methodology Overview

### **1. Canonical Dataset Preparation**
All formats originate from a single normalized dataset.  
This ensures fair comparison across formats.

### **2. Format Generation**
The same data is represented as:

- flat CSV  
- hierarchical JSON  
- type-annotated TOON  

### **3. Metric Computation**
Multiple scripts generate structured metrics:

- schema-level  
- structural  
- ML runtime  
- LLM embeddings  
- token-level  
- hallucination metrics  
- parsing reliability  

### **4. Tableau Dashboards**
Each dashboard focuses on one aspect of the format comparison and is designed to be presentation-ready.

---

## 🔍 Key Findings

### **CSV**
- fastest for ML training and inference  
- cheapest token cost  
- extremely simple structure  
- lowest semantic density  
- poor embedding quality  

### **JSON**
- highest token cost  
- deepest nested structures  
- highest preprocessing overhead  
- moderate embedding quality  
- prone to hallucination  

### **TOON**
- best embedding quality  
- lowest hallucination rate  
- strong schema guarantees  
- richer type system than JSON  
- lower token cost than JSON  
- strong semantic density  

---

## 🚀 When To Use Each Format

| Use Case | Best Format |
|----------|-------------|
| Fast ML training | CSV |
| LLM-rich workflows | TOON |
| Complex hierarchy | JSON or TOON |
| Minimal token cost | CSV |
| Best embedding quality | TOON |
| Easiest preprocessing | CSV |
| Strong type guarantees | TOON |
| High-fidelity semantic representation | TOON |

---

## 📝 Conclusion

This project provides one of the **most comprehensive comparisons ever done** between CSV, JSON, and TOON across ML, LLM, and schema dimensions.

It shows that:

- CSV is the **computationally fastest**,  
- JSON is the **most verbose and costly**,  
- TOON is the **best semantically and for LLM/embedding performance**.

If your workflow involves LLMs, embeddings, or semantic search, **TOON is the superior format**.  
If your work is traditional ML, CSV remains optimal.

---

## 📬 Contact

For discussions, questions, or collaborations:  
**Apoorv Thite**  
Applied Data Science | AI/ML Engineering  
Penn State University  

---

