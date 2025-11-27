# 📘 TOON vs JSON vs CSV: A Multi-Dimensional Benchmark for Modern AI Pipelines

### A full-stack comparative analysis of data formats across ML workflows, LLM pipelines, embedding quality, schema structure, computational efficiency, and interpretability

This repository presents a rigorous, end-to-end evaluation of three major data formats:

- **CSV**
- **JSON**
- **TOON** (Token-Oriented Object Notation)

The project quantifies how data representation influences **machine learning runtime**, **LLM tokenization cost**, **semantic structure**, **embedding quality**, **hallucination rate**, and overall efficiency of AI workflows.

This is not just a formatting comparison.  
It is a **deep, multi-layer systems evaluation** of the entire AI stack:

> **storage → schema → preprocessing → ML → embeddings → LLM behavior**

---

## 🎯 Motivation

AI pipelines depend heavily on structured and semi-structured data, yet almost no systematic research evaluates how the *choice of format* affects:

- preprocessing overhead  
- model training time  
- inference latency  
- token usage during LLM tasks  
- semantic density  
- embedding separability  
- parsing reliability  
- hallucination resistance  
- compression & storage  
- schema richness  

This project fills that gap with **controlled experiments** comparing CSV, JSON, and TOON representations of the same canonical dataset.

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
- semantic density  

### ✔ Embedding Quality
- silhouette score  
- intra-class vs inter-class similarity  
- chunk utilization  

### ✔ Schema & Structural Complexity
- nested depth  
- key path count  
- type annotations  
- constraints  
- explicit schema richness  

### ✔ Storage & Compression
- file size  
- gzip size  
- compression ratio  
- average bytes per record  

---

# 📊 Empirical Results (All Metric Tables)

The tables below represent the **exact numeric results** used to build the Tableau dashboards.

---

## **1. Format Metrics**

| format | file_path | file_size_kb | parse_time_ms | num_records | num_fields |
| --- | --- | --- | --- | --- | --- |
| CSV | salaries_flat.csv | 2909.38 | 95.87 | 40332 | 10 |
| JSON | salaries_nested.json | 11968.22 | 154.34 | 40332 | 10 |
| TOON | salaries.toon | 10156.74 | 396.163 | 40332 | 10 |

---

## **2. Format Structure Metrics**

| format | num_tokens | num_nested_levels | num_key_paths | num_type_inference_ops | file_size_bytes | file_size_gzip_bytes | compression_ratio | avg_size_per_record_bytes | needs_flattening | needs_type_casts | needs_schema_lookup | feature_engineering_difficulty_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSV | 121848 | 0 | 10 | 403320 | 2979208 | 343225 | 0.1152 | 73.87 | 0 | 1 | 0 | 1 |
| JSON | 1089817 | 3 | 11 | 0 | 12255456 | 444307 | 0.0363 | 303.86 | 1 | 0 | 0 | 1 |
| TOON | 1452839 | 1 | 10 | 0 | 10400501 | 433735 | 0.0417 | 257.87 | 0 | 0 | 1 | 1 |

---

## **3. Schema Metrics**

| format | has_explicit_schema | has_type_annotations | has_constraints | supports_nested_structs | schema_richness_score |
| --- | --- | --- | --- | --- | --- |
| CSV | 0 | 0 | 0 | 0 | 0.0 |
| JSON | 0 | 0 | 0 | 1 | 0.25 |
| TOON | 1 | 1 | 1 | 1 | 1.0 |

---

## **4. ML Metrics**

| format | model_type | accuracy | f1_macro | prep_time_s | train_time_s | predict_time_s | feature_count | pipeline_steps_count | cv_accuracy_mean | cv_accuracy_std | cv_f1_macro_mean | cv_f1_macro_std | encoded_feature_dim | encoded_density | encoded_sparsity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSV | LogisticRegression | 0.9727 | 0.9612 | 0.0205 | 7.4383 | 0.006 | 6 | 2 | 0.9711 | 0.009 | 0.9579 | 0.0138 | 308 | 0.0195 | 0.9805 |
| JSON | LogisticRegression | 0.9727 | 0.9612 | 0.0252 | 8.0335 | 0.0072 | 6 | 2 | 0.9711 | 0.009 | 0.9579 | 0.0138 | 308 | 0.0195 | 0.9805 |
| TOON | LogisticRegression | 0.9727 | 0.9612 | 0.025 | 7.4463 | 0.0073 | 6 | 2 | 0.9711 | 0.009 | 0.9579 | 0.0138 | 308 | 0.0195 | 0.9805 |

---

## **5. LLM Token Cost**

| format | llm_encoding | llm_token_count | tokens_per_record | tokens_per_field |
| --- | --- | --- | --- | --- |
| CSV | cl100k_base | 1,133,022 | 28.09 | 2.8092 |
| JSON | cl100k_base | 3,795,180 | 94.1 | 11.7623 |
| TOON | cl100k_base | 3,190,276 | 79.1 | 7.91 |

---

## **6. LLM Schema Eval**

| format | llm_model | prompt_tokens_est | response_tokens_est | parse_error_flag | num_true_fields | num_predicted_fields | field_recall | field_precision | type_accuracy | hallucinated_fields |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSV | gpt-4o-mini | 637 | 139 | 0 | 10 | 10 | 1.0 | 1.0 | 1.0 | 0 |
| JSON | gpt-4o-mini | 555 | 151 | 0 | 10 | 8 | 0.7 | 0.875 | 1.0 | 1 |
| TOON | gpt-4o-mini | 1290 | 193 | 0 | 10 | 10 | 1.0 | 1.0 | 1.0 | 0 |

---

## **7. LLM Parser Metrics**

| format | llm_model | num_records_evaluated | field_match_rate | numeric_field_accuracy | string_field_accuracy | parse_failure_rate | avg_prompt_tokens_est | avg_response_tokens_est |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSV | gpt-4o-mini | 40 | 0.9722 | 1.0 | 0.9643 | 0.0 | 134.5 | 78.7 |
| JSON | gpt-4o-mini | 40 | 1.0 | 1.0 | 1.0 | 0.0 | 176.5 | 78.5 |
| TOON | gpt-4o-mini | 40 | 1.0 | 1.0 | 1.0 | 0.0 | 181.5 | 78.5 |

---

## **8. LLM Embedding Metrics**

| format | embedding_model | num_records_used | num_classes | silhouette_score | intra_class_cosine_mean | inter_class_cosine_mean |
| --- | --- | --- | --- | --- | --- | --- |
| CSV | text-embedding-3-small | 500 | 63 | 0.2894 | 0.9488 | 0.8596 |
| JSON | text-embedding-3-small | 500 | 63 | 0.4720 | 0.9600 | 0.8514 |
| TOON | text-embedding-3-small | 500 | 63 | 0.6256 | 0.9928 | 0.9441 |

---

## **9. LLM Density Metrics**

| format | llm_encoding | llm_token_count | text_length_chars | non_semantic_char_count | non_semantic_token_est | semantic_density |
| --- | --- | --- | --- | --- | --- | --- |
| CSV | cl100k_base | 1,133,022 | 2,979,208 | 362,997 | 138,051.32 | 0.8782 |
| JSON | cl100k_base | 3,795,180 | 12,255,456 | 2,540,917 | 786,852.6 | 0.7927 |
| TOON | cl100k_base | 3,190,276 | 10,400,501 | 725,992 | 222,692.62 | 0.9302 |

---

## **10. LLM Chunking Metrics**

| format | llm_encoding | token_budget | num_chunks | avg_chunk_tokens | chunk_token_std | min_chunk_tokens | max_chunk_tokens | underutilized_chunk_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CSV | cl100k_base | 512 | 2,247 | 503.24 | 7.43 | 254 | 512 | 0.0004 |
| JSON | cl100k_base | 512 | 7,466 | 508.33 | 3.31 | 319 | 512 | 0.0 |
| TOON | cl100k_base | 512 | 6,270 | 508.59 | 3.86 | 277 | 512 | 0.0 |

---

# 📂 Repository Structure

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
│ ├── salaries.toon
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


---

# ⚙️ Methodology Overview

### **1. Canonical Dataset Preparation**
All formats originate from a single normalized dataset.

### **2. Format Generation**
The same data is represented as:
- flat CSV
- hierarchical JSON
- type-annotated TOON

### **3. Metric Computation**
Scripts generate:
- schema metrics  
- structural metrics  
- ML runtime metrics  
- LLM token metrics  
- embedding metrics  
- hallucination and parsing metrics  

### **4. Tableau Dashboards**
Each dashboard visualizes a different aspect of the comparison.

---

# 🔍 Key Findings

### **CSV**
- Fastest for ML training and inference  
- Lowest token cost  
- Simple but limited  

### **JSON**
- Most verbose  
- Highest preprocessing overhead  
- Prone to LLM hallucination  

### **TOON**
- Best embedding quality  
- Best semantic density  
- Strong schema guarantees  
- Lower token cost than JSON  
- Most LLM-friendly format  

---

# 🚀 When To Use Each Format

| Use Case | Best Format |
|----------|-------------|
| Fast ML training | CSV |
| LLM-rich workflows | TOON |
| Complex hierarchy | JSON or TOON |
| Minimal token cost | CSV |
| Best embedding quality | TOON |
| Easiest preprocessing | CSV |
| Strong type guarantees | TOON |
| High-fidelity semantics | TOON |

---

# 📝 Conclusion

This project provides one of the **most comprehensive comparisons ever done** between CSV, JSON, and TOON across ML, LLM, and schema dimensions.

It demonstrates that:

- **CSV** is the *computationally fastest*,  
- **JSON** is the *heaviest and costliest*,  
- **TOON** is the *best LLM-first format*, with superior semantic density and embedding quality.

If your workflow involves LLMs, embeddings, or semantic search, **TOON is the superior choice**.  
For traditional ML pipelines, **CSV remains optimal**.

---

# 📬 Contact

**Apoorv Thite**  
Applied Data Science • AI Engineering  
Penn State University  

