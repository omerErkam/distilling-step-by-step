# Distilling Step-by-Step: Outperforming LLMs with Less Data and Smaller Models

## Project Overview
Deploying massive Large Language Models (LLMs) like the 540-billion parameter PaLM or GPT-4 for real-time applications creates severe bottlenecks due to prohibitive VRAM requirements, high latency, and massive compute costs[cite: 3]. Traditional solutions to this problem—standard finetuning and standard task distillation—fail because they require expensive human-annotated data or massive amounts of unlabeled data[cite: 4]. 

This project implements **Distilling Step-by-Step**, a novel machine learning pipeline that extracts Chain-of-Thought (CoT) rationales from LLMs to train much smaller, task-specific student models[cite: 5, 6]. By framing distillation as a multi-task learning problem (predicting both the label and the reasoning rationale), this solution achieves LLM-level intelligence on hardware as accessible as a single standard GPU or CPU, while requiring significantly less training data than standard finetuning[cite: 7, 17].

## Tech Stack & Tools
*   **Teacher LLMs:** PaLM (540B), GPT-NeoX (20B)[cite: 9, 29].
*   **Student Models:** T5 Pretrained Architectures (T5-Base 220M, T5-Large 770M, T5-XXL 11B)[cite: 9, 50].
*   **Frameworks & Libraries:** Huggingface `transformers`, PyTorch[cite: 50].
*   **Hardware / Infrastructure:** Cloud A100x16 GPU instances[cite: 50].
*   **Evaluation Datasets:** e-SNLI, ANLI, Commonsense QA (CQA), SVAMP[cite: 9].

## Methodology
1.  **Data Curation & Few-Shot Prompting:** Unlabeled datasets are curated into Few-Shot Prompts containing triplets: Input, Rationale, and Label[cite: 6].
2.  **Rationale Extraction:** The Teacher LLM (e.g., PaLM 540B) mimics the prompt reasoning to generate high-quality rationales and pseudo-labels for the unlabeled dataset, acting as a rich supervision source rather than a simple label generator[cite: 5, 6].
3.  **Multi-Task Training:** The Student Model (e.g., T5) is trained simultaneously on two tasks using distinct input prefixes[cite: 7]:
    *   `[label] + Input Text` $\rightarrow$ Target: Predicted Label
    *   `[rationale] + Input Text` $\rightarrow$ Target: Explanatory Rationale
4.  **Optimized Deployment:** At test time, the rationale generation head is completely discarded[cite: 7]. The model only executes the label prediction task, ensuring there is zero computational overhead during inference[cite: 7].

## Results & Evaluation
*   **Extreme Model Compression:** A 770M parameter distilled T5 model outperformed a 540B PaLM model on the ANLI benchmark, achieving an over 700x reduction in model size[cite: 17]. On the e-SNLI benchmark, a tiny 220M model surpassed the 540B teacher[cite: 17].
*   **Data Efficiency (Labeled):** Achieved state-of-the-art performance on e-SNLI using only 12.5% of the data, outperforming standard finetuning models trained on 100% of the dataset[cite: 13].
*   **Data Efficiency (Unlabeled):** Matched the teacher's performance on ANLI using only 12.5% of the unlabeled data, whereas standard distillation required 100%[cite: 21].
