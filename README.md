# Distilling Step-by-Step: Research & Analysis

## 📌 Project Overview
This repository contains a comprehensive research review and presentation based on the paper **"Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes."** 

Rather than a codebase, this project serves as a structured academic analysis of the deployment bottlenecks facing Large Language Models (LLMs) and the multi-task distillation mechanism proposed to solve them. It was prepared and presented as part of the CENG 543 Information Retrieval course.

## 📊 Presentation Materials
* **[Presentation.pdf](./docs/Presentation.pdf):** The full slide deck detailing the problem definition, methodology, experimental setup, and critical analysis.
* **[Original Paper](./docs/Distilling_Step_by_Step.pdf):** The reference research paper used for this literature review.

## 🧠 Methodology Breakdown
The research explores shifting the paradigm from viewing LLMs purely as label generators to viewing them as *reasoners*. The core pipeline involves:
1. **Few-Shot Prompting:** Curating unlabeled datasets into prompts containing an Input, a Chain-of-Thought (CoT) Rationale, and a Label.
2. **Rationale Extraction:** Using a massive Teacher LLM (e.g., 540B PaLM) to generate high-quality reasoning steps (rationales) and pseudo-labels for unlabeled data.
3. **Multi-Task Training:** Training a small Student Model (e.g., T5) to simultaneously predict the label and generate the rationale using task prefixes (`[label]` and `[rationale]`).
4. **Efficient Deployment:** Discarding the rationale generation head during inference, leaving a lightweight model that matches LLM performance with zero computational overhead.

## 🏆 Key Findings
* **Extreme Compression:** A 770M parameter distilled T5 model outperformed a 540B PaLM model on the ANLI benchmark, representing a >700x reduction in model size.
* **Data Efficiency:** On the e-SNLI dataset, the distilling method used only 12.5% of the data to outperform standard finetuning models trained on 100% of the data.
* **Unlabeled Data Maximization:** Matched teacher performance using only 12.5% of unlabeled data, whereas standard task distillation required the full dataset.

## 🔍 Critical Analysis
Based on the paper's findings, several limitations and real-world considerations were identified during the review:
* **Prompt Dependency:** The quality of the student model is heavily reliant on carefully designed few-shot prompts to extract good rationales from the teacher.
* **Hallucination Transfer:** If the Teacher LLM hallucinates a flawed rationale, the Student model will learn and internalize that flawed logic.
* **Complexity Limits:** This methodology cannot distill capabilities the Teacher doesn't possess, particularly in complex planning tasks where LLMs still struggle.

## 🛠️ Theoretical Implementation 
If this architecture were to be implemented in a production environment, the proposed system design would be:
* **Infrastructure:** Cloud A100x16 GPU instances for the initial distillation phase.
* **Tech Stack:** PyTorch, Huggingface `transformers` for T5 model initialization, and `DeepSpeed` for memory-efficient multi-GPU training.
* **Pipeline:** A data ingestion script to parse Teacher LLM outputs, formatting them into the required `[label]` and `[rationale]` prefix structures, followed by a multi-task training loop minimizing the weighted sum of both Cross-Entropy losses.

## 📁 Repository Structure
```text
distilling-step-by-step-research/
├── docs/
│   ├── Presentation.pdf # Main slide deck
│   └── Distilling_Step_by_Step.pdf         # Original research paper for reference
└── README.md
