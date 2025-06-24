
# **DREX-062225-exp**

> The **DREX-062225-exp** (**Document Retrieval and Extraction eXpert**) model is a specialized fine-tuned version of **docscopeOCR-7B-050425-exp**, optimized for **Document Retrieval**, **Content Extraction**, and **Analysis Recognition**. Built on top of the Qwen2.5-VL architecture, this model enhances document comprehension capabilities with focused training on the Opendoc2-Analysis-Recognition dataset for superior document analysis and information extraction tasks.


> [!note]
DREX: Document Retrieval and Extraction eXpert [ experimental ]

# Key Enhancements

* **Advanced Document Retrieval**: Specialized capabilities for locating and retrieving specific information from complex document structures and layouts.

* **Enhanced Content Extraction**: Optimized for extracting structured data, key information, and relevant content from diverse document types including reports, forms, and technical documentation.

* **Superior Analysis Recognition**: Fine-tuned recognition abilities for document analysis tasks, pattern identification, and contextual understanding of document hierarchies.

* **Inherited OCR Excellence**: Maintains all advanced OCR capabilities from the base docscopeOCR model including mathematical LaTeX formatting and multi-language support.

* **Document-Centric Understanding**: Specialized training for understanding document relationships, cross-references, and contextual dependencies within complex document sets.

---

# Markdown (.MD) - Inference

![1.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/MbQ4l2xsMD3kUppqHC1_H.png)

---

![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/k_N9NppbakBo4iJM7LJnR.png)

---

# Demo Inference

https://github.com/user-attachments/assets/4ac2bc88-3a66-434e-86be-c395957bc158

---

https://github.com/user-attachments/assets/9bcfdd1a-1ad6-425f-9626-3d2a9934e67c

---

# Quick Start with Transformers

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/DREX-062225-exp", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("prithivMLmods/DREX-062225-exp")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Extract and analyze the key information from this document."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

---

## Training Details

| Parameter               | Value                                               |
|-------------------------|-----------------------------------------------------|
| **Dataset**             | Opendoc2-Analysis-Recognition                       |
| **Dataset Size**        | 6,910 samples                                       |
| **Base Model**          | docscopeOCR-7B-050425-exp                          |
| **Model Architecture**  | `Qwen2_5_VLForConditionalGeneration`                |
| **Hardware**            | 2 × A40 (19 vCPUs)                                 |
| **Total Disk**          | 280,000 MB                                          |
| **Training Time**       | 3,407 seconds (~0.95 hours)                        |
| **Warmup Steps**        | 250                                                |
| **Precision**           | bfloat16                                            |

> [!note]
> This model builds upon the robust foundation of docscopeOCR-7B-050425-exp with specialized training for document retrieval and extraction tasks.

# Intended Use

This model is specifically designed for:

* **Document Retrieval**: Efficiently locating specific information within large document collections and complex layouts.
* **Content Extraction**: Precise extraction of structured data, tables, forms, and key information from various document types.
* **Analysis Recognition**: Advanced recognition and analysis of document patterns, structures, and contextual relationships.
* **Enterprise Document Processing**: Automated processing of business documents, reports, contracts, and administrative forms.
* **Research Document Analysis**: Academic paper analysis, citation extraction, and research document comprehension.
* **Regulatory Compliance**: Processing of compliance documents, regulatory filings, and standardized reporting formats.

# Limitations

* Inherits computational requirements from the base docscopeOCR model, requiring substantial resources for optimal performance.
* Performance may vary on document types significantly different from the Opendoc2-Analysis-Recognition training dataset.
* May show reduced accuracy on extremely specialized or domain-specific document formats not covered in training.
* Long document processing requires adequate memory allocation and may not be suitable for real-time streaming applications.
* Optimal performance depends on proper visual token configuration and input preprocessing.

## References

- **Base Model**: docscopeOCR-7B-050425-exp
  [https://huggingface.co/prithivMLmods/docscopeOCR-7B-050425-exp](https://huggingface.co/prithivMLmods/docscopeOCR-7B-050425-exp)

- **DocVLM: Make Your VLM an Efficient Reader** 
  [https://arxiv.org/pdf/2412.08746v1](https://arxiv.org/pdf/2412.08746v1)

- **YaRN: Efficient Context Window Extension of Large Language Models**  
  [https://arxiv.org/pdf/2309.00071](https://arxiv.org/pdf/2309.00071)

- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution**  
  [https://arxiv.org/pdf/2409.12191](https://arxiv.org/pdf/2409.12191)

- **Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond**  
  [https://arxiv.org/pdf/2308.12966](https://arxiv.org/pdf/2308.12966)

- **A Comprehensive and Challenging OCR Benchmark for Evaluating Large Multimodal Models in Literacy**
  [https://arxiv.org/pdf/2412.02210](https://arxiv.org/pdf/2412.02210) 
