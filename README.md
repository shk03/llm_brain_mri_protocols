# Evaluating LLM-Generated Brain MRI Protocols
Official implementation by the [HAIM Lab (Human-Centered AI in Medicine)](https://www.neurokopfzentrum.med.tum.de/neuroradiologie/forschung_projekt_haim.html) of the Technical University of Munich (TUM). 

Study Title: "Evaluating Large Language Model-Generated Brain MRI Protocols: Performance of GPT4o, o3-mini, DeepSeek-R1 and Qwen2.5"

Contact: Dr. med. Su Hwan Kim (suhwan.kim@tum.de)

This study evaluated the performance of four large language models (LLMs) in generating granular, sequence-level brain MRI imaging protocols from clinical case descriptions. Accuracy was evaluated against reference MRI protocols established by two board-certified neuroradiologists. 

![image](https://github.com/user-attachments/assets/9c8b8e86-dc73-4b18-800e-a70468a1073c)

In summary, o3-mini demonstrated superior performance, followed by GPT-4o. All four models showed improved accuracy when augmented with external knowledge (see below).

![image](https://github.com/user-attachments/assets/d0dd9660-f152-49e6-9471-fdb97348456b)


## Files Contained in This Repo

External Knowledge
- domain-knowledge-formatted.txt: Explanations and indications for each of the 27 MRI sequences
- standard-protocols-formatted.txt: Local standard brain MRI protocols from the Neuroradiology Department of the Technical University of Munich

Scripts
Python scripts per LLM-condition pair. 
'Base' indicates that the LLM generated brain MRI protocols without external knowledge. 
'Enhanced' indicates that the LLM was provided with the external knowledge above (as part of the context window).
- gpt4o_base.py
- gpt4o_enhanced.py
- o3_mini_base.py
- o3_mini_enhanced.py
- qwen2p5_base.py
- qwen2p5_enhanced.py
- r1_base.py
- r1_enhanced.py


## LLMs Used in This Study 
