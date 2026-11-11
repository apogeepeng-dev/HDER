HDER: Hierarchical Diagnostic Evidence Refinement for
Few-Shot Vision-Language Differentiation of Primary Liver
Cancers

Simple Summary 

Hepatocellular carcinoma and intrahepatic cholangiocarcinoma are the two main primary 
liver cancers. They are treated differently, so telling them apart on a routine section matters. 
Whole-slide images make the morphology available to a computer, but labelled slides 
are scarce and most patches carry no diagnostic information. We ask whether the patch 
features themselves can be adapted while the image encoder stays fixed. HDER adds 
two small, purely additive adjustments: a residual adapter that refines each patch, and a 
global context block that summarises the whole slide and writes the summary back to every 
patch. With 8 and 16 labelled slides per class, the two adjustments together outperform the 
multiple-instance and vision–language baselines we compare against, and each of them 
contributes on its own. 

Overview of model

<img width="1150" height="669" alt="image" src="https://github.com/user-attachments/assets/af310522-50fa-4d43-af19-8827c33aefc4" />

1 Code execution reference ViLa MIL (2024 CVPR)

2 CONCH weight configuration reference FOCUS (2025 CVPR) and CONCH

