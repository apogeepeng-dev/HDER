HDER: Hierarchical Diagnostic Evidence Refinement for
Few-Shot Vision-Language Differentiation of Primary Liver
Cancers

Simple Summary 1
Hepatocellular carcinoma and intrahepatic cholangiocarcinoma are the two main primary 2
liver cancers. They are treated differently, so telling them apart on a routine section matters. 3
Whole-slide images make the morphology available to a computer, but labelled slides 4
are scarce and most patches carry no diagnostic information. We ask whether the patch 5
features themselves can be adapted while the image encoder stays fixed. HDER adds 6
two small, purely additive adjustments: a residual adapter that refines each patch, and a 7
global context block that summarises the whole slide and writes the summary back to every 8
patch. With 8 and 16 labelled slides per class, the two adjustments together outperform the 9
multiple-instance and vision–language baselines we compare against, and each of them 10
contributes on its own. 
