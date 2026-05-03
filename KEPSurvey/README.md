# KEPSurvey
This repository contains the code for all model combinations discussed in the paper "Operational research approaches and mathematical models for kidney exchange: A literature survey and empirical evaluation" by Mathijs Barkel, Rachael Colley, Maxence Delorme, David Manlove and William Pettersson.

All algorithms are coded in C++ and require the commercial solver Gurobi (we used version 10.0.3). The files have the following contents:
- main.h/cpp             - example of front-end code for solving a given instance using any of the ILP models that we consider;
- main_functions.h/cpp   - code containing our main functions;
- RCVF.h/cpp             - code for solving combinations of CF-CYCLE, HCF-CYCLE or PIEF-CYCLE with CF-CHAIN or PIEF-CHAIN using reduced cost variable fixing;
- helper_functions.h/cpp - code containing miscellaneous simple/supportive functions; and
- json.hpp               - package required for reading the JSON instance files.

The file main.cpp is meant to be self-explanatory, one simply needs to make the following ten choices:
1. Choose an instance file containing a compatibility graph(*).
2. Decide whether to solve the problem hierarchically (first maximise the number of transplants, then maximise the weight).
3. Decide whether to solve the unweighted or weighted variant of the instance. (This choice is only relevant when not solving the problem hierarchically)
4. Decide whether to count the weights corresponding to donations to the DDWL or donors becoming bridge donors.
5. Determine the maximum cycle and chain length.
6. Determine which cycle and chain models to apply and/or which hybrid models to apply(**).
7. Determine if reduced cost variable fixing should be used.
8. Determine if the LP relaxation should be solved seperately.
9. Determine the time limit per run (in seconds).
10. Determine what information to print.

(*)The JSON-files for the compatibility graphs used in our experiments are available for download at https://researchdata.gla.ac.uk/1878/. 
That webpage also contains a README file detailing the structure of these instance files. These files, together with choices 2-5, define our instances. 
Our compatibility graphs are generated using the instance generator created by Delorme et al. (2022), which is available at https://wpettersson.github.io/kidney-webapp/#/generator.
More specifically, we used their ``SplitPRA BandXMatch PRA0'' profile, which was shown by these authors to create instances with similar characteristics to those found in historical UKLKSS datasets.
The results of our computational experiments based on this code and these instances can be found on https://www.optimalmatching.com/kep-survey-2025/.

(**)This code also contains some uncompetitive models that were not part of the final computational experiments presented in the paper, but that were mentioned in Appendix A.3 or in Section 4.6). Namely:
- EF-CYCLE-OLD is a variant of EF-CYCLE in that includes constraints (59) instead of constraints (19).
- EF-CYCLE-STRONG is a variant of EF-CYCLE that includes constraints (60) instead of constraints (19).
- EF-CHAIN-CUTSET is a variant of EF-CHAIN-EXP that includes constraints (61) instead of constraints (25).
- EF-HYBRID-IMPROVED is a variant of EF-HYBRID that takes into account special cases of the relationship between K and L.
- EEF-CYCLE-OLD is a variant of EEF-CYCLE that includes constraints (63) and (64) instead of constraints (38).
- EEF-CYCLE-STRONG is a variant of EEF-CYCLE that includes constraints (64) on top of constraints (38).
- EEF-CHAIN-CUTSET is a variant of EEF-CHAIN-EXP that includes constraints (62) instead of constraints (45).
- PIEF-CYCLE-REDUCED is a variant of PIEF-CYCLE (discussed in Section 4.6) that does not include the variables for positions 1 and K.
- Moreover, one can apply constraint generation to models containing an exponential number of constraints by adding the suffix -CG to the model name. In the paper, we only present results for the -CG variants of the tested models.

For questions on the code, please send an e-mail to mathijsbarkel3[at]gmail[dot]com.

References:
- Delorme, M., García, S., Gondzio, J., Kalcsics, J., Manlove, D., Pettersson, W., and Trimble, J. (2022). Improved instance generation for kidney exchange programmes. Computers & Operations Research, 141:105707.
