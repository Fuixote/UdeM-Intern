#ifndef RCVF_H
#define RCVF_H

#include "helper_functions.h"
#include "main_functions.h"

Solution solveKEPusingRCVF(const Instance& inst, string modelCycle, string modelChain, double timeLimit = 3600, bool printRCVFLog = true, bool printGurobiLog = false);
Solution solveCFCFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog = true, bool printGurobiLog = false);
Solution solveCFPIEFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog = true, bool printGurobiLog = false);
Solution solveHCFCFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog = true, bool printGurobiLog = false);
Solution solveHCFPIEFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog = true, bool printGurobiLog = false);
Solution solvePIEFCFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog = true, bool printGurobiLog = false);
Solution solvePIEFPIEFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog = true, bool printGurobiLog = false);

#endif