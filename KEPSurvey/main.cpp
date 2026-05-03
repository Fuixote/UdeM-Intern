#include "main.h"

int main() {
	//---------------------------USER INPUT---------------------------

	// CHOICE 1: 
	// - Choose which instance to solve: specify separately the path and the name of the file itself
	string pathInstances = "Final instances";	
	string filename = "uk_2019_splitpra_bandxmatch_pra0_pdd_0.05_50_1.json";

	// CHOICE 2:
	// - Decide whether to solve the problem hierarchically (first maximize number of transplants, then maximize weight)
	bool solveHierarchically = false;

	// CHOICE 3: 
	// - Decide whether to solve the weighted or nonweighted variant of the problem
	// - (This choice is ignored when solving the hierarchical variant of the problem)
	bool useWeightedVariant = true;

	// CHOICE 4: 
	// - Decide whether to count the weights to the terminal vertex tau (representing donations to the DDWL or donors becoming bridge donors)
	bool useWToTau = false;

	// CHOICE 5: 
	// - Determine the maximum cycle and chain length
	int K = 3;		// maximum cycle length
	int L = 3;		// maximum chain length
	
	// CHOICE 6:
	// - Determine which cycle and chain models to apply and/or which hybrid models to apply
	// - Add "-CG" to the model name if it should be solved using constraint generation (this is different from the convention in the survey paper)
	
	// These are the models considered in the survey paper:
	vector<string> cycleModels = { "CF-CYCLE", "HCF-CYCLE", "EF-CYCLE-CG", "EEF-CYCLE", "PIEF-CYCLE"};
	vector<string> chainModels = { "CF-CHAIN", "HCF-CHAIN", "EF-CHAIN-EXP-CG", "EF-CHAIN-MTZ", "EEF-CHAIN-EXP-CG", "EEF-CHAIN-MTZ", "PIEF-CHAIN"};
	vector<string> hybridModels = { "EF-HYBRID-CG" };

	//// These are all implemented models:
	//vector<string> cycleModels = { "CF-CYCLE", "HCF-CYCLE", "EF-CYCLE", "EF-CYCLE-CG", "EF-CYCLE-OLD", "EF-CYCLE-OLD-CG", "EF-CYCLE-STRONG", "EF-CYCLE-STRONG-CG", "EEF-CYCLE", "EEF-CYCLE-STRONG", "EEF-CYCLE-OLD", "PIEF-CYCLE", "PIEF-CYCLE-REDUCED"};
	//vector<string> chainModels = { "CF-CHAIN", "HCF-CHAIN", "EF-CHAIN-EXP", "EF-CHAIN-EXP-CG", "EF-CHAIN-CUTSET", "EF-CHAIN-CUTSET-CG", "EF-CHAIN-MTZ", "EEF-CHAIN-EXP", "EEF-CHAIN-EXP-CG", "EEF-CHAIN-CUTSET", "EEF-CHAIN-CUTSET-CG", "EEF-CHAIN-MTZ", "PIEF-CHAIN" };
	//vector<string> hybridModels = { "EF-HYBRID", "EF-HYBRID-CG", "EF-HYBRID-IMPROVED", "EF-HYBRID-IMPROVED-CG" };

	// CHOICE 7:
	// - Determine if reduced cost variable fixing should be used
	// - (RCVF can be used with any model combination, but we have improved code for combinations of CF-CYCLE, HCF-CYCLE or PIEF-CYCLE with CF-CHAIN or PIEF-CHAIN)
	bool useRCVF = false;

	// CHOICE 8:
	// - Determine if the LP relaxation should be computed seperately
	bool solveLP = true;

	// CHOICE 9: 
	// - Determine the time limit (in seconds)
	double timeLimit = 3600;

	// CHOICE 10:
	// - Determine what to print 
	bool printFullInstanceInfo = true;	// determine if the full instance information(i.e., the compatibility graph) should be printed
	bool printGurobiLog = true;			// determine whether the Gurobi optimization log should be printed
	bool printRCVFLog = true;			// determines if a log containing the RCVF progress is printed
	int printSolution = 1;				// determines which solution statistics to print:
										// -2: print a summary in just 1 line, excluding LP relaxation value
										// -1: print a summary in just 1 line, including LP relaxation value
										// 0: print a detailed report
										// 1: print a detailed report, including the best feasible solution

	//---------------------------APPLY ALL MODELS TO THE CHOSEN INSTANCE---------------------------
	
	// Read the instance
	Instance inst = readInstance(pathInstances, filename);

	// If counting weights to tau, generate those weights
	if (useWToTau) { inst.generateRandomWToTau(); }

	// Set all weights to 1 in the unweighted case
	if (not useWeightedVariant and not solveHierarchically) { inst.nonWeightedVariant(useWToTau); }
	
	// Set the maximum cycle and chain length, ensuring that they satisfy trivial upper bounds
	inst.K = min(K, inst.nR); inst.L = min(L, inst.nR + 1);

	// Print the instance
	inst.print(printFullInstanceInfo);

	// Initialize variables to store the best solution and the best LB and UB
	Solution solBest; int LBBest = 0; int UBBest = 1000000;

	// Run all methods
	for (const string& modelCycle : cycleModels) {			// loop over the cycle models
		for (const string& modelChain : chainModels) {		// loop over the chain models

			// Solve the KEP instance for the chosen model combination under the time limit
			Solution sol;
			if (useRCVF) { sol = solveKEPusingRCVF(inst, modelCycle, modelChain, timeLimit, printRCVFLog, printGurobiLog); }
			else { sol = solveKEP(inst, modelCycle, modelChain, "", timeLimit, printGurobiLog, solveLP, useRCVF, printRCVFLog, solveHierarchically); }
			
			// Print solution statistics
			sol.print(printSolution);

			// Check if the solution is correct
			bool isCorrect = sol.check(inst);
			if (not isCorrect) {
				// This line can be used to print errors in the main file, but inside the check function we already print a detailed report in case the solution is incorrect
				cout << "ERROR: the solution found by "<< modelCycle << " + " << modelChain <<  " is infeasible!\n"; 
			}

			// Update the best solution
			if ((sol.opt && (not solBest.opt || sol.timeT < solBest.timeT)) || (sol.feas && not sol.opt && sol.UB - sol.LB < solBest.UB - solBest.LB)) {solBest = sol;}
			if (sol.LB > LBBest) { LBBest = sol.LB; }
			if (sol.UB < UBBest) { UBBest = sol.UB; }
		}
	}

	for (const string& modelHybrid : hybridModels) {	// loop over all hybrid models as well
		Solution sol = solveKEP(inst, "", "", modelHybrid, timeLimit, printGurobiLog, solveLP, useRCVF, printRCVFLog, solveHierarchically);
		sol.print(printSolution);
		bool isCorrect = sol.check(inst);
		if (not isCorrect) {
			cout << "ERROR: the solution found by "<< modelHybrid <<  " is infeasible!\n"; 
		}
		if ((sol.opt && (not solBest.opt || sol.timeT < solBest.timeT)) || (not sol.opt && sol.UB - sol.LB < solBest.UB - solBest.LB)) { solBest = sol; }
		if (sol.LB > LBBest) { LBBest = sol.LB; }
		if (sol.UB < UBBest) { UBBest = sol.UB; }
	}

	// Print the solution statistics of the best model
	 cout << "\nThe best performing model:";
	 solBest.print(printSolution);

	// Check if the best LB is at most the best UB
	if (LBBest > UBBest) { cout << "ERROR: the best lower bound is higher than the best upper bound!\n"; }
}
