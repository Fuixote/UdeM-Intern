#include "RCVF.h"

Solution solveKEPusingRCVF(const Instance& inst, string modelCycle, string modelChain, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function selects the right function to apply RCVF to solve some model combination
	// We have specialised code for CF-CYCLE, HCF-CYCLE or PIEF-CYCLE combined with CF-CHAIN or PIEF-CHAIN
	// Moreover, we have a general code that can be used for any model combination
	Solution sol;
	if (modelCycle == "CF-CYCLE" && modelChain == "CF-CHAIN") { sol = solveCFCFusingRCVF(inst, timeLimit, printRCVFLog, printGurobiLog); }
	else if (modelCycle == "CF-CYCLE" && modelChain == "PIEF-CHAIN") { sol = solveCFPIEFusingRCVF(inst, timeLimit, printRCVFLog, printGurobiLog); }
	else if (modelCycle == "HCF-CYCLE" && modelChain == "CF-CHAIN") { sol = solveHCFCFusingRCVF(inst, timeLimit, printRCVFLog, printGurobiLog); }
	else if (modelCycle == "HCF-CYCLE" && modelChain == "PIEF-CHAIN") { sol = solveHCFPIEFusingRCVF(inst, timeLimit, printRCVFLog, printGurobiLog); }
	else if (modelCycle == "PIEF-CYCLE" && modelChain == "CF-CHAIN") { sol = solvePIEFCFusingRCVF(inst, timeLimit, printRCVFLog, printGurobiLog); }
	else if (modelCycle == "PIEF-CYCLE" && modelChain == "PIEF-CHAIN") { sol = solvePIEFPIEFusingRCVF(inst, timeLimit, printRCVFLog, printGurobiLog); }
	else { sol = solveKEP(inst, modelCycle, modelChain, "", timeLimit, printGurobiLog, false, true, printRCVFLog, false); }
	return sol;
}

Solution solveCFCFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function is used to solve CF-CYCLE + CF-CHAIN using RCVF

	//---------------------------START OF CODE---------------------------

	// Create a solution object
	Solution sol; sol.modelCycle = "CF-CYCLE"; sol.modelChain = "CF-CHAIN"; sol.RCVFUsed = true;
	sol.modelFull = sol.modelCycle + " + " + sol.modelChain + " (RCVF)";

	// Start the timer
	double start = getCPUTime();

	// Decide whether to use the early termination strategy
	int earlyTermination = 1;

	//---------------------------STEP 1: CREATE A MODEL FOR THE LP RELAXATION---------------------------

	// Create an environment and a model
	GRBEnv env = GRBEnv();              							// create an environment
	removeLine();                       							// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);									// create a new model

	// Declare general variables and linear expressions
	vector<GRBVar> x, y;											// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;										// objective value
	vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

	// Declare model-specific variables and linear expressions
	// - (none)

	// Declare other model-specific objects
	vector<vector<vector<int>>> cycles;								// all cycles of size at most K 
	vector<vector<vector<int>>> chains;								// all chains of size at most L

	// Create other model-specific objects
	cycles = findCycles(inst);										// find the set of all cycles of length at most K
	chains = findChains(inst);										// find the set of all chains of length at most L

	// For the cycle model: create (continuous) variables and fill up linear expressions
	if (inst.K > 0) {
		x.resize(cycles.size());												// declare a variable for every cycle
		// for c in C: x_c = 1 iff cycle c is used
		for (int c = 0; c < cycles.size(); c++) {								// loop over all cycles
			x[c] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "x_" + to_string(c));	// add a corresponding variables
			totalWeight += cycles[c][1][0] * x[c];								// add the weight to the objective
			for (const int& r : cycles[c][0]) {									// loop over all recipients in the current cycle
				xIn[r] += x[c];													// update the in-flow
			}
		}
	}

	// For the chain model: create (continuous) variables and fill up linear expressions
	if (inst.L > 0) {
		y.resize(chains.size());												// declare a variable for every chain
		// for c in C: y_c = 1 iff chain c is used
		for (int c = 0; c < chains.size(); c++) {								// loop over all chains
			y[c] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "y_" + to_string(c));	// add a corresponding variable
			totalWeight += chains[c][1][0] * y[c];								// add the weight to the objective
			yOut[chains[c][0].front()] += y[c];									// update the out-flow of the NDD
			for (int i = 1; i < chains[c][0].size() - 1; i++) {					// loop over all intermediate recipients in the current chain
				yOut[chains[c][0][i]] += y[c];									// update the out-flow
				yIn[chains[c][0][i]] += y[c];									// update the in-flow
			}
			yIn[chains[c][0].back()] += y[c];									// update the in-flow of the final recipient
		}

		// add arc weights of arcs to tau													// if chains are allowed
		for (int d = 0; d < inst.nR; d++) {									// loop over all RDPs
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);				// add the weight of the corresponding final arc to the objective
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {					// loop over all NDDs
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);					// add the weight of the corresponding final arc to the objective
		}
	}

	model.update();

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	model.setObjective(totalWeight, GRB_MAXIMIZE);

	//  Constraint for every model 1: every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) {										// loop over NDDs
		if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) {														// loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
	}

	// For the cycle model: add the model-specific constraints
	// - (none)

	// For the chain model: add the model-specific constraints
	// - (none)

	model.update();

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, 2);										// use the barrier method to solve the LP relaxation 
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; }			// return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);			// update the remaining time limit
	model.getEnv().set(GRB_IntParam_Crossover, 0);									// crossover can be turned off

	//---------------------------STEP 2: SOLVE THE LP RELAXATION AND FIND ITS VALUE AND THE REDUCED COSTS---------------------------

	// Solve the LP relaxation model
	if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
	if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving initial LP...\n"; }
	model.optimize();

	// Find the LP relaxation value
	double valLP = model.get(GRB_DoubleAttr_ObjVal);
	sol.LPrel = valLP;
	if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

	// Save the model size
	sol.nVar = model.get(GRB_IntAttr_NumVars);			// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);	// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);			// number of non-zero coefficients

	// Terminate if the time limit has already been exceeded after solving the LP relaxation and simply return an empty solution
	sol.timeLP = getCPUTime() - start;
	if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }

	// Find the reduced costs of the variables
	int numVarsCycles = cycles.size(), numVarsChains = chains.size();
	vector<double> RCCycles(numVarsCycles, 0), RCChains(numVarsChains, 0);
	GRBVar* vars = model.getVars();
	for (int i = 0; i < numVarsCycles; i++) {
		if (vars[i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCCycles[i] = vars[i].get(GRB_DoubleAttr_RC);
		}
	}
	for (int i = 0; i < numVarsChains; i++) {
		if (vars[numVarsCycles + i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCChains[i] = vars[numVarsCycles + i].get(GRB_DoubleAttr_RC);
		}
	}

	// Clear up the memory
	model.reset(1);
	model.update();

	//---------------------------STEP 3: INITIALIZE THE REST OF THE RCVF PROCEDURE---------------------------

	// Set the initial target
	int T = floor(valLP + EPSILON);

	// Some other declarations before while-loop
	int xval, yval;														// the value of the currently considered x- or y-variable
	sol.RCVFNIter = 0;													// the number of RCVF iterations
	sol.RCVFNVarDeact = sol.nVar + 1;									// the number of variables that are deactivated (initialized to impossibly high value)
	int nVarDeactPrev;													// the number of variables that were deactivated in the previous iteration
	vector<int> bestSolCycles, bestSolChains;							// the currently best solution for the cycles and chains

	// Start the main loop of the RCVF procedure
	while (true) {									// the loop will be terminated by any of the "break" statements
		
		//---------------------------STEP 4: CREATE A REDUCED ILP MODEL---------------------------	

		sol.RCVFNIter++;												// increment the number of RCVF iterations by 1
		int LBRedILP = 0, UBRedILP = 1000000;							// the LB and UB of the current reduced ILP model

		// Create a new environment and a new model (these are only in scope inside of the while-loop)
		GRBEnv env = GRBEnv();              							// create an environment
		removeLine();                       							// remove Gurobi message
		if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
		GRBModel model = GRBModel(env);									// create a new model

		// Re-declare general variables and linear expressions (temporary scope)
		vector<GRBVar> x, y;											// decision variables (interpretation different per model)
		GRBLinExpr totalWeight = 0;										// objective value
		vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
		vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
		vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
		vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

		// Redeclare model-specific variables and linear expressions
		// - (none)

		// Keep track of which variables are activated
		vector<bool> isActiveCycles(numVarsCycles, false);
		vector<bool> isActiveChains(numVarsChains, false);

		// For the cycle model: create binary variables (but only when the reduced cost is high enough) and fill up linear expressions
		if (inst.K > 0) {
			x.resize(cycles.size());																// declare a variable for every cycle
			// for c in C: x_c = 1 iff cycle c is used
			for (int c = 0; c < cycles.size(); c++) {												// loop over all cycles
				if (RCCycles[c] > T - valLP - EPSILON) {											// if the reduced cost is high enough
					isActiveCycles[c] = true;														// activate the variable
					x[c] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(c));					// create the variable
					if (bestSolCycles.size() > 0) { x[c].set(GRB_DoubleAttr_Start, bestSolCycles[c]); }	// set a warm start
					totalWeight += cycles[c][1][0] * x[c];											// add the weight to the objective value
					for (const int& r : cycles[c][0]) {												// loop over all recipients in the current cycle
						xIn[r] += x[c];																// update the in-flow of the recipient
					}
				}
			}
		}

		// For the chain model: create binary variables (but only when the RC is high enough) and fill up linear expressions
		if (inst.L > 0) {
			y.resize(chains.size());																// declare a variable for every chain
			// for c in C: y_c = 1 iff chain c is used
			for (int c = 0; c < chains.size(); c++) {												// loop over all chains
				if (RCChains[c] > T - valLP - EPSILON) {											// if the reduced cost is high enough
					isActiveChains[c] = true;														// activate the variable
					y[c] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(c));					// create the variable
					if (bestSolChains.size() > 0) { y[c].set(GRB_DoubleAttr_Start, bestSolChains[c]); }	// set a warm start
					totalWeight += chains[c][1][0] * y[c];											// add the weight to the objective value
					yOut[chains[c][0].front()] += y[c];												// update the out-flow of the NDD
					for (int i = 1; i < chains[c][0].size() - 1; i++) {								// loop over all intermediate RDPs in the current chain
						yOut[chains[c][0][i]] += y[c];												// update their out-flow
						yIn[chains[c][0][i]] += y[c];												// update their in-flow
					}
					yIn[chains[c][0].back()] += y[c];												// update the in-flow of the final RDP
				}
			}
			// add arc weights of arcs to tau
			for (int d = 0; d < inst.nR; d++) {							// loop over all RDPs
				totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);		// add the weight of the corresponding final arc to the objective
			}
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) {			// loop over all NDDs
				totalWeight += inst.wToTau[d] * (1 - yOut[d]);			// add the weight of the corresponding final arc to the objective
			}
		}

		model.update();

		// Skip the rest of the iteration if no additional variables were reactivated
		nVarDeactPrev = sol.RCVFNVarDeact;									// update the number of variables that were deactivated in the prevvious iteration
		sol.RCVFNVarDeact = sol.nVar - model.get(GRB_IntAttr_NumVars);		// find the current number of deactivated variables
		if (sol.RCVFNVarDeact == nVarDeactPrev and earlyTermination == 0) {	// if no additional variables are activated
			if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
			sol.RCVFNSkips++;												// increment the number of skips
			sol.RCVFNIter--;												// undo the increment of the number of (proper) iterations (in which an ILP model is solved)
			T--;															// reduce the target by 1
			if (sol.LB == T) {												// if the best solution reaches the new target, it is optimal
				if (printRCVFLog) { cout << "Target T = " << T << " had already been reached--> solution is optimal.\n"; }
				break;
			}
		}
		else {

			// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// Change a setting so that we terminate once the UB of the current reduced ILP drops below T (using some tolerance)
			if (earlyTermination == 1) { model.set(GRB_DoubleParam_BestBdStop, T - 0.01); }
			else if (earlyTermination == 2) { model.set(GRB_DoubleParam_Cutoff, T - 0.01); }

			//  Constraint for every model 1: every NDD can donate at most once
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
				if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
			}

			// Constraint for every model 2: every RDP can be donated to at most once
			// - note that these are the only contraints linking the cycle and chain component of the model
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
			}

			// For the cycle model: add the model-specific constraints
			// - (none)

			// For the chain model: add the model-specific constraints
			// - (none)

			model.update();

			// Change some settings
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_IntParam_Method, 2);														// use the barrier method to solve the LP relaxation
			if (timeLimit - getCPUTime() + start < 0) { sol.timeT = timeLimit; sol.UB = T;  return sol; }	// return an empty solution if the time limit has already been exceeded in pre-processing
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - getCPUTime() + start);				// update the remaining time limit
			// model.getEnv().set(GRB_DoubleParam_BestObjStop, T);												// we can terminate early if a solution with value T is found
			// don't turn off crossover here


			//---------------------------STEP 5: SOLVE THE REDUCED ILP MODEL---------------------------	

			if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving ILP model, iter = " << sol.RCVFNIter << ", time = " << getCPUTime() - start << "s, target value T = " << T << "\n"; }
			if (printRCVFLog) { cout << "Number of variables removed = " << sol.RCVFNVarDeact << " (out of " << sol.nVar << " originally), " << model.get(GRB_IntAttr_NumVars) << " variables remain\n"; }

			model.optimize();

			// Find the best upper bound
			UBRedILP = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (UBRedILP < 0) { UBRedILP = 1000000; }						// if no upper bound was found yet, set it to 1000000

			// If the model is infeasible
			if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
				if (earlyTermination != 0) {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
					T--;							// decrement the target by 1
					continue;
				}
				else {
					sol.feas = false;											// indicate that the instance is infeasible
					break;
				}
			}

			// If a solution has been found within the time limit
			else if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.feas = true;												// indicate that the instance is feasible
				LBRedILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// find the best lower bound

				if (printRCVFLog) { cout << "Value reduced ILP = " << LBRedILP << "\n"; }

				// If the solution is better than the current best one, update the solution (incl. the warm start)
				if (LBRedILP >= sol.LB) {

					// Set the new best LB
					sol.LB = LBRedILP;

					// Reset the vectors containing the best solution found so far
					bestSolCycles.clear(); bestSolCycles.resize(numVarsCycles, 0);
					bestSolChains.clear(); bestSolChains.resize(numVarsChains, 0);

					// Decompose the solution into the selected cycles and chains
					sol.cyclesAndChains.clear();
					sol.cyclesAndChains.resize(2);

					// First consider the cycles
					if (inst.K > 0) {
						for (int c = 0; c < cycles.size(); c++) {					// loop over all cycles
							if (isActiveCycles[c]) {								// if the variable was activated
								xval = ceil(x[c].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
								if (xval == 1) {									// if the cycle is selected
									sol.cyclesAndChains[0].push_back(cycles[c]);	// add the cycle to the solution
									bestSolCycles[c] = 1;							// save the solution
								}
							}
						}
					}

					// Second consider the chains
					if (inst.L > 0) {
						for (int c = 0; c < chains.size(); c++) {												// loop over all chains
							if (isActiveChains[c]) {															// if the variable was activated
								yval = ceil(y[c].get(GRB_DoubleAttr_X) - EPSILON);								// find variable value
								if (yval == 1) {																// if the chain is selected
									sol.cyclesAndChains[1].push_back(chains[c]);								// add the chain to the solution
									sol.cyclesAndChains[1].back()[1][0] += inst.wToTau[chains[c][0].back()];	// also add the weight of the final arc
									bestSolChains[c] = 1;														// save the solution
								}
							}
						}
						// Also add trivial n-tau chains for unused NDDs n
						for (int d = inst.nR; d < inst.nR + inst.nN; d++) {						// loop over all NDDs
							if (ceil(yOut[d].getValue() - EPSILON) == 0) {						// if its out-flow is 0
								sol.cyclesAndChains[1].push_back({ {d}, {inst.wToTau[d]} });	// add the corresponding trivial chain
							}
						}
					}
				}
			}

			//---------------------------STEP 6: CONCLUDE ABOUT THE NEXT STEP BASED ON THE SOLUTION---------------------------	

			// If the reduced ILP was solved to optimality
			if (LBRedILP == UBRedILP) {

				// If the target has been reached: the solution is optimal and we may terminate
				if (LBRedILP >= T) {
					if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
					break;
				}

				// If no variables were deactivated: the solution is optimal and we may terminate
				else if (sol.RCVFNVarDeact == 0) {
					if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
					T = sol.LB;				// update the best upper bound
					break;
				}

				// If the target was not reached, but a feasible solution with value T - 1 was found: the solution is also optimal and we may terminate
				else if (LBRedILP >= T - 1) {
					sol.RCVFNSkips++;				// increment the number of skips
					T--;
					if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T << " was reached --> solution is optimal.\n"; }
					break;
				}

				// Otherwise: decrease T by 1 and try again
				else {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << "\n"; }
					T--;							// decrement the target by 1
				}
			}

			// If we terminated early because the UB of the reduced ILP dropped below the target 
			else if (model.get(GRB_IntAttr_Status) == GRB_USER_OBJ_LIMIT or model.get(GRB_IntAttr_Status) == GRB_CUTOFF) {
				if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
				T--;							// decrement the target by 1
			}


			else { // If we ran out of time
				cout << "Time limit exceeded --> return the best solution found so far\n";
				break;
			}
		}
	}

	// Save the final upper bound
	sol.UB = T;

	// Set the optimality attribute
	if (sol.LB == sol.UB) {sol.opt = true;}

	// Save the total time
	sol.timeT = getCPUTime() - start;

	// Return the solution
	return sol;
}

Solution solveCFPIEFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function is used to solve CF-CYCLE + PIEF-CHAIN using RCVF

	//---------------------------START OF CODE---------------------------

	// Create a solution object
	Solution sol; sol.modelCycle = "CF-CYCLE"; sol.modelChain = "PIEF-CHAIN"; sol.RCVFUsed = true;
	sol.modelFull = sol.modelCycle + " + " + sol.modelChain + " (RCVF)";

	// Start the timer
	double start = getCPUTime();

	// Decide whether to use the early termination strategy
	int earlyTermination = 1;

	//---------------------------STEP 1: CREATE A MODEL FOR THE LP RELAXATION---------------------------

	// Create an environment and a model
	GRBEnv env = GRBEnv();              							// create an environment
	removeLine();                       							// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);									// create a new model

	// Declare general variables and linear expressions
	vector<GRBVar> x, y;											// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;										// objective value
	vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

	// Declare model-specific variables and linear expressions
	vector<vector<GRBVar>> yL;									// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
	vector<vector<GRBLinExpr>> yOutL, yInL;						// number of transplants made by each donor/recieved by each recipient at every position

	// Declare other model-specific objects
	vector<vector<vector<int>>> cycles;								// all cycles of size at most K 
	vector<Subgraph> subgraphPIEFCHAIN;								// the reduced arc sets for each arc position 1,...,L-1

	// Create other model-specific objects
	cycles = findCycles(inst);										// find the set of all cycles of length at most K
	subgraphPIEFCHAIN = createSubgraphPIEFCHAIN(inst);				// create the position-indexed graph

	// For the cycle model: create (continuous) variables and fill up linear expressions
	if (inst.K > 0) {
		x.resize(cycles.size());												// declare a variable for every cycle
		// for c in C: x_c = 1 iff cycle c is used
		for (int c = 0; c < cycles.size(); c++) {								// loop over all cycles
			x[c] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "x_" + to_string(c));	// add a corresponding variables
			totalWeight += cycles[c][1][0] * x[c];								// add the weight to the objective
			for (const int& r : cycles[c][0]) {									// loop over all recipients in the current cycle
				xIn[r] += x[c];													// update the in-flow
			}
		}
	}

	// For the chain model: create (continuous) variables and fill up linear expressions
	int numVarsChains = 0;
	if (inst.L > 0) {
		yL.resize(inst.L); // seperate set of variables for all positions
		int d, r, w;
		yOutL.resize(inst.nR + inst.nN, vector<GRBLinExpr>(inst.L, 0));
		yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
		// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
		for (int k = 1; k <= inst.L - 1; k++) {
			yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
			for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
				yL[k][a] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "y_" + to_string(a));
				numVarsChains++;
				d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
				totalWeight += w * yL[k][a];
				yOut[d] += yL[k][a];
				yOutL[d][k] += yL[k][a];
				yIn[r] += yL[k][a];
				yInL[r][k] += yL[k][a];
			}
		}

		// add arc weights of arcs to tau													// if chains are allowed
		for (int d = 0; d < inst.nR; d++) {									// loop over all RDPs
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);				// add the weight of the corresponding final arc to the objective
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {					// loop over all NDDs
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);					// add the weight of the corresponding final arc to the objective
		}
	}

	model.update();

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	model.setObjective(totalWeight, GRB_MAXIMIZE);

	//  Constraint for every model 1: every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) {										// loop over NDDs
		if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) {														// loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
	}

	// For the cycle model: add the model-specific constraints
	// - (none)

	// For the chain model: add the model-specific constraints
	if (inst.L > 0) {
		// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
		for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
			for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
				if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
			}
		}
	}

	model.update();

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, 2);										// use the barrier method to solve the LP relaxation 
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; }			// return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);			// update the remaining time limit
	model.getEnv().set(GRB_IntParam_Crossover, 0);									// crossover can be turned off

	//---------------------------STEP 2: SOLVE THE LP RELAXATION AND FIND ITS VALUE AND THE REDUCED COSTS---------------------------

	// Solve the LP relaxation model
	if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
	if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving initial LP...\n"; }
	model.optimize();

	// Find the LP relaxation value
	double valLP = model.get(GRB_DoubleAttr_ObjVal);
	sol.LPrel = valLP;
	if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

	// Save the model size
	sol.nVar = model.get(GRB_IntAttr_NumVars);			// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);	// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);			// number of non-zero coefficients

	// Terminate if the time limit has already been exceeded after solving the LP relaxation and simply return an empty solution
	sol.timeLP = getCPUTime() - start;
	if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }

	// Find the reduced costs of the variables
	int numVarsCycles = cycles.size();
	vector<double> RCCycles(numVarsCycles, 0), RCChains(numVarsChains, 0);
	GRBVar* vars = model.getVars();
	for (int i = 0; i < numVarsCycles; i++) {
		if (vars[i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCCycles[i] = vars[i].get(GRB_DoubleAttr_RC);
		}
	}
	for (int i = 0; i < numVarsChains; i++) {
		if (vars[numVarsCycles + i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCChains[i] = vars[numVarsCycles + i].get(GRB_DoubleAttr_RC);
		}
	}

	// Clear up the memory
	model.reset(1);
	model.update();

	//---------------------------STEP 3: INITIALIZE THE REST OF THE RCVF PROCEDURE---------------------------

	// Set the initial target
	int T = floor(valLP + EPSILON);

	// Some other declarations before while-loop
	int xval, yval;														// the value of the currently considered x- or y-variable
	sol.RCVFNIter = 0;													// the number of RCVF iterations
	sol.RCVFNVarDeact = sol.nVar + 1;									// the number of variables that are deactivated (initialized to impossibly high value)
	int nVarDeactPrev;													// the number of variables that were deactivated in the previous iteration
	vector<int> bestSolCycles, bestSolChains;							// the currently best solution for the cycles and chains

	// Start the main loop of the RCVF procedure
	while (true) {									// the loop will be terminated by any of the "break" statements

		//---------------------------STEP 4: CREATE A REDUCED ILP MODEL---------------------------	

		sol.RCVFNIter++;												// increment the number of RCVF iterations by 1
		int LBRedILP = 0, UBRedILP = 1000000;							// the LB and UB of the current reduced ILP model

		// Create a new environment and a new model (these are only in scope inside of the while-loop)
		GRBEnv env = GRBEnv();              							// create an environment
		removeLine();                       							// remove Gurobi message
		if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
		GRBModel model = GRBModel(env);									// create a new model

		// Re-declare general variables and linear expressions (temporary scope)
		vector<GRBVar> x, y;											// decision variables (interpretation different per model)
		GRBLinExpr totalWeight = 0;										// objective value
		vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
		vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
		vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
		vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

		// Redeclare model-specific variables and linear expressions
		vector<vector<GRBVar>> yL;										// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
		vector<vector<GRBLinExpr>> yOutL, yInL;							// number of transplants made by each donor/recieved by each recipient at every position

		// Keep track of which variables are activated
		vector<bool> isActiveCycles(numVarsCycles, false);
		vector<bool> isActiveChains(numVarsChains, false);

		// For the cycle model: create binary variables (but only when the reduced cost is high enough) and fill up linear expressions
		if (inst.K > 0) {
			x.resize(cycles.size());																// declare a variable for every cycle
			// for c in C: x_c = 1 iff cycle c is used
			for (int c = 0; c < cycles.size(); c++) {												// loop over all cycles
				if (RCCycles[c] > T - valLP - EPSILON) {											// if the reduced cost is high enough
					isActiveCycles[c] = true;														// activate the variable
					x[c] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(c));					// create the variable
					if (bestSolCycles.size() > 0) { x[c].set(GRB_DoubleAttr_Start, bestSolCycles[c]); } // set a warm start
					totalWeight += cycles[c][1][0] * x[c];											// add the weight to the objective value
					for (const int& r : cycles[c][0]) {												// loop over all recipients in the current cycle
						xIn[r] += x[c];																// update the in-flow of the recipient
					}
				}
			}
		}

		// For the chain model: create binary variables (but only when the RC is high enough) and fill up linear expressions
		if (inst.L > 0) {
			yL.resize(inst.L); // seperate set of variables for all positions
			int d, r, w;
			yOutL.resize(inst.nR + inst.nN, vector<GRBLinExpr>(inst.L, 0));
			yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
			int varIdx = -1;
			// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
			for (int k = 1; k <= inst.L - 1; k++) {
				yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
				for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
					varIdx++;
					if (RCChains[varIdx] > T - valLP - EPSILON) {
						yL[k][a] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(a));
						isActiveChains[varIdx] = true;
						if (bestSolChains.size() > 0) { yL[k][a].set(GRB_DoubleAttr_Start, bestSolChains[varIdx]); } // set a warm start
						d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
						totalWeight += w * yL[k][a];
						yOut[d] += yL[k][a];
						yOutL[d][k] += yL[k][a];
						yIn[r] += yL[k][a];
						yInL[r][k] += yL[k][a];
					}
				}
			}
			// add arc weights of arcs to tau
			for (int d = 0; d < inst.nR; d++) {							// loop over all RDPs
				totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);		// add the weight of the corresponding final arc to the objective
			}
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) {			// loop over all NDDs
				totalWeight += inst.wToTau[d] * (1 - yOut[d]);			// add the weight of the corresponding final arc to the objective
			}
		}

		model.update();

		// Skip the rest of the iteration if no additional variables were reactivated
		nVarDeactPrev = sol.RCVFNVarDeact;									// update the number of variables that were deactivated in the prevvious iteration
		sol.RCVFNVarDeact = sol.nVar - model.get(GRB_IntAttr_NumVars);		// find the current number of deactivated variables
		if (sol.RCVFNVarDeact == nVarDeactPrev and earlyTermination == 0) {	// if no additional variables are activated
			if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
			sol.RCVFNSkips++;												// increment the number of skips
			sol.RCVFNIter--;												// undo the increment of the number of (proper) iterations (in which an ILP model is solved)
			T--;															// reduce the target by 1
			if (sol.LB == T) {												// if the best solution reaches the new target, it is optimal
				if (printRCVFLog) { cout << "Target T = " << T << " had already been reached--> solution is optimal.\n"; }
				break;
			}
		}
		else {

			// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// Change a setting so that we terminate once the UB of the current reduced ILP drops below T (using some tolerance)
			if (earlyTermination == 1) { model.set(GRB_DoubleParam_BestBdStop, T - 0.01); }
			else if (earlyTermination == 2) { model.set(GRB_DoubleParam_Cutoff, T - 0.01); }


			//  Constraint for every model 1: every NDD can donate at most once
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
				if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
			}

			// Constraint for every model 2: every RDP can be donated to at most once
			// - note that these are the only contraints linking the cycle and chain component of the model
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
			}

			// For the cycle model: add the model-specific constraints
			// - (none)

			// For the chain model: add the model-specific constraints
			if (inst.L > 0) {
				// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
				for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
					for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
						if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
					}
				}
			}

			model.update();

			// Change some settings
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_IntParam_Method, 2);														// use the barrier method to solve the LP relaxation
			if (timeLimit - getCPUTime() + start < 0) { sol.timeT = timeLimit; sol.UB = T;  return sol; }	// return an empty solution if the time limit has already been exceeded in pre-processing
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - getCPUTime() + start);				// update the remaining time limit
			// model.getEnv().set(GRB_DoubleParam_BestObjStop, T);												// we can terminate early if a solution with value T is found
			// don't turn off crossover here


			//---------------------------STEP 5: SOLVE THE REDUCED ILP MODEL---------------------------	

			if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving ILP model, iter = " << sol.RCVFNIter << ", time = " << getCPUTime()-start << "s, target value T = " << T << "\n"; }
			if (printRCVFLog) { cout << "Number of variables removed = " << sol.RCVFNVarDeact << " (out of " << sol.nVar << " originally), " << model.get(GRB_IntAttr_NumVars) << " variables remain\n"; }

			model.optimize();

			// Find the best upper bound
			UBRedILP = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (UBRedILP < 0) { UBRedILP = 1000000; }						// if no upper bound was found yet, set it to 1000000

			// If the model is infeasible
			if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
				if (earlyTermination != 0) {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
					T--;							// decrement the target by 1
					continue;
				}
				else {
					sol.feas = false;											// indicate that the instance is infeasible
					break;
				}
			}

			// If a solution has been found within the time limit
			else if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.feas = true;												// indicate that the instance is feasible
				LBRedILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// find the best lower bound
				
				if (printRCVFLog) { cout << "Value reduced ILP = " << LBRedILP << "\n"; }

				// If the solution is better than the current best one, update the solution (incl. the warm start)
				if (LBRedILP >= sol.LB) {

					// Set the new best LB
					sol.LB = LBRedILP;

					// Reset the vectors containing the best solution found so far
					bestSolCycles.clear(); bestSolCycles.resize(numVarsCycles, 0);
					bestSolChains.clear(); bestSolChains.resize(numVarsChains, 0);

					// Decompose the solution into the selected cycles and chains
					sol.cyclesAndChains.clear();
					sol.cyclesAndChains.resize(2);

					// First consider the cycles
					if (inst.K > 0) {
						for (int c = 0; c < cycles.size(); c++) {					// loop over all cycles
							if (isActiveCycles[c]) {								// if the variable was activated
								xval = ceil(x[c].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
								if (xval == 1) {									// if the cycle is selected
									sol.cyclesAndChains[0].push_back(cycles[c]);	// add the cycle to the solution
									bestSolCycles[c] = 1;							// save the solution
								}
							}
						}
					}

					// Second consider the chains
					if (inst.L > 0) {
						vector<vector<int>> donationPerDonor(inst.nR + inst.nN);		// array containing [r, w_dr] for each donor d that donates
						int d, r, w;
						int varIdx = -1;
						for (int k = 1; k <= inst.L - 1; k++) {								// loop over all arc positions
							for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {	// loop over all arcs
								varIdx++;
								if (isActiveChains[varIdx]) {
									yval = ceil(yL[k][a].get(GRB_DoubleAttr_X) - EPSILON);		// find variable value
									if (yval == 1) {											// if the arc is selected
										d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2]; // decompose the arc
										donationPerDonor[d] = { r, w };							// add the arc to the list of donations
										bestSolChains[varIdx] = 1;								// save the solution
									}
								}
							}
						}
						// Also find the donations to tau
						if (inst.L > 1) {
							for (int d = 0; d < inst.nR; d++) {
								if (ceil(yIn[d].getValue() - EPSILON) - ceil(yOut[d].getValue() - EPSILON) == 1) {
									donationPerDonor[d] = { -1, inst.wToTau[d] };
								}
							}
						}
						for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
							if (ceil(yOut[d].getValue() - EPSILON) == 0) {
								donationPerDonor[d] = { -1, inst.wToTau[d] };
							}
						}
						// Find the chains
						vector<vector<int>> chain;					// array containing the current chain and its weight
						for (int s = inst.nR; s < inst.nR + inst.nN; s++) { // loop over all non-directed donors
							if (donationPerDonor[s].size() > 0) {	// if the non-directed donor actually donated
								chain = { {s}, {0} };				// start the chain
								d = s;								// initialize the current donor
								while (true) {						// keep looping until the cycle is closed
									r = donationPerDonor[d][0]; w = donationPerDonor[d][1];	// find recipient and weight
									donationPerDonor[d].clear();	// remove the used arc
									chain[1][0] += w;				// add the arc's weight to the chain
									if (r != -1) {					// if we're not at tau (-1)
										chain[0].push_back(r);		// add the recipient to the chain
										d = r;						// the new donor is the recipient's donor
									}
									else { break; }					// break once tau (-1) has been reached
								}
								sol.cyclesAndChains[1].push_back(chain);	// add the completed chain
							}
						}
					}
				}
			}

			//---------------------------STEP 6: CONCLUDE ABOUT THE NEXT STEP BASED ON THE SOLUTION---------------------------	

			// If the reduced ILP was solved to optimality
			if (LBRedILP == UBRedILP) {

				// If the target has been reached: the solution is optimal and we may terminate
				if (LBRedILP >= T) {
					if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
					break;
				}

				// If no variables were deactivated: the solution is optimal and we may terminate
				else if (sol.RCVFNVarDeact == 0) {
					if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
					T = sol.LB;				// update the best upper bound
					break;
				}

				// If the target was not reached, but a feasible solution with value T - 1 was found: the solution is also optimal and we may terminate
				else if (LBRedILP >= T - 1) {
					sol.RCVFNSkips++;				// increment the number of skips
					T--;					// decrement the best upper bound by 1
					if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T << " was reached --> solution is optimal.\n"; }
					break;
				}

				// Otherwise: decrease T by 1 and try again
				else {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << "\n"; }
					T--;							// decrement the target by 1
				}
			}
			
			// If we terminated early because the UB of the reduced ILP dropped below the target 
			else if (model.get(GRB_IntAttr_Status) == GRB_USER_OBJ_LIMIT or model.get(GRB_IntAttr_Status) == GRB_CUTOFF) {
				if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
				T--;							// decrement the target by 1
			}
			
			else { // If we ran out of time
				cout << "Time limit exceeded --> return the best solution found so far\n";
				break;
			}
		}
	}

	// Save the final upper bound
	sol.UB = T;

	// Set the optimality attribute
	if (sol.LB == sol.UB) { sol.opt = true; }

	// Save the total time
	sol.timeT = getCPUTime() - start;

	// Return the solution
	return sol;
}

Solution solveHCFCFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function is used to solve HCF-CYCLE + CF-CHAIN using RCVF

	//---------------------------START OF CODE---------------------------

	// Create a solution object
	Solution sol; sol.modelCycle = "HCF-CYCLE"; sol.modelChain = "CF-CHAIN"; sol.RCVFUsed = true;
	sol.modelFull = sol.modelCycle + " + " + sol.modelChain + " (RCVF)";

	// Start the timer
	double start = getCPUTime();

	// Decide whether to use the early termination strategy
	int earlyTermination = 1;

	//---------------------------STEP 1: CREATE A MODEL FOR THE LP RELAXATION---------------------------

	// Create an environment and a model
	GRBEnv env = GRBEnv();              							// create an environment
	removeLine();                       							// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);									// create a new model

	// Declare general variables and linear expressions
	vector<GRBVar> x, y;											// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;										// objective value
	vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

	// Declare model-specific variables and linear expressions
	vector<vector<GRBLinExpr>> halfCyclesPerStartEnd;				// halfCyclesPerStartEnd[s][e] gives the sum of all half-cycle variables starting at s and ending at e

	// Declare other model-specific objects
	vector<vector<vector<int>>> halfCycles;							// all half-cycles based on max cycle length K 
	vector<vector<vector<int>>> chains;								// all chains of size at most L

	// Sort the vertices according to descending total degree
	vector<int> vertexOrder = findVertexOrder(inst, false);							// vertexOrder[i] gives the original index of the vertex that is on index i after sorting
	vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
	changeOrder(inst, vertexOrder, ANew, wToTauNew);

	// Create other model-specific objects
	halfCycles = findHalfCycles(inst, ANew);						// find the set of all half-cycles for a maximum cycle length of K
	chains = findChains(inst);										// find the set of all chains of length at most L

	// For the cycle model: create (continuous) variables and fill up linear expressions
	if (inst.K > 0) {
		halfCyclesPerStartEnd.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0));
		int s, e;
		x.resize(halfCycles.size());													// declare a variable for every half-cycle
		// for h in H: x_h = 1 iff half-cycle h is used
		for (int h = 0; h < halfCycles.size(); h++) {									// loop over all half-cycles
			x[h] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "x_" + to_string(h));			// add a corresponding vaiable
			totalWeight += halfCycles[h][1][0] * x[h];									// add the weight to the objective function
			s = halfCycles[h][0].front(); e = halfCycles[h][0].back();					// find the start and end of the half-cycle
			xIn[vertexOrder[s]] += x[h] / 2;											// find in-flow of the starting vertex: counts half
			xIn[vertexOrder[e]] += x[h] / 2;											// the in-flow of the ending vertex: counts half
			for (int i = 1; i < halfCycles[h][0].size() - 1; i++) {						// loop over all intermediate recipients in the current half-cycle
				xIn[vertexOrder[halfCycles[h][0][i]]] += x[h];							// find in-flow of theintermediate vertex: counts fully
			}
			halfCyclesPerStartEnd[s][e] += x[h];										// add the half-cycle to the sum of half-cycles starting at s and ending at t
		}

	}

	// For the chain model: create (continuous) variables and fill up linear expressions
	if (inst.L > 0) {
		y.resize(chains.size());												// declare a variable for every chain
		// for c in C: y_c = 1 iff chain c is used
		for (int c = 0; c < chains.size(); c++) {								// loop over all chains
			y[c] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "y_" + to_string(c));	// add a corresponding variable
			totalWeight += chains[c][1][0] * y[c];								// add the weight to the objective
			yOut[chains[c][0].front()] += y[c];									// update the out-flow of the NDD
			for (int i = 1; i < chains[c][0].size() - 1; i++) {					// loop over all intermediate recipients in the current chain
				yOut[chains[c][0][i]] += y[c];									// update the out-flow
				yIn[chains[c][0][i]] += y[c];									// update the in-flow
			}
			yIn[chains[c][0].back()] += y[c];									// update the in-flow of the final recipient
		}

		// add arc weights of arcs to tau													// if chains are allowed
		for (int d = 0; d < inst.nR; d++) {									// loop over all RDPs
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);				// add the weight of the corresponding final arc to the objective
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {					// loop over all NDDs
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);					// add the weight of the corresponding final arc to the objective
		}
	}

	model.update();

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	model.setObjective(totalWeight, GRB_MAXIMIZE);

	//  Constraint for every model 1: every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) {										// loop over NDDs
		if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) {														// loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
	}

	// For the cycle model: add the model-specific constraints
	if (inst.K > 0) {
		// Additional constraints 1 for HCF: matching half-cycles
		for (int s = 0; s < inst.nR - 1; s++) {				// loop over all possible start vertices s
			for (int e = s + 1; e < inst.nR; e++) {			// loop over all possible end vertices e
				// if we use a half-cycle starting at s and ending at e, we must use a matching half-cycle starting at e and ending at s
				if (halfCyclesPerStartEnd[s][e].size() >= 0) { model.addConstr(halfCyclesPerStartEnd[s][e] == halfCyclesPerStartEnd[e][s]); }
			}
		}
	}

	// For the chain model: add the model-specific constraints
	// - (none)

	model.update();

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, 2);										// use the barrier method to solve the LP relaxation 
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; }			// return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);			// update the remaining time limit
	model.getEnv().set(GRB_IntParam_Crossover, 0);									// crossover can be turned off

	//---------------------------STEP 2: SOLVE THE LP RELAXATION AND FIND ITS VALUE AND THE REDUCED COSTS---------------------------

	// Solve the LP relaxation model
	if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
	if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving initial LP...\n"; }
	model.optimize();

	// Find the LP relaxation value
	double valLP = model.get(GRB_DoubleAttr_ObjVal);
	sol.LPrel = valLP;
	if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

	// Save the model size
	sol.nVar = model.get(GRB_IntAttr_NumVars);			// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);	// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);			// number of non-zero coefficients

	// Terminate if the time limit has already been exceeded after solving the LP relaxation and simply return an empty solution
	sol.timeLP = getCPUTime() - start;
	if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }

	// Find the reduced costs of the variables
	int numVarsCycles = halfCycles.size(), numVarsChains = chains.size();
	vector<double> RCCycles(numVarsCycles, 0), RCChains(numVarsChains, 0);
	GRBVar* vars = model.getVars();
	for (int i = 0; i < numVarsCycles; i++) {
		if (vars[i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCCycles[i] = vars[i].get(GRB_DoubleAttr_RC);
		}
	}
	for (int i = 0; i < numVarsChains; i++) {
		if (vars[numVarsCycles + i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCChains[i] = vars[numVarsCycles + i].get(GRB_DoubleAttr_RC);
		}
	}

	// Clear up the memory
	model.reset(1);
	model.update();

	//---------------------------STEP 3: INITIALIZE THE REST OF THE RCVF PROCEDURE---------------------------

	// Set the initial target
	int T = floor(valLP + EPSILON);

	// Some other declarations before while-loop
	int xval, yval;														// the value of the currently considered x- or y-variable
	sol.RCVFNIter = 0;													// the number of RCVF iterations
	sol.RCVFNVarDeact = sol.nVar + 1;									// the number of variables that are deactivated (initialized to impossibly high value)
	int nVarDeactPrev;													// the number of variables that were deactivated in the previous iteration
	vector<int> bestSolCycles, bestSolChains;							// the currently best solution for the cycles and chains

	// Start the main loop of the RCVF procedure
	while (true) {									// the loop will be terminated by any of the "break" statements

		//---------------------------STEP 4: CREATE A REDUCED ILP MODEL---------------------------	

		sol.RCVFNIter++;												// increment the number of RCVF iterations by 1
		int LBRedILP = 0, UBRedILP = 1000000;							// the LB and UB of the current reduced ILP model

		// Create a new environment and a new model (these are only in scope inside of the while-loop)
		GRBEnv env = GRBEnv();              							// create an environment
		removeLine();                       							// remove Gurobi message
		if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
		GRBModel model = GRBModel(env);									// create a new model

		// Re-declare general variables and linear expressions (temporary scope)
		vector<GRBVar> x, y;											// decision variables (interpretation different per model)
		GRBLinExpr totalWeight = 0;										// objective value
		vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
		vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
		vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
		vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

		// Redeclare model-specific variables and linear expressions
		vector<vector<GRBLinExpr>> halfCyclesPerStartEnd;				// halfCyclesPerStartEnd[s][e] gives the sum of all half-cycle variables starting at s and ending at e

		// Keep track of which variables are activated
		vector<bool> isActiveCycles(numVarsCycles, false);
		vector<bool> isActiveChains(numVarsChains, false);

		// For the cycle model: create binary variables (but only when the reduced cost is high enough) and fill up linear expressions
		if (inst.K > 0) {
			halfCyclesPerStartEnd.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0));
			int s, e;
			x.resize(halfCycles.size());															// declare a variable for every half-cycle
			for (int h = 0; h < halfCycles.size(); h++) {											// loop over all half-cycles
				if (RCCycles[h] > T - valLP - EPSILON) {											// only add the variable if its RC is high enough
					isActiveCycles[h] = true;														// activate the variable
					// for h in H: x_h = 1 iff half-cycle h is used
					x[h] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(h));					// add the variable
					if (bestSolCycles.size() > 0) { x[h].set(GRB_DoubleAttr_Start, bestSolCycles[h]); }	// set a warm start
					totalWeight += halfCycles[h][1][0] * x[h];										// add the weight to the objective function
					s = halfCycles[h][0].front(); e = halfCycles[h][0].back();						// find the start and end vertex
					xIn[vertexOrder[s]] += x[h] / 2;												// update in-flow of start vertex: counts half
					xIn[vertexOrder[e]] += x[h] / 2;												// update in-flow of start vertex: counts half
					for (int i = 1; i < halfCycles[h][0].size() - 1; i++) {							// loop over all intermediate recipients in the current half-cycle
						xIn[vertexOrder[halfCycles[h][0][i]]] += x[h];								// update in-flow of intermediate recipient: counts fully
					}
					halfCyclesPerStartEnd[s][e] += x[h];											// add the half-cycle to the sum of half-cycles starting at s and ending at t
				}
			}
		}

		// For the chain model: create binary variables (but only when the RC is high enough) and fill up linear expressions
		if (inst.L > 0) {
			y.resize(chains.size());																// declare a variable for every chain
			// for c in C: y_c = 1 iff chain c is used
			for (int c = 0; c < chains.size(); c++) {												// loop over all chains
				if (RCChains[c] > T - valLP - EPSILON) {											// if the reduced cost is high enough
					isActiveChains[c] = true;														// activate the variable
					y[c] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(c));					// create the variable
					if (bestSolChains.size() > 0) { y[c].set(GRB_DoubleAttr_Start, bestSolChains[c]); }	// set a warm start
					totalWeight += chains[c][1][0] * y[c];											// add the weight to the objective value
					yOut[chains[c][0].front()] += y[c];												// update the out-flow of the NDD
					for (int i = 1; i < chains[c][0].size() - 1; i++) {								// loop over all intermediate RDPs in the current chain
						yOut[chains[c][0][i]] += y[c];												// update their out-flow
						yIn[chains[c][0][i]] += y[c];												// update their in-flow
					}
					yIn[chains[c][0].back()] += y[c];												// update the in-flow of the final RDP
				}
			}
			// add arc weights of arcs to tau
			for (int d = 0; d < inst.nR; d++) {							// loop over all RDPs
				totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);		// add the weight of the corresponding final arc to the objective
			}
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) {			// loop over all NDDs
				totalWeight += inst.wToTau[d] * (1 - yOut[d]);			// add the weight of the corresponding final arc to the objective
			}
		}

		model.update();

		// Skip the rest of the iteration if no additional variables were reactivated
		nVarDeactPrev = sol.RCVFNVarDeact;									// update the number of variables that were deactivated in the prevvious iteration
		sol.RCVFNVarDeact = sol.nVar - model.get(GRB_IntAttr_NumVars);		// find the current number of deactivated variables
		if (sol.RCVFNVarDeact == nVarDeactPrev and earlyTermination == 0) {	// if no additional variables are activated
			if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
			sol.RCVFNSkips++;												// increment the number of skips
			sol.RCVFNIter--;												// undo the increment of the number of (proper) iterations (in which an ILP model is solved)
			T--;															// reduce the target by 1
			if (sol.LB == T) {												// if the best solution reaches the new target, it is optimal
				if (printRCVFLog) { cout << "Target T = " << T << " had already been reached--> solution is optimal.\n"; }
				break;
			}
		}
		else {

			// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// Change a setting so that we terminate once the UB of the current reduced ILP drops below T (using some tolerance)
			if (earlyTermination == 1) { model.set(GRB_DoubleParam_BestBdStop, T - 0.01); }
			else if (earlyTermination == 2) { model.set(GRB_DoubleParam_Cutoff, T - 0.01); }

			//  Constraint for every model 1: every NDD can donate at most once
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
				if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
			}

			// Constraint for every model 2: every RDP can be donated to at most once
			// - note that these are the only contraints linking the cycle and chain component of the model
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
			}

			// For the cycle model: add the model-specific constraints
			if (inst.K > 0) {
				// Additional constraints 1 for HCF: matching half-cycles
				for (int s = 0; s < inst.nR - 1; s++) {				// loop over all possible start vertices s
					for (int e = s + 1; e < inst.nR; e++) {			// loop over all possible end vertices e
						// if we use a half-cycle starting at s and ending at e, we must use a matching half-cycle starting at e and ending at s
						if (halfCyclesPerStartEnd[s][e].size() >= 0) { model.addConstr(halfCyclesPerStartEnd[s][e] == halfCyclesPerStartEnd[e][s]); }
					}
				}
			}

			// For the chain model: add the model-specific constraints
			// - (none)

			model.update();

			// Change some settings
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_IntParam_Method, 2);														// use the barrier method to solve the LP relaxation
			if (timeLimit - getCPUTime() + start < 0) { sol.timeT = timeLimit; sol.UB = T;  return sol; }	// return an empty solution if the time limit has already been exceeded in pre-processing
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - getCPUTime() + start);				// update the remaining time limit
			// model.getEnv().set(GRB_DoubleParam_BestObjStop, T);												// we can terminate early if a solution with value T is found
			// don't turn off crossover here


			//---------------------------STEP 5: SOLVE THE REDUCED ILP MODEL---------------------------	

			if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving ILP model, iter = " << sol.RCVFNIter << ", time = " << getCPUTime() - start << "s, target value T = " << T << "\n"; }
			if (printRCVFLog) { cout << "Number of variables removed = " << sol.RCVFNVarDeact << " (out of " << sol.nVar << " originally), " << model.get(GRB_IntAttr_NumVars) << " variables remain\n"; }

			model.optimize();

			// Find the best upper bound
			UBRedILP = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (UBRedILP < 0) { UBRedILP = 1000000; }						// if no upper bound was found yet, set it to 1000000

			// If the model is infeasible
			if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
				if (earlyTermination != 0) {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
					T--;							// decrement the target by 1
					continue;
				}
				else {
					sol.feas = false;											// indicate that the instance is infeasible
					break;
				}
			}

			// If a solution has been found within the time limit
			else if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.feas = true;												// indicate that the instance is feasible
				LBRedILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// find the best lower bound

				if (printRCVFLog) { cout << "Value reduced ILP = " << LBRedILP << "\n"; }

				// If the solution is better than the current best one, update the solution (incl. the warm start)
				if (LBRedILP >= sol.LB) {

					// Set the new best LB
					sol.LB = LBRedILP;


					// Reset the vectors containing the best solution found so far
					bestSolCycles.clear(); bestSolCycles.resize(numVarsCycles, 0);
					bestSolChains.clear(); bestSolChains.resize(numVarsChains, 0);

					// Decompose the solution into the selected cycles and chains
					sol.cyclesAndChains.clear();
					sol.cyclesAndChains.resize(2);

					// First consider the cycles
					if (inst.K > 0) {
						vector<vector<vector<vector<int>>>> halfCyclesUsedPerStartEnd(inst.nR, vector<vector<vector<int>>>(inst.nR));
						vector<vector<int>> h, h1, h2, c;
						int s, e;
						for (int i = 0; i < halfCycles.size(); i++) {							// loop over all half-cycles
							if (isActiveCycles[i]) {											// if the corresponding variable is active
								xval = ceil(x[i].get(GRB_DoubleAttr_X) - EPSILON);				// find variable value
								if (xval == 1) {												// if the half-cycle is selected
									h = halfCycles[i]; s = h[0].front(); e = h[0].back();		// find the half-cycle, its start and end
									halfCyclesUsedPerStartEnd[s][e] = h;						// save the half-cycle as a half-cycle starting at s and ending at e
									bestSolCycles[i] = 1;										// save the solution
								}
							}
						}
						for (int s = 0; s < inst.nR - 1; s++) {								// loop over all starting points	
							for (int e = s + 1; e < inst.nR; e++) {							// loop over all ending points
								if (halfCyclesUsedPerStartEnd[s][e].size() > 0) {			// if there is a half-cycle starting at s and ending at e
									h1 = halfCyclesUsedPerStartEnd[s][e];					// find the half-cycle
									h2 = halfCyclesUsedPerStartEnd[e][s];					// find a matching half-cycle
									h1[0].pop_back(); h2[0].pop_back();						// remove the ends of both half-cycles (they are double)
									c.clear(); c.resize(2);									// initialize a cycle that will be the combination of both half-cycles
									for (int d : h1[0]) { c[0].push_back(vertexOrder[d]); }	// add the (original) vertices of the first half-cycle
									for (int d : h2[0]) { c[0].push_back(vertexOrder[d]); }	// add the (original) vertices of the second half-cycle
									c[1] = { h1[1][0] + h2[1][0] };							// the weight is the sum of the two weights
									sol.cyclesAndChains[0].push_back(c);					// add the completed cycle
								}
							}
						}
					}

					// Second consider the chains
					if (inst.L > 0) {
						for (int c = 0; c < chains.size(); c++) {												// loop over all chains
							if (isActiveChains[c]) {															// if the variable was activated
								yval = ceil(y[c].get(GRB_DoubleAttr_X) - EPSILON);								// find variable value
								if (yval == 1) {																// if the chain is selected
									sol.cyclesAndChains[1].push_back(chains[c]);								// add the chain to the solution
									sol.cyclesAndChains[1].back()[1][0] += inst.wToTau[chains[c][0].back()];	// also add the weight of the final arc
									bestSolChains[c] = 1;														// save the solution
								}
							}
						}
						// Also add trivial n-tau chains for unused NDDs n
						for (int d = inst.nR; d < inst.nR + inst.nN; d++) {						// loop over all NDDs
							if (ceil(yOut[d].getValue() - EPSILON) == 0) {						// if its out-flow is 0
								sol.cyclesAndChains[1].push_back({ {d}, {inst.wToTau[d]} });	// add the corresponding trivial chain
							}
						}
					}
				}
			}

			//---------------------------STEP 6: CONCLUDE ABOUT THE NEXT STEP BASED ON THE SOLUTION---------------------------	

			// If the reduced ILP was solved to optimality
			if (LBRedILP == UBRedILP) {

				// If the target has been reached: the solution is optimal and we may terminate
				if (LBRedILP >= T) {
					if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
					break;
				}

				// If no variables were deactivated: the solution is optimal and we may terminate
				else if (sol.RCVFNVarDeact == 0) {
					if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
					T = sol.LB;				// update the best upper bound
					break;
				}

				// If the target was not reached, but a feasible solution with value T - 1 was found: the solution is also optimal and we may terminate
				else if (LBRedILP >= T - 1) {
					sol.RCVFNSkips++;				// increment the number of skips
					T--;							// decrement the best upper bound by 1
					if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T << " was reached --> solution is optimal.\n"; }
					break;
				}

				// Otherwise: decrease T by 1 and try again
				else {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << "\n"; }
					T--;							// decrement the target by 1
				}
			}

			// If we terminated early because the UB of the reduced ILP dropped below the target 
			else if (model.get(GRB_IntAttr_Status) == GRB_USER_OBJ_LIMIT or model.get(GRB_IntAttr_Status) == GRB_CUTOFF) {
				if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
				T--;							// decrement the target by 1
			}

			else { // If we ran out of time
				cout << "Time limit exceeded --> return the best solution found so far\n";
				break;
			}
		}
	}

	// Save the final upper bound
	sol.UB = T;

	// Set the optimality attribute
	if (sol.LB == sol.UB) { sol.opt = true; }

	// Save the total time
	sol.timeT = getCPUTime() - start;

	// Return the solution
	return sol;
}

Solution solveHCFPIEFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function is used to solve HCF-CYCLE + PIEF-CHAIN using RCVF


	//---------------------------START OF CODE---------------------------

	// Create a solution object
	Solution sol; sol.modelCycle = "HCF-CYCLE"; sol.modelChain = "PIEF-CHAIN"; sol.RCVFUsed = true;
	sol.modelFull = sol.modelCycle + " + " + sol.modelChain + " (RCVF)";

	// Start the timer
	double start = getCPUTime();

	// Decide whether to use the early termination strategy
	int earlyTermination = 1;

	//---------------------------STEP 1: CREATE A MODEL FOR THE LP RELAXATION---------------------------

	// Create an environment and a model
	GRBEnv env = GRBEnv();              							// create an environment
	removeLine();                       							// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);									// create a new model

	// Declare general variables and linear expressions
	vector<GRBVar> x, y;											// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;										// objective value
	vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

	// Declare model-specific variables and linear expressions
	vector<vector<GRBLinExpr>> halfCyclesPerStartEnd;				// halfCyclesPerStartEnd[s][e] gives the sum of all half-cycle variables starting at s and ending at e
	vector<vector<GRBVar>> yL;										// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
	vector<vector<GRBLinExpr>> yOutL, yInL;							// number of transplants made by each donor/recieved by each recipient at every position

	// Declare other model-specific objects
	vector<vector<vector<int>>> halfCycles;							// all half-cycles based on max cycle length K 
	vector<Subgraph> subgraphPIEFCHAIN;								// the reduced arc sets for each arc position 1,...,L-1

	// Sort the vertices according to descending total degree
	vector<int> vertexOrder = findVertexOrder(inst, false);							// vertexOrder[i] gives the original index of the vertex that is on index i after sorting
	vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
	changeOrder(inst, vertexOrder, ANew, wToTauNew);

	// Create other model-specific objects
	halfCycles = findHalfCycles(inst, ANew);						// find the set of all half-cycles for a maximum cycle length of K
	subgraphPIEFCHAIN = createSubgraphPIEFCHAIN(inst);				// create the position-indexed graph

	// For the cycle model: create (continuous) variables and fill up linear expressions
	if (inst.K > 0) {
		halfCyclesPerStartEnd.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0));
		int s, e;
		x.resize(halfCycles.size());													// declare a variable for every half-cycle
		// for h in H: x_h = 1 iff half-cycle h is used
		for (int h = 0; h < halfCycles.size(); h++) {									// loop over all half-cycles
			x[h] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "x_" + to_string(h));			// add a corresponding vaiable
			totalWeight += halfCycles[h][1][0] * x[h];									// add the weight to the objective function
			s = halfCycles[h][0].front(); e = halfCycles[h][0].back();					// find the start and end of the half-cycle
			xIn[vertexOrder[s]] += x[h] / 2;											// find in-flow of the starting vertex: counts half
			xIn[vertexOrder[e]] += x[h] / 2;											// the in-flow of the ending vertex: counts half
			for (int i = 1; i < halfCycles[h][0].size() - 1; i++) {						// loop over all intermediate recipients in the current half-cycle
				xIn[vertexOrder[halfCycles[h][0][i]]] += x[h];							// find in-flow of theintermediate vertex: counts fully
			}
			halfCyclesPerStartEnd[s][e] += x[h];										// add the half-cycle to the sum of half-cycles starting at s and ending at t
		}

	}

	// For the chain model: create (continuous) variables and fill up linear expressions
	int numVarsChains = 0;
	if (inst.L > 0) {
		yL.resize(inst.L); // seperate set of variables for all positions
		int d, r, w;
		yOutL.resize(inst.nR + inst.nN, vector<GRBLinExpr>(inst.L, 0));
		yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
		// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
		for (int k = 1; k <= inst.L - 1; k++) {
			yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
			for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
				yL[k][a] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "y_" + to_string(a));
				numVarsChains++;
				d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
				totalWeight += w * yL[k][a];
				yOut[d] += yL[k][a];
				yOutL[d][k] += yL[k][a];
				yIn[r] += yL[k][a];
				yInL[r][k] += yL[k][a];
			}
		}

		// add arc weights of arcs to tau													// if chains are allowed
		for (int d = 0; d < inst.nR; d++) {									// loop over all RDPs
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);				// add the weight of the corresponding final arc to the objective
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {					// loop over all NDDs
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);					// add the weight of the corresponding final arc to the objective
		}
	}

	model.update();

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	model.setObjective(totalWeight, GRB_MAXIMIZE);

	//  Constraint for every model 1: every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) {										// loop over NDDs
		if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) {														// loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
	}

	// For the cycle model: add the model-specific constraints
	if (inst.K > 0) {
		// Additional constraints 1 for HCF: matching half-cycles
		for (int s = 0; s < inst.nR - 1; s++) {				// loop over all possible start vertices s
			for (int e = s + 1; e < inst.nR; e++) {			// loop over all possible end vertices e
				// if we use a half-cycle starting at s and ending at e, we must use a matching half-cycle starting at e and ending at s
				if (halfCyclesPerStartEnd[s][e].size() >= 0) { model.addConstr(halfCyclesPerStartEnd[s][e] == halfCyclesPerStartEnd[e][s]); }
			}
		}
	}

	// For the chain model: add the model-specific constraints
	if (inst.L > 0) {
		// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
		for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
			for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
				if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
			}
		}
	}

	model.update();

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, 2);										// use the barrier method to solve the LP relaxation 
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; }			// return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);			// update the remaining time limit
	model.getEnv().set(GRB_IntParam_Crossover, 0);									// crossover can be turned off

	//---------------------------STEP 2: SOLVE THE LP RELAXATION AND FIND ITS VALUE AND THE REDUCED COSTS---------------------------

	// Solve the LP relaxation model
	if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
	if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving initial LP...\n"; }
	model.optimize();

	// Find the LP relaxation value
	double valLP = model.get(GRB_DoubleAttr_ObjVal);
	sol.LPrel = valLP;
	if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

	// Save the model size
	sol.nVar = model.get(GRB_IntAttr_NumVars);			// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);	// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);			// number of non-zero coefficients

	// Terminate if the time limit has already been exceeded after solving the LP relaxation and simply return an empty solution
	sol.timeLP = getCPUTime() - start;
	if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }

	// Find the reduced costs of the variables
	int numVarsCycles = halfCycles.size();
	vector<double> RCCycles(numVarsCycles, 0), RCChains(numVarsChains, 0);
	GRBVar* vars = model.getVars();
	for (int i = 0; i < numVarsCycles; i++) {
		if (vars[i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCCycles[i] = vars[i].get(GRB_DoubleAttr_RC);
		}
	}
	for (int i = 0; i < numVarsChains; i++) {
		if (vars[numVarsCycles + i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCChains[i] = vars[numVarsCycles + i].get(GRB_DoubleAttr_RC);
		}
	}

	// Clear up the memory
	model.reset(1);
	model.update();

	//---------------------------STEP 3: INITIALIZE THE REST OF THE RCVF PROCEDURE---------------------------

	// Set the initial target
	int T = floor(valLP + EPSILON);

	// Some other declarations before while-loop
	int xval, yval;														// the value of the currently considered x- or y-variable
	sol.RCVFNIter = 0;													// the number of RCVF iterations
	sol.RCVFNVarDeact = sol.nVar + 1;									// the number of variables that are deactivated (initialized to impossibly high value)
	int nVarDeactPrev;													// the number of variables that were deactivated in the previous iteration
	vector<int> bestSolCycles, bestSolChains;							// the currently best solution for the cycles and chains

	// Start the main loop of the RCVF procedure
	while (true) {		// the loop will be terminated by any of the "break" statements

		//---------------------------STEP 4: CREATE A REDUCED ILP MODEL---------------------------	

		sol.RCVFNIter++;												// increment the number of RCVF iterations by 1
		int LBRedILP = 0, UBRedILP = 1000000;							// the LB and UB of the current reduced ILP model

		// Create a new environment and a new model (these are only in scope inside of the while-loop)
		GRBEnv env = GRBEnv();              							// create an environment
		removeLine();                       							// remove Gurobi message
		if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
		GRBModel model = GRBModel(env);									// create a new model

		// Re-declare general variables and linear expressions (temporary scope)
		vector<GRBVar> x, y;											// decision variables (interpretation different per model)
		GRBLinExpr totalWeight = 0;										// objective value
		vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
		vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
		vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
		vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

		// Redeclare model-specific variables and linear expressions
		vector<vector<GRBLinExpr>> halfCyclesPerStartEnd;				// halfCyclesPerStartEnd[s][e] gives the sum of all half-cycle variables starting at s and ending at e
		vector<vector<GRBVar>> yL;										// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
		vector<vector<GRBLinExpr>> yOutL, yInL;							// number of transplants made by each donor/recieved by each recipient at every position

		// Keep track of which variables are activated
		vector<bool> isActiveCycles(numVarsCycles, false);
		vector<bool> isActiveChains(numVarsChains, false);

		// For the cycle model: create binary variables (but only when the reduced cost is high enough) and fill up linear expressions
		if (inst.K > 0) {
			halfCyclesPerStartEnd.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0));
			int s, e;
			x.resize(halfCycles.size());															// declare a variable for every half-cycle
			for (int h = 0; h < halfCycles.size(); h++) {											// loop over all half-cycles
				if (RCCycles[h] > T - valLP - EPSILON) {											// only add the variable if its RC is high enough
					isActiveCycles[h] = true;														// activate the variable
					// for h in H: x_h = 1 iff half-cycle h is used
					x[h] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(h));					// add the variable
					if (bestSolCycles.size() > 0) { x[h].set(GRB_DoubleAttr_Start, bestSolCycles[h]); }	// set a warm start
					totalWeight += halfCycles[h][1][0] * x[h];										// add the weight to the objective function
					s = halfCycles[h][0].front(); e = halfCycles[h][0].back();						// find the start and end vertex
					xIn[vertexOrder[s]] += x[h] / 2;												// update in-flow of start vertex: counts half
					xIn[vertexOrder[e]] += x[h] / 2;												// update in-flow of start vertex: counts half
					for (int i = 1; i < halfCycles[h][0].size() - 1; i++) {							// loop over all intermediate recipients in the current half-cycle
						xIn[vertexOrder[halfCycles[h][0][i]]] += x[h];								// update in-flow of intermediate recipient: counts fully
					}
					halfCyclesPerStartEnd[s][e] += x[h];											// add the half-cycle to the sum of half-cycles starting at s and ending at t
				}
			}
		}

		// For the chain model: create binary variables (but only when the RC is high enough) and fill up linear expressions
		if (inst.L > 0) {
			yL.resize(inst.L); // seperate set of variables for all positions
			int d, r, w;
			yOutL.resize(inst.nR + inst.nN, vector<GRBLinExpr>(inst.L, 0));
			yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
			int varIdx = -1;
			// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
			for (int k = 1; k <= inst.L - 1; k++) {
				yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
				for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
					varIdx++;
					if (RCChains[varIdx] > T - valLP - EPSILON) {
						yL[k][a] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(a));
						isActiveChains[varIdx] = true;
						if (bestSolChains.size() > 0) { yL[k][a].set(GRB_DoubleAttr_Start, bestSolChains[varIdx]); }	// set a warm start
						d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
						totalWeight += w * yL[k][a];
						yOut[d] += yL[k][a];
						yOutL[d][k] += yL[k][a];
						yIn[r] += yL[k][a];
						yInL[r][k] += yL[k][a];
					}
				}
			}
			// add arc weights of arcs to tau
			for (int d = 0; d < inst.nR; d++) {							// loop over all RDPs
				totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);		// add the weight of the corresponding final arc to the objective
			}
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) {			// loop over all NDDs
				totalWeight += inst.wToTau[d] * (1 - yOut[d]);			// add the weight of the corresponding final arc to the objective
			}
		}

		model.update();

		// Skip the rest of the iteration if no additional variables were reactivated
		nVarDeactPrev = sol.RCVFNVarDeact;									// update the number of variables that were deactivated in the prevvious iteration
		sol.RCVFNVarDeact = sol.nVar - model.get(GRB_IntAttr_NumVars);		// find the current number of deactivated variables
		if (sol.RCVFNVarDeact == nVarDeactPrev and earlyTermination == 0) {	// if no additional variables are activated
			if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
			sol.RCVFNSkips++;												// increment the number of skips
			sol.RCVFNIter--;												// undo the increment of the number of (proper) iterations (in which an ILP model is solved)
			T--;															// reduce the target by 1
			if (sol.LB == T) {												// if the best solution reaches the new target, it is optimal
				if (printRCVFLog) { cout << "Target T = " << T << " had already been reached--> solution is optimal.\n"; }
				break;
			}
		}
		else {

			// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// Change a setting so that we terminate once the UB of the current reduced ILP drops below T (using some tolerance)
			if (earlyTermination == 1) { model.set(GRB_DoubleParam_BestBdStop, T - 0.01); }
			else if (earlyTermination == 2) { model.set(GRB_DoubleParam_Cutoff, T - 0.01); }

			//  Constraint for every model 1: every NDD can donate at most once
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
				if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
			}

			// Constraint for every model 2: every RDP can be donated to at most once
			// - note that these are the only contraints linking the cycle and chain component of the model
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
			}

			// For the cycle model: add the model-specific constraints
			if (inst.K > 0) {
				// Additional constraints 1 for HCF: matching half-cycles
				for (int s = 0; s < inst.nR - 1; s++) {				// loop over all possible start vertices s
					for (int e = s + 1; e < inst.nR; e++) {			// loop over all possible end vertices e
						// if we use a half-cycle starting at s and ending at e, we must use a matching half-cycle starting at e and ending at s
						if (halfCyclesPerStartEnd[s][e].size() >= 0) { model.addConstr(halfCyclesPerStartEnd[s][e] == halfCyclesPerStartEnd[e][s]); }
					}
				}
			}

			// For the chain model: add the model-specific constraints
			if (inst.L > 0) {
				// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
				for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
					for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
						if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
					}
				}
			}

			model.update();

			// Change some settings
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_IntParam_Method, 2);														// use the barrier method to solve the LP relaxation
			if (timeLimit - getCPUTime() + start < 0) { sol.timeT = timeLimit; sol.UB = T;  return sol; }	// return an empty solution if the time limit has already been exceeded in pre-processing
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - getCPUTime() + start);				// update the remaining time limit
			// model.getEnv().set(GRB_DoubleParam_BestObjStop, T);												// we can terminate early if a solution with value T is found
			// don't turn off crossover here


			//---------------------------STEP 5: SOLVE THE REDUCED ILP MODEL---------------------------	
			if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving ILP model, iter = " << sol.RCVFNIter << ", time = " << getCPUTime() - start << "s, target value T = " << T << "\n"; }
			if (printRCVFLog) { cout << "Number of variables removed = " << sol.RCVFNVarDeact << " (out of " << sol.nVar << " originally), " << model.get(GRB_IntAttr_NumVars) << " variables remain\n"; }

			model.optimize();

			// Find the best upper bound
			UBRedILP = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (UBRedILP < 0) { UBRedILP = 1000000; }						// if no upper bound was found yet, set it to 1000000

			// If the model is infeasible
			if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
				if (earlyTermination != 0) {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
					T--;							// decrement the target by 1
					continue;
				}
				else {
					sol.feas = false;											// indicate that the instance is infeasible
					break;
				}
			}

			// If a solution has been found within the time limit
			else if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.feas = true;												// indicate that the instance is feasible
				LBRedILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// find the best lower bound

				if (printRCVFLog) { cout << "Value reduced ILP = " << LBRedILP << "\n"; }

				// If the solution is better than the current best one, update the solution (incl. the warm start)
				if (LBRedILP >= sol.LB) {

					// Set the new best LB
					sol.LB = LBRedILP;


					// Reset the vectors containing the best solution found so far
					bestSolCycles.clear(); bestSolCycles.resize(numVarsCycles, 0);
					bestSolChains.clear(); bestSolChains.resize(numVarsChains, 0);

					// Decompose the solution into the selected cycles and chains
					sol.cyclesAndChains.clear();
					sol.cyclesAndChains.resize(2);

					// First consider the cycles
					if (inst.K > 0) {
						vector<vector<vector<vector<int>>>> halfCyclesUsedPerStartEnd(inst.nR, vector<vector<vector<int>>>(inst.nR));
						vector<vector<int>> h, h1, h2, c;
						int s, e;
						for (int i = 0; i < halfCycles.size(); i++) {							// loop over all half-cycles
							if (isActiveCycles[i]) {											// if the corresponding variable is active
								xval = ceil(x[i].get(GRB_DoubleAttr_X) - EPSILON);				// find variable value
								if (xval == 1) {												// if the half-cycle is selected
									h = halfCycles[i]; s = h[0].front(); e = h[0].back();		// find the half-cycle, its start and end
									halfCyclesUsedPerStartEnd[s][e] = h;						// save the half-cycle as a half-cycle starting at s and ending at e
									bestSolCycles[i] = 1;										// save the solution
								}
							}
						}
						for (int s = 0; s < inst.nR - 1; s++) {								// loop over all starting points	
							for (int e = s + 1; e < inst.nR; e++) {							// loop over all ending points
								if (halfCyclesUsedPerStartEnd[s][e].size() > 0) {			// if there is a half-cycle starting at s and ending at e
									h1 = halfCyclesUsedPerStartEnd[s][e];					// find the half-cycle
									h2 = halfCyclesUsedPerStartEnd[e][s];					// find a matching half-cycle
									h1[0].pop_back(); h2[0].pop_back();						// remove the ends of both half-cycles (they are double)
									c.clear(); c.resize(2);									// initialize a cycle that will be the combination of both half-cycles
									for (int d : h1[0]) { c[0].push_back(vertexOrder[d]); }	// add the (original) vertices of the first half-cycle
									for (int d : h2[0]) { c[0].push_back(vertexOrder[d]); }	// add the (original) vertices of the second half-cycle
									c[1] = { h1[1][0] + h2[1][0] };							// the weight is the sum of the two weights
									sol.cyclesAndChains[0].push_back(c);					// add the completed cycle
								}
							}
						}
					}

					// Second consider the chains
					if (inst.L > 0) {
						vector<vector<int>> donationPerDonor(inst.nR + inst.nN);		// array containing [r, w_dr] for each donor d that donates
						int d, r, w;
						int varIdx = -1;
						for (int k = 1; k <= inst.L - 1; k++) {								// loop over all arc positions
							for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {	// loop over all arcs
								varIdx++;
								if (isActiveChains[varIdx]) {
									yval = ceil(yL[k][a].get(GRB_DoubleAttr_X) - EPSILON);		// find variable value
									if (yval == 1) {											// if the arc is selected
										d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2]; // decompose the arc
										donationPerDonor[d] = { r, w };							// add the arc to the list of donations
										bestSolChains[varIdx] = 1;								// save the solution
									}
								}
							}
						}
						// Also find the donations to tau
						if (inst.L > 1) {
							for (int d = 0; d < inst.nR; d++) {
								if (ceil(yIn[d].getValue() - EPSILON) - ceil(yOut[d].getValue() - EPSILON) == 1) {
									donationPerDonor[d] = { -1, inst.wToTau[d] };
								}
							}
						}
						for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
							if (ceil(yOut[d].getValue() - EPSILON) == 0) {
								donationPerDonor[d] = { -1, inst.wToTau[d] };
							}
						}
						// Find the chains
						vector<vector<int>> chain;					// array containing the current chain and its weight
						for (int s = inst.nR; s < inst.nR + inst.nN; s++) { // loop over all non-directed donors
							if (donationPerDonor[s].size() > 0) {	// if the non-directed donor actually donated
								chain = { {s}, {0} };				// start the chain
								d = s;								// initialize the current donor
								while (true) {						// keep looping until the cycle is closed
									r = donationPerDonor[d][0]; w = donationPerDonor[d][1];	// find recipient and weight
									donationPerDonor[d].clear();	// remove the used arc
									chain[1][0] += w;				// add the arc's weight to the chain
									if (r != -1) {					// if we're not at tau (-1)
										chain[0].push_back(r);		// add the recipient to the chain
										d = r;						// the new donor is the recipient's donor
									}
									else { break; }					// break once tau (-1) has been reached
								}
								sol.cyclesAndChains[1].push_back(chain);	// add the completed chain
							}
						}
					}
				}
			}

			//---------------------------STEP 6: CONCLUDE ABOUT THE NEXT STEP BASED ON THE SOLUTION---------------------------	

			// If the reduced ILP was solved to optimality
			if (LBRedILP == UBRedILP) {

				// If the target has been reached: the solution is optimal and we may terminate
				if (LBRedILP >= T) {
					if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
					break;
				}

				// If no variables were deactivated: the solution is optimal and we may terminate
				else if (sol.RCVFNVarDeact == 0) {
					if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
					T = sol.LB;				// update the best upper bound
					break;
				}

				// If the target was not reached, but a feasible solution with value T - 1 was found: the solution is also optimal and we may terminate
				else if (LBRedILP >= T - 1) {
					sol.RCVFNSkips++;				// increment the number of skips
					T--;							// decrement the best upper bound by 1
					if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T << " was reached --> solution is optimal.\n"; }
					break;
				}

				// Otherwise: decrease T by 1 and try again
				else {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << "\n"; }
					T--;							// decrement the target by 1
				}
			}

			// If we terminated early because the UB of the reduced ILP dropped below the target 
			else if (model.get(GRB_IntAttr_Status) == GRB_USER_OBJ_LIMIT or model.get(GRB_IntAttr_Status) == GRB_CUTOFF) {
				if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
				T--;							// decrement the target by 1
			}

			else { // If we ran out of time
				cout << "Time limit exceeded --> return the best solution found so far\n";
				break;
			}
		}
	}

	// Save the final upper bound
	sol.UB = T;

	// Set the optimality attribute
	if (sol.LB == sol.UB) { sol.opt = true; }

	// Save the total time
	sol.timeT = getCPUTime() - start;

	// Return the solution
	return sol;
}

Solution solvePIEFCFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function is used to solve PIEF-CYCLE + CF-CHAIN using RCVF

	//---------------------------START OF CODE---------------------------

	// Create a solution object
	Solution sol; sol.modelCycle = "PIEF-CYCLE"; sol.modelChain = "CF-CHAIN"; sol.RCVFUsed = true;
	sol.modelFull = sol.modelCycle + " + " + sol.modelChain + " (RCVF)";

	// Start the timer
	double start = getCPUTime();

	// Decide whether to use the early termination strategy
	int earlyTermination = 1;

	//---------------------------STEP 1: CREATE A MODEL FOR THE LP RELAXATION---------------------------

	// Create an environment and a model
	GRBEnv env = GRBEnv();              							// create an environment
	removeLine();                       							// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);									// create a new model

	// Declare general variables and linear expressions
	vector<GRBVar> x, y;											// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;										// objective value
	vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

	// Declare model-specific variables and linear expressions
	vector<vector<vector<GRBVar>>> xSubK;							// decision variables modelling for each arc in each subgraph at every possible position whether the arc is selected at that position (1) or not (0)
	vector<vector<vector<GRBLinExpr>>> xOutSubK, xInSubK;			// number of transplants made by each donor/recieved by each recipient in each subgraph at every position

	// Declare other model-specific objects
	vector<vector<Subgraph>> subgraphsPIEF;							// the reduced arc sets for each (sorted) subgraph G^1,...,G^|R| for each arc position 1,...,K
	vector<vector<vector<int>>> chains;								// all chains of size at most L

	// Sort the vertices according to descending total degree
	vector<int> vertexOrder = findVertexOrder(inst, false);							// vertexOrder[i] gives the original index of the vertex that is on index i after sorting
	vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
	changeOrder(inst, vertexOrder, ANew, wToTauNew);

	// Create other model-specific objects
	subgraphsPIEF.resize(inst.nR);								// declare R subgraphs for PIEF
	for (int l = 0; l < inst.nR; l++) {							// loop over all subgraphs
		subgraphsPIEF[l] = createSubgraphPIEFCYCLE(inst, ANew, l);	// create the subgraph
	}
	chains = findChains(inst);										// find the set of all chains of length at most L

	// For the cycle model: create (continuous) variables and fill up linear expressions
	int numVarsCycles = 0;
	if (inst.K > 0) {
		xSubK.resize(inst.nR, vector<vector<GRBVar>>(inst.K + 1));
		int d, r, w;
		xOutSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
		xInSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
		// for l in R, a in Al and k in K(a,l): x_lka = 1 iff arc a is used in position k in subgraph l
		for (int l = 0; l < inst.nR; l++) {															// loop over all subgraphs
			for (int k = 1; k <= inst.K; k++) {														// loop over all arc positions
				xSubK[l][k].resize(subgraphsPIEF[l][k].idx2Arc.size());								// reshape the matrix of decision variables
				for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {						// loop over all arcs
					numVarsCycles++;																// increment the number of cycle variables
					xSubK[l][k][a] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "x_" + to_string(a));
					d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2];
					totalWeight += w * xSubK[l][k][a];												// update the objective value
					xOut[vertexOrder[d]] += xSubK[l][k][a];											// note: in terms of original vertex index
					xOutSubK[l][d][k] += xSubK[l][k][a];
					xIn[vertexOrder[r]] += xSubK[l][k][a];											// note: in terms of original vertex index
					xInSubK[l][r][k] += xSubK[l][k][a];
				}
			}
		}
	}

	// For the chain model: create (continuous) variables and fill up linear expressions
	if (inst.L > 0) {
		y.resize(chains.size());												// declare a variable for every chain
		// for c in C: y_c = 1 iff chain c is used
		for (int c = 0; c < chains.size(); c++) {								// loop over all chains
			y[c] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "y_" + to_string(c));	// add a corresponding variable
			totalWeight += chains[c][1][0] * y[c];								// add the weight to the objective
			yOut[chains[c][0].front()] += y[c];									// update the out-flow of the NDD
			for (int i = 1; i < chains[c][0].size() - 1; i++) {					// loop over all intermediate recipients in the current chain
				yOut[chains[c][0][i]] += y[c];									// update the out-flow
				yIn[chains[c][0][i]] += y[c];									// update the in-flow
			}
			yIn[chains[c][0].back()] += y[c];									// update the in-flow of the final recipient
		}

		// add arc weights of arcs to tau													// if chains are allowed
		for (int d = 0; d < inst.nR; d++) {									// loop over all RDPs
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);				// add the weight of the corresponding final arc to the objective
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {					// loop over all NDDs
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);					// add the weight of the corresponding final arc to the objective
		}
	}

	model.update();

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	model.setObjective(totalWeight, GRB_MAXIMIZE);

	//  Constraint for every model 1: every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) {										// loop over NDDs
		if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) {														// loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
	}

	// For the cycle model: add the model-specific constraints
	if (inst.K > 0) {
		// Additional constraints 1 for PIEF: flow conservation per subgraph per index
		for (int l = 0; l < inst.nR; l++) {						// loop over subgraphs
			for (int r = l + 1; r < inst.nR; r++) {				// loop over RDPs
				for (int k = 1; k <= inst.K - 1; k++) {			// loop over arc indices
					if (xInSubK[l][r][k].size() + xOutSubK[l][r][k + 1].size() > 0) { model.addConstr(xInSubK[l][r][k] == xOutSubK[l][r][k + 1]); }
				}
			}
		}
	}

	// For the chain model: add the model-specific constraints
	// - (none)

	model.update();

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, 2);										// use the barrier method to solve the LP relaxation 
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; }			// return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);			// update the remaining time limit
	model.getEnv().set(GRB_IntParam_Crossover, 0);									// crossover can be turned off

	//---------------------------STEP 2: SOLVE THE LP RELAXATION AND FIND ITS VALUE AND THE REDUCED COSTS---------------------------

	// Solve the LP relaxation model
	if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
	if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving initial LP...\n"; }
	model.optimize();

	// Find the LP relaxation value
	double valLP = model.get(GRB_DoubleAttr_ObjVal);
	sol.LPrel = valLP;
	if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

	// Save the model size
	sol.nVar = model.get(GRB_IntAttr_NumVars);			// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);	// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);			// number of non-zero coefficients

	// Terminate if the time limit has already been exceeded after solving the LP relaxation and simply return an empty solution
	sol.timeLP = getCPUTime() - start;
	if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }

	// Find the reduced costs of the variables
	int numVarsChains = chains.size();
	vector<double> RCCycles(numVarsCycles, 0), RCChains(numVarsChains, 0);
	GRBVar* vars = model.getVars();
	for (int i = 0; i < numVarsCycles; i++) {
		if (vars[i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCCycles[i] = vars[i].get(GRB_DoubleAttr_RC);
		}
	}
	for (int i = 0; i < numVarsChains; i++) {
		if (vars[numVarsCycles + i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCChains[i] = vars[numVarsCycles + i].get(GRB_DoubleAttr_RC);
		}
	}

	// Clear up the memory
	model.reset(1);
	model.update();

	//---------------------------STEP 3: INITIALIZE THE REST OF THE RCVF PROCEDURE---------------------------

	// Set the initial target
	int T = floor(valLP + EPSILON);

	// Some other declarations before while-loop
	int xval, yval;														// the value of the currently considered x- or y-variable
	sol.RCVFNIter = 0;													// the number of RCVF iterations
	sol.RCVFNVarDeact = sol.nVar + 1;									// the number of variables that are deactivated (initialized to impossibly high value)
	int nVarDeactPrev;													// the number of variables that were deactivated in the previous iteration
	vector<int> bestSolCycles, bestSolChains;							// the currently best solution for the cycles and chains

	// Start the main loop of the RCVF procedure
	while (true) {									// the loop will be terminated by any of the "break" statements

		//---------------------------STEP 4: CREATE A REDUCED ILP MODEL---------------------------	

		sol.RCVFNIter++;												// increment the number of RCVF iterations by 1
		int LBRedILP = 0, UBRedILP = 1000000;							// the LB and UB of the current reduced ILP model

		// Create a new environment and a new model (these are only in scope inside of the while-loop)
		GRBEnv env = GRBEnv();              							// create an environment
		removeLine();                       							// remove Gurobi message
		if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
		GRBModel model = GRBModel(env);									// create a new model

		// Re-declare general variables and linear expressions (temporary scope)
		vector<GRBVar> x, y;											// decision variables (interpretation different per model)
		GRBLinExpr totalWeight = 0;										// objective value
		vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
		vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
		vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
		vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

		// Redeclare model-specific variables and linear expressions
		vector<vector<vector<GRBVar>>> xSubK;							// decision variables modelling for each arc in each subgraph at every possible position whether the arc is selected at that position (1) or not (0)
		vector<vector<vector<GRBLinExpr>>> xOutSubK, xInSubK;			// number of transplants made by each donor/recieved by each recipient in each subgraph at every position

		// Keep track of which variables are activated
		vector<bool> isActiveCycles(numVarsCycles, false);
		vector<bool> isActiveChains(numVarsChains, false);

		// For the cycle model: create binary variables (but only when the reduced cost is high enough) and fill up linear expressions
		if (inst.K > 0) {
			xSubK.resize(inst.nR, vector<vector<GRBVar>>(inst.K + 1));
			int d, r, w;
			xOutSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
			xInSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
			// for l in R, a in Al and k in K(a,l): x_lka = 1 iff arc a is used in position k in subgraph l
			int idxVar = -1;
			for (int l = 0; l < inst.nR; l++) {															// loop over all subgraphs
				for (int k = 1; k <= inst.K; k++) {														// loop over all arc positions
					xSubK[l][k].resize(subgraphsPIEF[l][k].idx2Arc.size());								// reshape the matrix of decision variables
					for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {						// loop over all arcs
						idxVar++;																		// increment the number of cycle variables
						if (RCCycles[idxVar] > T - valLP - EPSILON) {									// only add the variable if its RC is high enough
							isActiveCycles[idxVar] = true;												// activate the variable
							xSubK[l][k][a] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(a));
							if (bestSolCycles.size() > 0) { xSubK[l][k][a].set(GRB_DoubleAttr_Start, bestSolCycles[idxVar]); }		// set a warm start
							d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2];
							totalWeight += w * xSubK[l][k][a];												// update the objective value
							xOut[vertexOrder[d]] += xSubK[l][k][a];											// note: in terms of original vertex index
							xOutSubK[l][d][k] += xSubK[l][k][a];
							xIn[vertexOrder[r]] += xSubK[l][k][a];											// note: in terms of original vertex index
							xInSubK[l][r][k] += xSubK[l][k][a];
						}
					}
				}
			}
		}

		// For the chain model: create binary variables (but only when the RC is high enough) and fill up linear expressions
		if (inst.L > 0) {
			y.resize(chains.size());																// declare a variable for every chain
			// for c in C: y_c = 1 iff chain c is used
			for (int c = 0; c < chains.size(); c++) {												// loop over all chains
				if (RCChains[c] > T - valLP - EPSILON) {											// if the reduced cost is high enough
					isActiveChains[c] = true;														// activate the variable
					y[c] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(c));					// create the variable
					if (bestSolChains.size() > 0) {y[c].set(GRB_DoubleAttr_Start, bestSolChains[c]); }	// set a warm start
					totalWeight += chains[c][1][0] * y[c];											// add the weight to the objective value
					yOut[chains[c][0].front()] += y[c];												// update the out-flow of the NDD
					for (int i = 1; i < chains[c][0].size() - 1; i++) {								// loop over all intermediate RDPs in the current chain
						yOut[chains[c][0][i]] += y[c];												// update their out-flow
						yIn[chains[c][0][i]] += y[c];												// update their in-flow
					}
					yIn[chains[c][0].back()] += y[c];												// update the in-flow of the final RDP
				}
			}
			// add arc weights of arcs to tau
			for (int d = 0; d < inst.nR; d++) {							// loop over all RDPs
				totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);		// add the weight of the corresponding final arc to the objective
			}
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) {			// loop over all NDDs
				totalWeight += inst.wToTau[d] * (1 - yOut[d]);			// add the weight of the corresponding final arc to the objective
			}
		}

		model.update();

		// Skip the rest of the iteration if no additional variables were reactivated
		nVarDeactPrev = sol.RCVFNVarDeact;									// update the number of variables that were deactivated in the prevvious iteration
		sol.RCVFNVarDeact = sol.nVar - model.get(GRB_IntAttr_NumVars);		// find the current number of deactivated variables
		if (sol.RCVFNVarDeact == nVarDeactPrev and earlyTermination == 0) {	// if no additional variables are activated
			if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
			sol.RCVFNSkips++;												// increment the number of skips
			sol.RCVFNIter--;												// undo the increment of the number of (proper) iterations (in which an ILP model is solved)
			T--;															// reduce the target by 1
			if (sol.LB == T) {												// if the best solution reaches the new target, it is optimal
				if (printRCVFLog) { cout << "Target T = " << T << " had already been reached--> solution is optimal.\n"; }
				break;
			}
		}
		else {

			// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// Change a setting so that we terminate once the UB of the current reduced ILP drops below T (using some tolerance)
			if (earlyTermination == 1) { model.set(GRB_DoubleParam_BestBdStop, T - 0.01); }
			else if (earlyTermination == 2) { model.set(GRB_DoubleParam_Cutoff, T - 0.01); }

			//  Constraint for every model 1: every NDD can donate at most once
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
				if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
			}

			// Constraint for every model 2: every RDP can be donated to at most once
			// - note that these are the only contraints linking the cycle and chain component of the model
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
			}

			// For the cycle model: add the model-specific constraints
			if (inst.K > 0) {
				// Additional constraints 1 for PIEF: flow conservation per subgraph per index
				for (int l = 0; l < inst.nR; l++) {						// loop over subgraphs
					for (int r = l + 1; r < inst.nR; r++) {				// loop over RDPs
						for (int k = 1; k <= inst.K - 1; k++) {			// loop over arc indices
							if (xInSubK[l][r][k].size() + xOutSubK[l][r][k + 1].size() > 0) { model.addConstr(xInSubK[l][r][k] == xOutSubK[l][r][k + 1]); }
						}
					}
				}
			}

			// For the chain model: add the model-specific constraints
			// - (none)

			model.update();

			// Change some settings
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_IntParam_Method, 2);														// use the barrier method to solve the LP relaxation
			if (timeLimit - getCPUTime() + start < 0) { sol.timeT = timeLimit; sol.UB = T;  return sol; }	// return an empty solution if the time limit has already been exceeded in pre-processing
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - getCPUTime() + start);				// update the remaining time limit
			// model.getEnv().set(GRB_DoubleParam_BestObjStop, T);												// we can terminate early if a solution with value T is found
			// don't turn off crossover here


			//---------------------------STEP 5: SOLVE THE REDUCED ILP MODEL---------------------------	

			if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving ILP model, iter = " << sol.RCVFNIter << ", time = " << getCPUTime() - start << "s, target value T = " << T << "\n"; }
			if (printRCVFLog) { cout << "Number of variables removed = " << sol.RCVFNVarDeact << " (out of " << sol.nVar << " originally), " << model.get(GRB_IntAttr_NumVars) << " variables remain\n"; }

			model.optimize();

			// Find the best upper bound
			UBRedILP = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (UBRedILP < 0) { UBRedILP = 1000000; }						// if no upper bound was found yet, set it to 1000000

			// If the model is infeasible
			if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
				if (earlyTermination != 0) {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
					T--;							// decrement the target by 1
					continue;
				}
				else {
					sol.feas = false;											// indicate that the instance is infeasible
					break;
				}
			}

			// If a solution has been found within the time limit
			else if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.feas = true;												// indicate that the instance is feasible
				LBRedILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// find the best lower bound

				if (printRCVFLog) { cout << "Value reduced ILP = " << LBRedILP << "\n"; }

				// If the solution is better than the current best one, update the solution (incl. the warm start)
				if (LBRedILP >= sol.LB) {

					// Set the new best LB
					sol.LB = LBRedILP;

					// Reset the vectors containing the best solution found so far
					bestSolCycles.clear(); bestSolCycles.resize(numVarsCycles, 0);
					bestSolChains.clear(); bestSolChains.resize(numVarsChains, 0);

					// Decompose the solution into the selected cycles and chains
					sol.cyclesAndChains.clear();
					sol.cyclesAndChains.resize(2);

					// First consider the cycles
					if (inst.K > 0) {
						vector<vector<int>> donationPerDonor(inst.nR);								// array containing [r, w_dr] for each donor d that donates
						int d, r, w;
						int varIdx = -1;
						for (int l = 0; l < inst.nR; l++) {											// loop over all subgraphs
							for (int k = 1; k <= inst.K; k++) {										// loop over all position indices
								for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {		// loop over all arcs
									varIdx++;														// increment the variable index
									if (isActiveCycles[varIdx]) {									// if the corresponding variable is active
										xval = ceil(xSubK[l][k][a].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
										if (xval == 1) {												// if the arc is selected
											d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2]; // decompose the arc
											donationPerDonor[vertexOrder[d]] = { vertexOrder[r], w };	// add the arc to the list of donations
											bestSolCycles[varIdx] = 1;									// save the solution
										}
									}
								}
							}
						}

						vector<vector<int>> cycle;					// array containing the current cycle and its weight
						for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
							if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
								cycle = { {s}, {0} };				// start the cycle
								d = s;								// initialize the current donor
								while (true) {						// keep looping until the cycle is closed
									r = donationPerDonor[d][0]; w = donationPerDonor[d][1];	// find recipient and weight
									donationPerDonor[d].clear();	// remove the used arc
									cycle[1][0] += w;				// add the arc's weight to the cycle
									if (r != s) {					// if we're not back at the start
										cycle[0].push_back(r);		// add the recipient to the cycle
										d = r;						// the new donor is the recipient's donor
									}
									else { break; }					// break once the cycle is closed
								}
								sol.cyclesAndChains[0].push_back(cycle);	// add the completed cycle
							}
						}
					}

					// Second consider the chains
					if (inst.L > 0) {
						for (int c = 0; c < chains.size(); c++) {												// loop over all chains
							if (isActiveChains[c]) {															// if the variable was activated
								yval = ceil(y[c].get(GRB_DoubleAttr_X) - EPSILON);								// find variable value
								if (yval == 1) {																// if the chain is selected
									sol.cyclesAndChains[1].push_back(chains[c]);								// add the chain to the solution
									sol.cyclesAndChains[1].back()[1][0] += inst.wToTau[chains[c][0].back()];	// also add the weight of the final arc
									bestSolChains[c] = 1;														// save the solution
								}
							}
						}
						// Also add trivial n-tau chains for unused NDDs n
						for (int d = inst.nR; d < inst.nR + inst.nN; d++) {						// loop over all NDDs
							if (ceil(yOut[d].getValue() - EPSILON) == 0) {						// if its out-flow is 0
								sol.cyclesAndChains[1].push_back({ {d}, {inst.wToTau[d]} });	// add the corresponding trivial chain
							}
						}
					}
				}
			}

			//---------------------------STEP 6: CONCLUDE ABOUT THE NEXT STEP BASED ON THE SOLUTION---------------------------	

			// If the reduced ILP was solved to optimality
			if (LBRedILP == UBRedILP) {

				// If the target has been reached: the solution is optimal and we may terminate
				if (LBRedILP >= T) {
					if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
					break;
				}

				// If no variables were deactivated: the solution is optimal and we may terminate
				else if (sol.RCVFNVarDeact == 0) {
					if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
					T = sol.LB;				// update the best upper bound
					break;
				}

				// If the target was not reached, but a feasible solution with value T - 1 was found: the solution is also optimal and we may terminate
				else if (LBRedILP >= T - 1) {
					sol.RCVFNSkips++;				// increment the number of skips
					T--;					// decrement the best upper bound by 1
					if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T << " was reached --> solution is optimal.\n"; }
					break;
				}

				// Otherwise: decrease T by 1 and try again
				else {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << "\n"; }
					T--;							// decrement the target by 1
				}
			}

			// If we terminated early because the UB of the reduced ILP dropped below the target 
			else if (model.get(GRB_IntAttr_Status) == GRB_USER_OBJ_LIMIT or model.get(GRB_IntAttr_Status) == GRB_CUTOFF) {
				if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
				T--;							// decrement the target by 1
			}

			else { // If we ran out of time
				cout << "Time limit exceeded --> return the best solution found so far\n";
				break;
			}
		}
	}

	// Save the final upper bound
	sol.UB = T;

	// Set the optimality attribute
	if (sol.LB == sol.UB) { sol.opt = true; }

	// Save the total time
	sol.timeT = getCPUTime() - start;

	// Return the solution
	return sol;
}

Solution solvePIEFPIEFusingRCVF(const Instance& inst, double timeLimit, bool printRCVFLog, bool printGurobiLog) {
	// This function is used to solve PIEF-CYCLE + PIEF-CHAIN using RCVF

	//---------------------------START OF CODE---------------------------

	// Create a solution object
	Solution sol; sol.modelCycle = "PIEF-CYCLE"; sol.modelChain = "PIEF-CHAIN"; sol.RCVFUsed = true;
	sol.modelFull = sol.modelCycle + " + " + sol.modelChain + " (RCVF)";

	// Start the timer
	double start = getCPUTime();

	// Decide whether to use the early termination strategy
	int earlyTermination = 1;

	//---------------------------STEP 1: CREATE A MODEL FOR THE LP RELAXATION---------------------------

	// Create an environment and a model
	GRBEnv env = GRBEnv();              							// create an environment
	removeLine();                       							// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);									// create a new model

	// Declare general variables and linear expressions
	vector<GRBVar> x, y;											// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;										// objective value
	vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

	// Declare model-specific variables and linear expressions
	vector<vector<vector<GRBVar>>> xSubK;							// decision variables modelling for each arc in each subgraph at every possible position whether the arc is selected at that position (1) or not (0)
	vector<vector<vector<GRBLinExpr>>> xOutSubK, xInSubK;			// number of transplants made by each donor/recieved by each recipient in each subgraph at every position
	vector<vector<GRBVar>> yL;										// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
	vector<vector<GRBLinExpr>> yOutL, yInL;							// number of transplants made by each donor/recieved by each recipient at every position

	// Declare other model-specific objects
	vector<vector<Subgraph>> subgraphsPIEF;							// the reduced arc sets for each (sorted) subgraph G^1,...,G^|R| for each arc position 1,...,K
	vector<Subgraph> subgraphPIEFCHAIN;								// the reduced arc sets for each arc position 1,...,L-1

	// Sort the vertices according to descending total degree
	vector<int> vertexOrder = findVertexOrder(inst, false);							// vertexOrder[i] gives the original index of the vertex that is on index i after sorting
	vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
	changeOrder(inst, vertexOrder, ANew, wToTauNew);

	// Create other model-specific objects
	subgraphsPIEF.resize(inst.nR);								// declare R subgraphs for PIEF
	for (int l = 0; l < inst.nR; l++) {							// loop over all subgraphs
		subgraphsPIEF[l] = createSubgraphPIEFCYCLE(inst, ANew, l);	// create the subgraph
	}
	subgraphPIEFCHAIN = createSubgraphPIEFCHAIN(inst);			// create the position-indexed graph for PIEF-CHAIN

	// For the cycle model: create (continuous) variables and fill up linear expressions
	int numVarsCycles = 0;
	if (inst.K > 0) {
		xSubK.resize(inst.nR, vector<vector<GRBVar>>(inst.K + 1));
		int d, r, w;
		xOutSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
		xInSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
		// for l in R, a in Al and k in K(a,l): x_lka = 1 iff arc a is used in position k in subgraph l
		for (int l = 0; l < inst.nR; l++) {															// loop over all subgraphs
			for (int k = 1; k <= inst.K; k++) {														// loop over all arc positions
				xSubK[l][k].resize(subgraphsPIEF[l][k].idx2Arc.size());								// reshape the matrix of decision variables
				for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {						// loop over all arcs
					numVarsCycles++;																// increment the number of cycle variables
					xSubK[l][k][a] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "x_" + to_string(a));
					d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2];
					totalWeight += w * xSubK[l][k][a];												// update the objective value
					xOut[vertexOrder[d]] += xSubK[l][k][a];											// note: in terms of original vertex index
					xOutSubK[l][d][k] += xSubK[l][k][a];
					xIn[vertexOrder[r]] += xSubK[l][k][a];											// note: in terms of original vertex index
					xInSubK[l][r][k] += xSubK[l][k][a];
				}
			}
		}
	}

	// For the chain model: create (continuous) variables and fill up linear expressions
	int numVarsChains = 0;
	if (inst.L > 0) {
		yL.resize(inst.L); // seperate set of variables for all positions
		int d, r, w;
		yOutL.resize(inst.nR + inst.nN, vector<GRBLinExpr>(inst.L, 0));
		yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
		// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
		for (int k = 1; k <= inst.L - 1; k++) {
			yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
			for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
				yL[k][a] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "y_" + to_string(a));
				numVarsChains++;
				d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
				totalWeight += w * yL[k][a];
				yOut[d] += yL[k][a];
				yOutL[d][k] += yL[k][a];
				yIn[r] += yL[k][a];
				yInL[r][k] += yL[k][a];
			}
		}

		// add arc weights of arcs to tau													// if chains are allowed
		for (int d = 0; d < inst.nR; d++) {									// loop over all RDPs
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);				// add the weight of the corresponding final arc to the objective
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {					// loop over all NDDs
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);					// add the weight of the corresponding final arc to the objective
		}
	}

	model.update();

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	model.setObjective(totalWeight, GRB_MAXIMIZE);

	//  Constraint for every model 1: every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) {										// loop over NDDs
		if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) {														// loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
	}

	// For the cycle model: add the model-specific constraints
	if (inst.K > 0) {
		// Additional constraints 1 for PIEF: flow conservation per subgraph per index
		for (int l = 0; l < inst.nR; l++) {						// loop over subgraphs
			for (int r = l + 1; r < inst.nR; r++) {				// loop over RDPs
				for (int k = 1; k <= inst.K - 1; k++) {			// loop over arc indices
					if (xInSubK[l][r][k].size() + xOutSubK[l][r][k + 1].size() > 0) { model.addConstr(xInSubK[l][r][k] == xOutSubK[l][r][k + 1]); }
				}
			}
		}
	}

	// For the chain model: add the model-specific constraints
	if (inst.L > 0) {
		// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
		for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
			for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
				if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
			}
		}
	}

	model.update();

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, 2);										// use the barrier method to solve the LP relaxation 
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; }			// return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);			// update the remaining time limit
	model.getEnv().set(GRB_IntParam_Crossover, 0);									// crossover can be turned off

	//---------------------------STEP 2: SOLVE THE LP RELAXATION AND FIND ITS VALUE AND THE REDUCED COSTS---------------------------

	// Solve the LP relaxation model
	if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
	if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving initial LP...\n"; }
	model.optimize();

	// Find the LP relaxation value
	double valLP = model.get(GRB_DoubleAttr_ObjVal);
	sol.LPrel = valLP;
	if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

	// Save the model size
	sol.nVar = model.get(GRB_IntAttr_NumVars);			// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);	// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);			// number of non-zero coefficients

	// Terminate if the time limit has already been exceeded after solving the LP relaxation and simply return an empty solution
	sol.timeLP = getCPUTime() - start;
	if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }

	// Find the reduced costs of the variables
	vector<double> RCCycles(numVarsCycles, 0), RCChains(numVarsChains, 0);
	GRBVar* vars = model.getVars();
	for (int i = 0; i < numVarsCycles; i++) {
		if (vars[i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCCycles[i] = vars[i].get(GRB_DoubleAttr_RC);
		}
	}
	for (int i = 0; i < numVarsChains; i++) {
		if (vars[numVarsCycles + i].get(GRB_DoubleAttr_X) < EPSILON) {
			RCChains[i] = vars[numVarsCycles + i].get(GRB_DoubleAttr_RC);
		}
	}

	// Clear up the memory
	model.reset(1);
	model.update();

	//---------------------------STEP 3: INITIALIZE THE REST OF THE RCVF PROCEDURE---------------------------

	// Set the initial target
	int T = floor(valLP + EPSILON);

	// Some other declarations before while-loop
	int xval, yval;														// the value of the currently considered x- or y-variable
	sol.RCVFNIter = 0;													// the number of RCVF iterations
	sol.RCVFNVarDeact = sol.nVar + 1;									// the number of variables that are deactivated (initialized to impossibly high value)
	int nVarDeactPrev;													// the number of variables that were deactivated in the previous iteration
	vector<int> bestSolCycles, bestSolChains;							// the currently best solution for the cycles and chains

	// Start the main loop of the RCVF procedure
	while (true) {									// the loop will be terminated by any of the "break" statements

		//---------------------------STEP 4: CREATE A REDUCED ILP MODEL---------------------------	

		sol.RCVFNIter++;												// increment the number of RCVF iterations by 1
		int LBRedILP = 0, UBRedILP = 1000000;							// the LB and UB of the current reduced ILP model

		// Create a new environment and a new model (these are only in scope inside of the while-loop)
		GRBEnv env = GRBEnv();              							// create an environment
		removeLine();                       							// remove Gurobi message
		if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
		GRBModel model = GRBModel(env);									// create a new model

		// Re-declare general variables and linear expressions (temporary scope)
		vector<GRBVar> x, y;											// decision variables (interpretation different per model)
		GRBLinExpr totalWeight = 0;										// objective value
		vector<GRBLinExpr> xOut(inst.nR, 0);							// number of transplants made by each directed donor using a cycle
		vector<GRBLinExpr> xIn(inst.nR, 0);								// number of transplants made to each recipient using a cycle
		vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);					// number of transplants made by each donor using a chain
		vector<GRBLinExpr> yIn(inst.nR, 0);								// number of transplants made to each recipient using a chain

		// Redeclare model-specific variables and linear expressions
		vector<vector<vector<GRBVar>>> xSubK;							// decision variables modelling for each arc in each subgraph at every possible position whether the arc is selected at that position (1) or not (0)
		vector<vector<vector<GRBLinExpr>>> xOutSubK, xInSubK;			// number of transplants made by each donor/recieved by each recipient in each subgraph at every position
		vector<vector<GRBVar>> yL;										// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
		vector<vector<GRBLinExpr>> yOutL, yInL;							// number of transplants made by each donor/recieved by each recipient at every position


		// Keep track of which variables are activated
		vector<bool> isActiveCycles(numVarsCycles, false);
		vector<bool> isActiveChains(numVarsChains, false);

		// For the cycle model: create binary variables (but only when the reduced cost is high enough) and fill up linear expressions
		if (inst.K > 0) {
			xSubK.resize(inst.nR, vector<vector<GRBVar>>(inst.K + 1));
			int d, r, w;
			xOutSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
			xInSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
			// for l in R, a in Al and k in K(a,l): x_lka = 1 iff arc a is used in position k in subgraph l
			int idxVar = -1;
			for (int l = 0; l < inst.nR; l++) {															// loop over all subgraphs
				for (int k = 1; k <= inst.K; k++) {														// loop over all arc positions
					xSubK[l][k].resize(subgraphsPIEF[l][k].idx2Arc.size());								// reshape the matrix of decision variables
					for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {						// loop over all arcs
						idxVar++;																		// increment the number of cycle variables
						if (RCCycles[idxVar] > T - valLP - EPSILON) {									// only add the variable if its RC is high enough
							isActiveCycles[idxVar] = true;												// activate the variable
							xSubK[l][k][a] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(a));
							if (bestSolCycles.size() > 0) { xSubK[l][k][a].set(GRB_DoubleAttr_Start, bestSolCycles[idxVar]); }		// set a warm start
							d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2];
							totalWeight += w * xSubK[l][k][a];												// update the objective value
							xOut[vertexOrder[d]] += xSubK[l][k][a];											// note: in terms of original vertex index
							xOutSubK[l][d][k] += xSubK[l][k][a];
							xIn[vertexOrder[r]] += xSubK[l][k][a];											// note: in terms of original vertex index
							xInSubK[l][r][k] += xSubK[l][k][a];
						}
					}
				}
			}
		}

		// For the chain model: create binary variables (but only when the RC is high enough) and fill up linear expressions
		if (inst.L > 0) {
			yL.resize(inst.L); // seperate set of variables for all positions
			int d, r, w;
			yOutL.resize(inst.nR + inst.nN, vector<GRBLinExpr>(inst.L, 0));
			yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
			int varIdx = -1;
			// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
			for (int k = 1; k <= inst.L - 1; k++) {
				yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
				for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
					varIdx++;
					if (RCChains[varIdx] > T - valLP - EPSILON) {
						yL[k][a] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(a));
						isActiveChains[varIdx] = true;
						if (bestSolChains.size() > 0) { yL[k][a].set(GRB_DoubleAttr_Start, bestSolChains[varIdx]); }	// set a warm start
						d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
						totalWeight += w * yL[k][a];
						yOut[d] += yL[k][a];
						yOutL[d][k] += yL[k][a];
						yIn[r] += yL[k][a];
						yInL[r][k] += yL[k][a];
					}
				}
			}
			// add arc weights of arcs to tau
			for (int d = 0; d < inst.nR; d++) {							// loop over all RDPs
				totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);		// add the weight of the corresponding final arc to the objective
			}
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) {			// loop over all NDDs
				totalWeight += inst.wToTau[d] * (1 - yOut[d]);			// add the weight of the corresponding final arc to the objective
			}
		}

		model.update();

		// Skip the rest of the iteration if no additional variables were reactivated
		nVarDeactPrev = sol.RCVFNVarDeact;									// update the number of variables that were deactivated in the prevvious iteration
		sol.RCVFNVarDeact = sol.nVar - model.get(GRB_IntAttr_NumVars);		// find the current number of deactivated variables
		if (sol.RCVFNVarDeact == nVarDeactPrev and earlyTermination == 0) {	// if no additional variables are activated
			if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
			sol.RCVFNSkips++;												// increment the number of skips
			sol.RCVFNIter--;												// undo the increment of the number of (proper) iterations (in which an ILP model is solved)
			T--;															// reduce the target by 1
			if (sol.LB == T) {												// if the best solution reaches the new target, it is optimal
				if (printRCVFLog) { cout << "Target T = " << T << " had already been reached--> solution is optimal.\n"; }
				break;
			}
		}
		else {

			// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// Change a setting so that we terminate once the UB of the current reduced ILP drops below T (using some tolerance)
			if (earlyTermination == 1) { model.set(GRB_DoubleParam_BestBdStop, T - 0.01); }
			else if (earlyTermination == 2) { model.set(GRB_DoubleParam_Cutoff, T - 0.01); }

			//  Constraint for every model 1: every NDD can donate at most once
			for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
				if (yOut[d].size() > 0) { model.addConstr(yOut[d] <= 1); }
			}

			// Constraint for every model 2: every RDP can be donated to at most once
			// - note that these are the only contraints linking the cycle and chain component of the model
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + yIn[r].size() > 0) { model.addConstr(xIn[r] + yIn[r] <= 1); }
			}

			// For the cycle model: add the model-specific constraints
			if (inst.K > 0) {
				// Additional constraints 1 for PIEF: flow conservation per subgraph per index
				for (int l = 0; l < inst.nR; l++) {						// loop over subgraphs
					for (int r = l + 1; r < inst.nR; r++) {				// loop over RDPs
						for (int k = 1; k <= inst.K - 1; k++) {			// loop over arc indices
							if (xInSubK[l][r][k].size() + xOutSubK[l][r][k + 1].size() > 0) { model.addConstr(xInSubK[l][r][k] == xOutSubK[l][r][k + 1]); }
						}
					}
				}
			}

			// For the chain model: add the model-specific constraints
			if (inst.L > 0) {
				// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
				for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
					for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
						if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
					}
				}
			}

			model.update();

			// Change some settings
			model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
			model.getEnv().set(GRB_IntParam_Threads, 1);
			model.getEnv().set(GRB_IntParam_Method, 2);														// use the barrier method to solve the LP relaxation
			if (timeLimit - getCPUTime() + start < 0) { sol.timeT = timeLimit; sol.UB = T;  return sol; }	// return an empty solution if the time limit has already been exceeded in pre-processing
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - getCPUTime() + start);				// update the remaining time limit
			// model.getEnv().set(GRB_DoubleParam_BestObjStop, T);												// we can terminate early if a solution with value T is found
			// don't turn off crossover here


			//---------------------------STEP 5: SOLVE THE REDUCED ILP MODEL---------------------------	

			if (printRCVFLog) { cout << "\n---------------------------------------\nStart solving ILP model, iter = " << sol.RCVFNIter << ", time = " << getCPUTime() - start << "s, target value T = " << T << "\n"; }
			if (printRCVFLog) { cout << "Number of variables removed = " << sol.RCVFNVarDeact << " (out of " << sol.nVar << " originally), " << model.get(GRB_IntAttr_NumVars) << " variables remain\n"; }

			model.optimize();

			// Find the best upper bound
			UBRedILP = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (UBRedILP < 0) { UBRedILP = 1000000; }						// if no upper bound was found yet, set it to 1000000

			// If the model is infeasible
			if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
				if (earlyTermination != 0) {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
					T--;							// decrement the target by 1
					continue;
				}
				else {
					sol.feas = false;											// indicate that the instance is infeasible
					break;
				}
			}

			// If a solution has been found within the time limit
			else if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.feas = true;												// indicate that the instance is feasible
				LBRedILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// find the best lower bound

				if (printRCVFLog) { cout << "Value reduced ILP = " << LBRedILP << "\n"; }

				// If the solution is better than the current best one, update the solution (incl. the warm start)
				if (LBRedILP >= sol.LB) {

					// Set the new best LB
					sol.LB = LBRedILP;

					// Reset the vectors containing the best solution found so far
					bestSolCycles.clear(); bestSolCycles.resize(numVarsCycles, 0);
					bestSolChains.clear(); bestSolChains.resize(numVarsChains, 0);

					// Decompose the solution into the selected cycles and chains
					sol.cyclesAndChains.clear();
					sol.cyclesAndChains.resize(2);

					// First consider the cycles
					if (inst.K > 0) {
						vector<vector<int>> donationPerDonor(inst.nR);								// array containing [r, w_dr] for each donor d that donates
						int d, r, w;
						int varIdx = -1;
						for (int l = 0; l < inst.nR; l++) {											// loop over all subgraphs
							for (int k = 1; k <= inst.K; k++) {										// loop over all position indices
								for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {		// loop over all arcs
									varIdx++;														// increment the variable index
									if (isActiveCycles[varIdx]) {									// if the corresponding variable is active
										xval = ceil(xSubK[l][k][a].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
										if (xval == 1) {												// if the arc is selected
											d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2]; // decompose the arc
											donationPerDonor[vertexOrder[d]] = { vertexOrder[r], w };	// add the arc to the list of donations
											bestSolCycles[varIdx] = 1;									// save the solution
										}
									}
								}
							}
						}

						vector<vector<int>> cycle;					// array containing the current cycle and its weight
						for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
							if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
								cycle = { {s}, {0} };				// start the cycle
								d = s;								// initialize the current donor
								while (true) {						// keep looping until the cycle is closed
									r = donationPerDonor[d][0]; w = donationPerDonor[d][1];	// find recipient and weight
									donationPerDonor[d].clear();	// remove the used arc
									cycle[1][0] += w;				// add the arc's weight to the cycle
									if (r != s) {					// if we're not back at the start
										cycle[0].push_back(r);		// add the recipient to the cycle
										d = r;						// the new donor is the recipient's donor
									}
									else { break; }					// break once the cycle is closed
								}
								sol.cyclesAndChains[0].push_back(cycle);	// add the completed cycle
							}
						}
					}

					// Second consider the chains
					if (inst.L > 0) {
						vector<vector<int>> donationPerDonor(inst.nR + inst.nN);		// array containing [r, w_dr] for each donor d that donates
						int d, r, w;
						int varIdx = -1;
						for (int k = 1; k <= inst.L - 1; k++) {								// loop over all arc positions
							for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {	// loop over all arcs
								varIdx++;
								if (isActiveChains[varIdx]) {
									yval = ceil(yL[k][a].get(GRB_DoubleAttr_X) - EPSILON);		// find variable value
									if (yval == 1) {											// if the arc is selected
										d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2]; // decompose the arc
										donationPerDonor[d] = { r, w };							// add the arc to the list of donations
										bestSolChains[varIdx] = 1;								// save the solution
									}
								}
							}
						}
						// Also find the donations to tau
						if (inst.L > 1) {
							for (int d = 0; d < inst.nR; d++) {
								if (ceil(yIn[d].getValue() - EPSILON) - ceil(yOut[d].getValue() - EPSILON) == 1) {
									donationPerDonor[d] = { -1, inst.wToTau[d] };
								}
							}
						}
						for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
							if (ceil(yOut[d].getValue() - EPSILON) == 0) {
								donationPerDonor[d] = { -1, inst.wToTau[d] };
							}
						}
						// Find the chains
						vector<vector<int>> chain;					// array containing the current chain and its weight
						for (int s = inst.nR; s < inst.nR + inst.nN; s++) { // loop over all non-directed donors
							if (donationPerDonor[s].size() > 0) {	// if the non-directed donor actually donated
								chain = { {s}, {0} };				// start the chain
								d = s;								// initialize the current donor
								while (true) {						// keep looping until the cycle is closed
									r = donationPerDonor[d][0]; w = donationPerDonor[d][1];	// find recipient and weight
									donationPerDonor[d].clear();	// remove the used arc
									chain[1][0] += w;				// add the arc's weight to the chain
									if (r != -1) {					// if we're not at tau (-1)
										chain[0].push_back(r);		// add the recipient to the chain
										d = r;						// the new donor is the recipient's donor
									}
									else { break; }					// break once tau (-1) has been reached
								}
								sol.cyclesAndChains[1].push_back(chain);	// add the completed chain
							}
						}
					}
				}
			}

			//---------------------------STEP 6: CONCLUDE ABOUT THE NEXT STEP BASED ON THE SOLUTION---------------------------	

			// If the reduced ILP was solved to optimality
			if (LBRedILP == UBRedILP) {

				// If the target has been reached: the solution is optimal and we may terminate
				if (LBRedILP >= T) {
					if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
					break;
				}

				// If no variables were deactivated: the solution is optimal and we may terminate
				else if (sol.RCVFNVarDeact == 0) {
					if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
					T = sol.LB;				// update the best upper bound
					break;
				}

				// If the target was not reached, but a feasible solution with value T - 1 was found: the solution is also optimal and we may terminate
				else if (LBRedILP >= T - 1) {
					sol.RCVFNSkips++;				// increment the number of skips
					T --;							// decrement the best upper bound by 1
					if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T << " was reached --> solution is optimal.\n"; }
					break;
				}

				// Otherwise: decrease T by 1 and try again
				else {
					if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << "\n"; }
					T--;							// decrement the target by 1
				}
			}

			// If we terminated early because the UB of the reduced ILP dropped below the target 
			else if (model.get(GRB_IntAttr_Status) == GRB_USER_OBJ_LIMIT or model.get(GRB_IntAttr_Status) == GRB_CUTOFF) {
				if (printRCVFLog) { cout << "Current value of T = " << T << " impossible --> decrease T to " << T - 1 << " (detected early)\n"; }
				T--;							// decrement the target by 1
			}

			else { // If we ran out of time
				cout << "Time limit exceeded --> return the best solution found so far\n";
				break;
			}
		}
	}

	// Save the final upper bound
	sol.UB = T;

	// Set the optimality attribute
	if (sol.LB == sol.UB) { sol.opt = true; }

	// Save the total time
	sol.timeT = getCPUTime() - start;

	// Return the solution
	return sol;
}
