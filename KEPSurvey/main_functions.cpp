#include "main_functions.h"

//---------------------------MAIN FUNCTION---------------------------
Solution solveKEP(const Instance& inst, string modelCycle, string modelChain, string modelHybrid, double timeLimit, bool printGurobiLog, bool solveLP, bool useRCVF, bool printRCVFLog, bool solveHierarchically) {
	// This is the main function, which solves a given KEP instance using a specified model.

	//---------------------------GENERAL INITIALIZATIONS---------------------------

	// Initialize Solution object
	Solution sol; sol.modelCycle = modelCycle; sol.modelChain = modelChain; sol.modelHybrid = modelHybrid; sol.RCVFUsed = useRCVF;
	if (modelHybrid != "" && useRCVF) { sol.modelFull = modelHybrid + " (RCVF)"; }
	else if (modelHybrid != "" && not(useRCVF)) { sol.modelFull = modelHybrid; }
	else if (modelHybrid == "" && useRCVF) { sol.modelFull = modelCycle + " + " + modelChain + " (RCVF)"; }
	else { sol.modelFull = modelCycle + " + " + modelChain; }

	sol.solvedHierarchically = solveHierarchically;				// indicate that the problem is solved hierarchically
	
	// Select the LP method: 0 = primal simplex, 1 = dual simplex, 2 = barrier
	int methodLP = selectLPMethod(sol);

	// Set the starting time
	double start = getCPUTime();

	// Create a model
	GRBEnv env = GRBEnv();              					// create an environment
	removeLine();                       					// remove Gurobi message
	if (!printGurobiLog) { env.set(GRB_IntParam_LogToConsole, 0); }	// turn off console output
	GRBModel model = GRBModel(env);								// create a new model

	// General declarations
	vector<GRBVar> x, y;										// decision variables (interpretation different per model)
	GRBLinExpr totalWeight = 0;									// objective value
	GRBLinExpr numberOfTransplants = 0;							// the total number of transplants
	vector<GRBLinExpr> xOut(inst.nR, 0);						// number of transplants made by each directed donor using a cycle
	vector<GRBLinExpr> xIn(inst.nR, 0);							// number of transplants made to each recipient using a cycle
	vector<GRBLinExpr> yOut(inst.nR + inst.nN, 0);				// number of transplants made by each donor using a chain
	vector<GRBLinExpr> yIn(inst.nR, 0);							// number of transplants made to each recipient using a chain
	Callback cb = Callback(inst, sol.modelCycle, sol.modelChain, sol.modelHybrid);	// callbacks used for implementing constraint generation/lazy constraints
	vector<int> vertexOrder;									// vertexOrder[i] gives the original index of the vertex that is on index i after sorting (used for HCF-CYCLE, EEF-CYCLE and PIEF-CYCLE)
	int boundRCVF = -1;											// the best bound found by RCVF if time has run out

	// Declarations for CF-CYCLE
	vector<vector<vector<int>>> cycles;							// all cycles of size at most K 
	vector<vector<vector<int>>> nonDomCycles;					// all non-dominated cycles of size at most K 

	// Declarations for HCF-CYCLE
	vector<vector<vector<int>>> halfCycles;						// all half-cycles based on max cycle length K 
	vector<vector<GRBLinExpr>> halfCyclesPerStartEnd;			// halfCyclesPerStartEnd[s][e] gives the sum of all half-cycle variables starting at s and ending at e

	// Declarations for EF-CYCLE variants (EF-CYCLE, EF-CYCLE-CG, EF2-CYCLE, EF2-CYCLE-CG)
	Subgraph subgraphEF;										// the reduced arc set
	vector<vector<int>> minInfeasiblePaths;						// all minimal infeasible paths (consisting of K+1 vertices and K arcs)
	vector<GRBLinExpr> arcsUsedOnMinInfeasiblePath;				// the sum of all variables used per minimal infeasible path
	vector<vector<int>> maxFeasiblePaths;						// all maximally cardinality feasible paths (consisting of K vertices and K-1 arcs) (EF2-CYCLE only)
	vector<GRBLinExpr> arcsUsedOnMaxFeasiblePath;				// the sum of all variables used per maximally cardinality feasible path (EF2-CYCLE only)

	// Declarations for EEF-CYCLE
	// see vertexOrder under General declarations
	vector<Subgraph> subgraphsEEF;								// the reduced arc sets for each (sorted) subgraph G^1,...,G^|R|
	vector<vector<GRBVar>> xSub;								// decision variables modelling for each arc in each subgraph whether the arc is selected (1) or not (0)
	vector<GRBLinExpr> arcsPerSubgraph;							// the sum of all variables used per subgraph
	vector<vector<GRBLinExpr>> xOutSub, xInSub;					// number of transplants made by each donor/recieved by each recipient in each subgraph

	// Declarations for PIEF(-REDUCED)-CYCLE
	// see vertexOrder under "Declarations for HCF-CYCLE"
	vector<vector<Subgraph>> subgraphsPIEF;						// the reduced arc sets for each (sorted) subgraph G^1,...,G^|R| for each arc position 1,...,K
	vector<vector<vector<GRBVar>>> xSubK;						// decision variables modelling for each arc in each subgraph at every possible position whether the arc is selected at that position (1) or not (0)
	vector<vector<vector<GRBLinExpr>>> xOutSubK, xInSubK;		// number of transplants made by each donor/recieved by each recipient in each subgraph at every position
	vector<vector<int>> arcToWeight;							// a mapping of arcs (two-index representation) to the corresponding weights (required for PIEF-CYCLE-REDUCED only)

	// Declarations for CF-CHAIN
	vector<vector<vector<int>>> chains;							// all chains of size at most L

	// Declarations for HCF-CHAIN
	vector<vector<vector<int>>> halfChains;						// all half-chains based on max chain length L 
	vector<GRBLinExpr> firstHalfChainsPerEnd;					// the number of first half-chains ending at each vertex
	vector<GRBLinExpr> secondHalfChainsPerStart;				// the number of second half-chains starting at each vertex
	vector<GRBLinExpr> longFirstHalfChainsPerEnd;				// the number of first half-chains of maximum length ending at each vertex

	// Declarations for EF-CHAIN-variants (EF-CHAIN-EXP, EF-CHAIN-EXP-CG, EF-CHAIN-CUTSET, EF-CHAIN-CUTSET-CG, EF-CHAIN-MTZ)
	Subgraph subgraphEFCHAIN;									// the reduced arc set
	vector<vector<int>> minInfeasibleChains;					// all minimal infeasible chains (consisting of L vertices and L-1 arcs in R) (STANDARD(2)/CUTSET(2) only)
	vector<GRBLinExpr> arcsUsedOnMinInfeasibleChain;			// the sum of all variables used per minimal infeasible chain (STANDARD(2)/CUTSET(2) only)
	vector<vector<int>> shortCycles;							// all cycles of size less than L (STANDARD(2)/CUTSET(2) only)
	vector<GRBLinExpr> arcsUsedOnShortCycles;					// the sum of all variables relating to arcs present in a short cycle per short cycle (STANDARD(2) only)
	vector<GRBLinExpr> arcsIntoCycles;							// the sum of all variables relating to arcs intro a short cycle per short cycle (CUTSET(2) only)
	vector<GRBVar> t;											// the time-stamp variables for all RDPs (MTZ only, also used for EEF-CHAIN-MTZ)

	// Declarations for EEF-CHAIN-variants (EEF-CHAIN-EXP, EEF-CHAIN-EXP-CG, EEF-CHAIN-CUTSET, EEF-CHAIN-CUTSET-CG, EEF-CHAIN-MTZ)
	vector<Subgraph> subgraphsEEFCHAIN;							// the reduced arc sets for each subgraph G^1,...,G^|N|
	vector<vector<GRBVar>> ySub;								// decision variables modelling for each arc in each subgraph whether the arc is selected (1) or not (0) 
	vector<vector<GRBLinExpr>> yOutSub, yInSub;					// number of transplants made by each donor/recieved by each recipient in each subgraph
	vector<GRBLinExpr> arcsPerSubgraphChains;					// the sum of all variables used per subgraph (not used by MTZ)
	vector<GRBLinExpr> arcsSummedSubgraphs;						// for every arc, the sum of all variables relating to that arc over all subgraphs containing that arc (MTZ only)
	// see t under "Declarations for EF-CHAIN-variants"

	// Declarations for PIEF-CHAIN
	vector<Subgraph> subgraphPIEFCHAIN;							// the reduced arc sets for each arc position 1,...,L-1
	vector<vector<GRBVar>> yL;									// decision variables modelling for each arc and every possible position whether the arc is selected at that position (1) or not (0) 
	vector<vector<GRBLinExpr>> yOutL, yInL;						// number of transplants made by each donor/recieved by each recipient at every position

	// Declarations for EF-HYBRID-variants (EF-HYBRID, EF-HYBRID-CG, EF-HYBRID-IMPROVED, EF-HYBRID-CG)
	Subgraph subgraphEFHYBRID;									// the reduced arc set
	vector<vector<int>> longCycles;								// the set of cycles of length more than K
	vector<GRBLinExpr> arcsUsedOnLongCycle;						// the sum of all variables per long cycle
	vector<vector<int>> mediumCycles;							// the set of cycles of length more than K but less than L
	vector<GRBLinExpr> arcsUsedOnMediumCycle;					// the sum of all variables per medium cycle


	//---------------------------DEFINING THE CHOSEN CYCLE MODEL---------------------------
	if (inst.K > 0) { // only define the cycle model when cycles are allowed
		if (sol.modelCycle == "CF-CYCLE") {
			// Find the set of all cycles of length at most K
			cycles = findCycles(inst);
			// printCyclesOrChains(cycles, "Cycles", "cycle");

			x.resize(cycles.size());
			// for c in C: x_c = 1 iff cycle c is used
			for (int c = 0; c < cycles.size(); c++) {
				x[c] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(c));
				totalWeight += cycles[c][1][0] * x[c];
				numberOfTransplants += cycles[c][0].size() * x[c];
				for (const int& r : cycles[c][0]) {		// loop over all recipients in the current cycle
					xIn[r] += x[c];
				}
			}
		}

		else if (sol.modelCycle == "HCF-CYCLE") {
			// First sort the vertices according to descending total degree
			// - vertexOrder[i] gives the original index of the vertex that is on index i after sorting
			vertexOrder = findVertexOrder(inst, false);
			vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
			changeOrder(inst, vertexOrder, ANew, wToTauNew);

			//// Print new ordering
			//printVector(vertexOrder, "vertexOrder");
			//printA(ANew, wToTauNew, "Anew");

			// Construct all half-cycles
			halfCycles = findHalfCycles(inst, ANew);

			// Define the sum of half-cycle variables starting at s and ending at e for each pair s,e
			halfCyclesPerStartEnd.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0));

			int s, e;

			// Make a variable for each half-cycle
			x.resize(halfCycles.size());
			for (int h = 0; h < halfCycles.size(); h++) {
				// for h in H: x_h = 1 iff half-cycle h is used
				x[h] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(h));
				totalWeight += halfCycles[h][1][0] * x[h];	// add the weight to the objective function
				numberOfTransplants += (halfCycles[h][0].size()-1) * x[h];

				// find the inflow of each recipient (in terms of original vertex indices)
				s = halfCycles[h][0].front(); e = halfCycles[h][0].back();
				xIn[vertexOrder[s]] += x[h] / 2;	// the starting vertex counts half
				xIn[vertexOrder[e]] += x[h] / 2;	// the ending vertex counts half
				for (int i = 1; i < halfCycles[h][0].size()-1; i++) { // loop over all intermediate recipients in the current half-cycle
					xIn[vertexOrder[halfCycles[h][0][i]]] += x[h];		// all intermediate vertices count fully
				}

				// add the half-cycle to the sum of half-cycles starting at s and ending at t
				halfCyclesPerStartEnd[s][e] += x[h];
			}
		}

		else if (sol.modelCycle == "EF-CYCLE" || sol.modelCycle == "EF-CYCLE-CG" || sol.modelCycle == "EF-CYCLE-OLD" || sol.modelCycle == "EF-CYCLE-OLD-CG" || sol.modelCycle == "EF-CYCLE-STRONG" || sol.modelCycle == "EF-CYCLE-STRONG-CG") {

			// Reduce A, and create mappings from the 2D-arcs to the arc indices, and vice versa
			subgraphEF = createSubgraphEFCYCLE(inst);
			//printA(subgraphEF.A, inst.wToTau, "reduced A used by EF-CYCLE");
			//cout << "Reduced number of arcs: " << subgraphEF.idx2Arc.size() << " (original: " << inst.nAR << ")\n";

			x.resize(subgraphEF.idx2Arc.size());
			int d, r, w;
			// for a in A: x_a = 1 iff arc a is used
			for (int a = 0; a < subgraphEF.idx2Arc.size(); a++) {
				x[a] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(a));
				d = subgraphEF.idx2Arc[a][0]; r = subgraphEF.idx2Arc[a][1]; w = subgraphEF.idx2Arc[a][2];
				totalWeight += w * x[a];
				numberOfTransplants += (w > 0) * x[a];
				xOut[d] += x[a];
				xIn[r] += x[a];
			}

			// When using the long-cycle elimination constraints based on maximally cardinality feasible paths, we can either use the standard constraints (EF-CYCLE) or strenghthened constraints (EF-CYCLE-STRONG)
			// If no constraint generation is used, all constraints are enumerated
			if (sol.modelCycle == "EF-CYCLE" || sol.modelCycle == "EF-CYCLE-STRONG") {
				// Create the set of maximally cardinality feasible paths (i.e., paths containing K vertices (so K-1 arcs), where vertices are not repeated)
				maxFeasiblePaths = findPaths(inst, subgraphEF.A, inst.K - 1);
				int d, r, a;
				// print2DVector(maxFeasiblePaths, "maxFeasiblePaths", "path");
				arcsUsedOnMaxFeasiblePath.resize(maxFeasiblePaths.size(), 0);
				for (int p = 0; p < maxFeasiblePaths.size(); p++) {
					if (sol.modelCycle == "EF-CYCLE") { // non-lifted version
						for (int i = 0; i < maxFeasiblePaths[p].size() - 1; i++) {
							d = maxFeasiblePaths[p][i];
							r = maxFeasiblePaths[p][i + 1];
							a = subgraphEF.arc2Idx[d][r];
							arcsUsedOnMaxFeasiblePath[p] += x[a];
						}
					}
					else if (sol.modelCycle == "EF-CYCLE-STRONG") { // lifted version
						for (int iD = 0; iD < maxFeasiblePaths[p].size() - 1; iD++) {
							d = maxFeasiblePaths[p][iD];
							for (int iR = iD + 1; iR < maxFeasiblePaths[p].size(); iR++) {
								r = maxFeasiblePaths[p][iR];
								a = subgraphEF.arc2Idx[d][r];
								if (a != -1) { arcsUsedOnMaxFeasiblePath[p] += x[a]; }
							}
						}
					}
					a = subgraphEF.arc2Idx[maxFeasiblePaths[p].back()][maxFeasiblePaths[p].front()];
					if (a != -1) { arcsUsedOnMaxFeasiblePath[p] -= x[a]; } // Note the minus sign here
				}
			}

			// If constraint generation is used, set the required callback attributes
			else if (sol.modelCycle == "EF-CYCLE-CG" || sol.modelCycle == "EF-CYCLE-STRONG-CG") {
				cb.setCycleCallback(subgraphEF, x);
			}


			// Alternatively, we can use the (original) long-cycle elimination constraints based on minimal infeasible paths
			else if (sol.modelCycle == "EF-CYCLE-OLD") {
				// Create the set of minimal infeasible paths (i.e., paths containing K arcs, where vertices are not repeated)
				minInfeasiblePaths = findPaths(inst, subgraphEF.A, inst.K);
				// print2DVector(minInfeasiblePaths, "minInfeasiblePaths", "path");
				arcsUsedOnMinInfeasiblePath.resize(minInfeasiblePaths.size(), 0);
				for (int p = 0; p < minInfeasiblePaths.size(); p++) {
					for (int v = 0; v < inst.K; v++) {
						int d = minInfeasiblePaths[p][v]; int r = minInfeasiblePaths[p][v + 1];
						arcsUsedOnMinInfeasiblePath[p] += x[subgraphEF.arc2Idx[d][r]];
					}
				}
			}
			// If constraint generation is used, set the required callback attributes
			else if (sol.modelCycle == "EF-CYCLE-OLD-CG") {
				cb.setCycleCallback(subgraphEF, x);
			}
		}

		else if (sol.modelCycle == "EEF-CYCLE" || sol.modelCycle == "EEF-CYCLE-OLD" || sol.modelCycle == "EEF-CYCLE-STRONG") {
			// First sort the vertices according to ascending total degree
			// - vertexOrder[i] gives the original index of the vertex that is on index i after sorting
			vertexOrder = findVertexOrder(inst, true);
			vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
			changeOrder(inst, vertexOrder, ANew, wToTauNew);

			//// Print new ordering
			//printVector(vertexOrder, "vertexOrder");
			//printA(ANew, wToTauNew, "Anew");

			// Create all subgraphs
			// double startSubgraphConstruction = getCPUTime();
			subgraphsEEF.resize(inst.nR);
			for (int l = 0; l < inst.nR; l++) {					// loop over all subgraphs
				subgraphsEEF[l] = createSubgraphEEFCYCLE(inst, ANew, l);

				//printA(subgraphsEEF[l].A, wToTauNew, "A_" + to_string(l));
				//print2DVector(subgraphsEEF[l].arc2Idx, "arc2Idx_" + to_string(l), "donor");
				// print2DVector(subgraphsEEF[l].idx2Arc, "idx2Arc_" + to_string(l), "arc");
			}
			//double timeSubgraphConstruction = getCPUTime() - startSubgraphConstruction;
			//cout << "Subgraph construction takes " << timeSubgraphConstruction << "s.\n";

			xSub.resize(inst.nR);
			int d, r, w;
			arcsPerSubgraph.resize(inst.nR, 0);
			xOutSub.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0)); xInSub.resize(inst.nR, vector<GRBLinExpr>(inst.nR, 0));
			// for l in R and a in Al: x_la = 1 iff arc a is used in subgraph l
			for (int l = 0; l < inst.nR; l++) {
				xSub[l].resize(subgraphsEEF[l].idx2Arc.size());
				for (int a = 0; a < subgraphsEEF[l].idx2Arc.size(); a++) {
					xSub[l][a] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(a));
					d = subgraphsEEF[l].idx2Arc[a][0]; r = subgraphsEEF[l].idx2Arc[a][1]; w = subgraphsEEF[l].idx2Arc[a][2];
					totalWeight += w * xSub[l][a];
					numberOfTransplants += (w>0) * xSub[l][a];
					xOut[vertexOrder[d]] += xSub[l][a]; // Note: in terms of original vertex index
					xOutSub[l][d] += xSub[l][a];
					xIn[vertexOrder[r]] += xSub[l][a];  // Note: in terms of original vertex index
					xInSub[l][r] += xSub[l][a];
					arcsPerSubgraph[l] += xSub[l][a];
				}
			}
		}

		else if (sol.modelCycle == "PIEF-CYCLE" || sol.modelCycle == "PIEF-CYCLE-REDUCED") {
			// First sort the vertices according to descending total degree
			// - vertexOrder[i] gives the original index of the vertex that is on index i after sorting

			if (sol.modelCycle == "PIEF-CYCLE-REDUCED" && inst.K <= 2) {
				cout << "PIEF-CYCLE-REDUCED is not applicable when K <= 2. Cycle model changed to (non)-reduced PIEF-CYCLE. \n";
				sol.modelCycle = "PIEF-CYCLE";
			}


			vertexOrder = findVertexOrder(inst, false);
			vector<vector<vector<int>>> ANew(inst.nR); vector<int> wToTauNew(inst.nR);
			changeOrder(inst, vertexOrder, ANew, wToTauNew);

			// Find a mapping of arcs to weights
			int r, w;
			if (sol.modelCycle == "PIEF-CYCLE-REDUCED") {
				arcToWeight.resize(inst.nR, vector<int>(inst.nR, 0));
				for (int d = 0; d < inst.nR; d++) {
					for (int i = 0; i < ANew[d].size(); i++) {		// loop over all arcs leaving d
						r = ANew[d][i][0]; w = ANew[d][i][1];		// find the recipient and weight
						arcToWeight[d][r] = w;						// save the weight of arc (d,r)
					}
				}
			}

			//// Print new ordering
			//printVector(vertexOrder, "vertexOrder");
			//printA(ANew, wToTauNew, "Anew");

			// Create all subgraphs
			subgraphsPIEF.resize(inst.nR);
			for (int l = 0; l < inst.nR; l++) {					// loop over all subgraphs
				subgraphsPIEF[l] = createSubgraphPIEFCYCLE(inst, ANew, l);

				//// Print the subgraphs
				//cout << "\nSubgraph " << l << ":\n";
				//for (int k = 1; k <= inst.K; k++) {
				//	// printA(subgraphsPIEF[l][k].A, wToTauNew, "A_" + to_string(l) + "[" + to_string(k) + "]");
				//	// print2DVector(subgraphsPIEF[l][k].arc2Idx, "arc2Idx_" + to_string(l) + "[" + to_string(k) + "]", "donor");
				//	print2DVector(subgraphsPIEF[l][k].idx2Arc, "idx2Arc_" + to_string(l) + "[" + to_string(k) + "]", "arc");
				//}
			}

			xSubK.resize(inst.nR, vector<vector<GRBVar>>(inst.K + 1));
			int d, w1, wK;
			xOutSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
			xInSubK.resize(inst.nR, vector<vector<GRBLinExpr>>(inst.nR, vector<GRBLinExpr>(inst.K + 1, 0)));
			// for l in R, a in Al and k in K(a,l): x_lka = 1 iff arc a is used in position k in subgraph l
			for (int l = 0; l < inst.nR; l++) {
				for (int k = 1+1*(sol.modelCycle == "PIEF-CYCLE-REDUCED"); k <= inst.K-1*(sol.modelCycle == "PIEF-CYCLE-REDUCED"); k++) {
					xSubK[l][k].resize(subgraphsPIEF[l][k].idx2Arc.size());
					for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {
						xSubK[l][k][a] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + to_string(a));
						d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2];
						// cout << "l = " << l << ", k = " << k << ", a = " << a << ", d = " << d << ", r = " << r <<  ", w = " << w << "\n";
						totalWeight += w * xSubK[l][k][a];
						numberOfTransplants += (w > 0) * xSubK[l][k][a];
						xOut[vertexOrder[d]] += xSubK[l][k][a]; // note: in terms of original vertex index
						xOutSubK[l][d][k] += xSubK[l][k][a];
						xIn[vertexOrder[r]] += xSubK[l][k][a];  // note: in terms of original vertex index
						xInSubK[l][r][k] += xSubK[l][k][a];

						// In the reduced variant of the model, the arcs on position 1 and K are implicit
						if (sol.modelCycle == "PIEF-CYCLE-REDUCED") {
							// Account for the implicit arcs (l,d) on position 1
							if (k == 2) {
								// add weight of arc (l, d)
								w1 = arcToWeight[l][d];
								totalWeight += w1 * xSubK[l][k][a];
								numberOfTransplants += (w1 > 0) * xSubK[l][k][a];
								xOut[vertexOrder[l]] += xSubK[l][k][a];
								xIn[vertexOrder[d]] += xSubK[l][k][a];
							}
							// Account for the implicit arcs (r,l) on position K
							if (k == inst.K - 1 && r != l) {
								// add weight of arc (r, l)
								wK = arcToWeight[r][l];
								totalWeight += wK * xSubK[l][k][a];
								numberOfTransplants += (wK > 0) * xSubK[l][k][a];
								xOut[vertexOrder[r]] += xSubK[l][k][a];
								xIn[vertexOrder[l]] += xSubK[l][k][a];
							}
						}
					}
				}
			}
		}

		else if (sol.modelCycle == "") {
			// placeholder in case a hybrid model is used
		}

		else {
			cout << "The chosen cycle model (" << sol.modelCycle << ") does not exist. Typo?\n";
			throw invalid_argument("The chosen cycle model (" + sol.modelCycle + ") does not exist. Typo?\n");
		}
	}

	//---------------------------DEFINING THE CHOSEN CHAIN MODEL---------------------------
	if (inst.L > 0) { // only define the chain model when chains are allowed
		if (sol.modelChain == "CF-CHAIN") {
			// Find the set of all chains of length at most L
			chains = findChains(inst);
			// printCyclesOrChains(chains, "Chains", "chain");

			y.resize(chains.size());
			// for c in C: y_c = 1 iff chain c is used
			for (int c = 0; c < chains.size(); c++) {
				y[c] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(c));
				totalWeight += chains[c][1][0] * y[c];
				numberOfTransplants += (chains[c][0].size()-1) * y[c];
				yOut[chains[c][0].front()] += y[c];
				for (int i = 1; i < chains[c][0].size()-1; i++) {		// loop over all recipients in the current chain
					yOut[chains[c][0][i]] += y[c];
					yIn[chains[c][0][i]] += y[c];
				}
				yIn[chains[c][0].back()] += y[c];
			}
		}

		else if (sol.modelChain == "HCF-CHAIN") {

			halfChains = findHalfChains(inst);

			// printCyclesOrChains(halfChains, "Half-chains", "half-chain");

			// Determine the maximum half-chain length for both types
			int limitLenH1 = floor(double(inst.L) / 2) + 1; 
			int limitLenH2 = ceil(double(inst.L) / 2);


			// Find the number of first half-chains ending at each vertex (of maximum length), and the number of second half-chains starting at each vertex
			firstHalfChainsPerEnd.resize(inst.nR);
			longFirstHalfChainsPerEnd.resize(inst.nR);
			secondHalfChainsPerStart.resize(inst.nR);

			int s, e, l;

			// Make a variable for each half-chain
			y.resize(halfChains.size());
			for (int h = 0; h < halfChains.size(); h++) {
				// for h in H: y_h = 1 iff half-chain h is used
				y[h] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(h));
				s = halfChains[h][0].front(); e = halfChains[h][0].back(); l = halfChains[h][0].size();	// find the start, end and size of the half-chain
				totalWeight += halfChains[h][1][0] * y[h];					// add the weight to the objective function
				numberOfTransplants += (halfChains[h][0].size() - 1) * y[h];

				if (l > 1) {													// for all non-auxiliary half-chains
					yOut[s] += y[h];											// the first vertex counts only towards outflow
					for (int i = 1; i < halfChains[h][0].size() - 1; i++) {		// loop over all intermediate recipients in the current half-chain
						yIn[halfChains[h][0][i]] += y[h];						// all intermediate vertices count towards inflow and outflow
						yOut[halfChains[h][0][i]] += y[h];
					}
					yIn[e] += y[h];												// the final vertex counts only towards inflow
				}

				// if the half-chain is a first half-chain
				if (inst.isN[s]) {
					firstHalfChainsPerEnd[e] += y[h];								// add the variable to the first half-chains ending at e
					if (halfChains[h][0].size() == limitLenH1) {					// if the half-chain has maximum size
						longFirstHalfChainsPerEnd[e] += y[h];						// add the variable to the max length first half-chains ending at e
					}
				}
				// if the half-chain is a second half-chain
				else {
					secondHalfChainsPerStart[s] += y[h];							// add the variable to the second half-chains starting at s
				}
			}
		}

		else if (sol.modelChain == "EF-CHAIN-EXP" || sol.modelChain == "EF-CHAIN-EXP-CG" || sol.modelChain == "EF-CHAIN-MTZ" || sol.modelChain == "EF-CHAIN-CUTSET" || sol.modelChain == "EF-CHAIN-CUTSET-CG") {
			// Reduce A, and create mappings from the 2D-arcs to the arc indices, and vice versa
			subgraphEFCHAIN = createSubgraphEFCHAIN(inst);
			// printA(subgraphEFCHAIN.A, inst.wToTau, "Ared");

			// for a in A: y_a = 1 iff arc a is used
			y.resize(subgraphEFCHAIN.idx2Arc.size());
			int d, r, w;
			for (int a = 0; a < subgraphEFCHAIN.idx2Arc.size(); a++) {
				y[a] = model.addVar(0, 1 * (inst.L > 1), 0, GRB_BINARY, "y_" + to_string(a));
				d = subgraphEFCHAIN.idx2Arc[a][0]; r = subgraphEFCHAIN.idx2Arc[a][1]; w = subgraphEFCHAIN.idx2Arc[a][2];
				totalWeight += w * y[a];
				numberOfTransplants += (w>0) * y[a];
				yOut[d] += y[a];
				yIn[r] += y[a];
			}
			if (sol.modelChain == "EF-CHAIN-EXP" || sol.modelChain == "EF-CHAIN-CUTSET") {
				int d, r, a;

				if (inst.L <= inst.nR) { // if L is not infinity, we need to add long chain elimination constraints
					minInfeasibleChains = findMinInfeasibleChains(inst, subgraphEFCHAIN.A, false);
					// print2DVector(minInfeasibleChains, "minInfeasibleChains", "chain");
					arcsUsedOnMinInfeasibleChain.resize(minInfeasibleChains.size());
					for (int c = 0; c < minInfeasibleChains.size(); c++) {
						for (int i = 0; i < minInfeasibleChains[c].size() - 1; i++) {
							d = minInfeasibleChains[c][i]; r = minInfeasibleChains[c][i + 1]; a = subgraphEFCHAIN.arc2Idx[d][r];
							arcsUsedOnMinInfeasibleChain[c] += y[a];
						}
					}
				}

				// Find all cycles of length at most L-1
				shortCycles = findShortCycles(inst, subgraphEFCHAIN.A, inst.L-1);
				// print2DVector(shortCycles, "shortCycles", "cycle");
				if (sol.modelChain == "EF-CHAIN-EXP") {
					arcsUsedOnShortCycles.resize(shortCycles.size());
					for (int c = 0; c < shortCycles.size(); c++) {
						for (int i = 0; i < shortCycles[c].size() - 1; i++) {
							d = shortCycles[c][i]; r = shortCycles[c][i + 1]; a = subgraphEFCHAIN.arc2Idx[d][r];
							arcsUsedOnShortCycles[c] += y[a];
						}
						d = shortCycles[c].back(); r = shortCycles[c].front(); a = subgraphEFCHAIN.arc2Idx[d][r];
						arcsUsedOnShortCycles[c] += y[a];
					}
				}
				else if (sol.modelChain == "EF-CHAIN-CUTSET") {
					vector<int> S, notS;
					arcsIntoCycles.resize(shortCycles.size(), 0);
					// printA(subgraphEFCHAIN.A, inst.wToTau, "A_EFCHAIN");
					// print2DVector(subgraphEFCHAIN.arc2Idx, "A_EFCHAIN", "donor");
					for (int c = 0; c < shortCycles.size(); c++) {
						S = shortCycles[c];
						notS = findComplement(S, inst.nR + inst.nN);
						for (int d : notS) {
							for (int r : S) {
								a = subgraphEFCHAIN.arc2Idx[d][r];
								if (a != -1) { arcsIntoCycles[c] += y[a];}
							}
						}
					}
				}
			}

			else if (sol.modelChain == "EF-CHAIN-MTZ") {
				// for r in R: t_r is the time-stamp of RDP r
				if (inst.L > 1) {
					t.resize(inst.nR);
					for (int r = 0; r < inst.nR; r++) {
						t[r] = model.addVar(1, inst.L - 1, 0, GRB_CONTINUOUS, "t_" + to_string(r));
					}
				}
			}

			// If constraint generation is used, set the required callback attributes
			else if (sol.modelChain == "EF-CHAIN-EXP-CG") {
				cb.setChainCallback(subgraphEFCHAIN, y);
			}
			else if (sol.modelChain == "EF-CHAIN-CUTSET-CG") {
				cb.setChainCallback(subgraphEFCHAIN, y);
			}

		}

		else if (sol.modelChain == "EEF-CHAIN-EXP" || sol.modelChain == "EEF-CHAIN-EXP-CG" || sol.modelChain == "EEF-CHAIN-MTZ" || sol.modelChain == "EEF-CHAIN-CUTSET" || sol.modelChain == "EEF-CHAIN-CUTSET-CG") {

			// Create all subgraphs
			subgraphsEEFCHAIN.resize(inst.nN);
			for (int l = 0; l < inst.nN; l++) {					// loop over all subgraphs
				subgraphsEEFCHAIN[l] = createSubgraphEEFCHAIN(inst, l + inst.nR);
			}

			ySub.resize(inst.nN);
			int d, r, w;
			arcsPerSubgraphChains.resize(inst.nN, 0);
			yOutSub.resize(inst.nN, vector<GRBLinExpr>(inst.nR + inst.nN, 0)); yInSub.resize(inst.nN, vector<GRBLinExpr>(inst.nR + inst.nN, 0));
			// for l in N and a in Al: x_la = 1 iff arc a is used in subgraph l
			for (int l = 0; l < inst.nN; l++) {	// loop over all subgraphs
				ySub[l].resize(subgraphsEEFCHAIN[l].idx2Arc.size());
				for (int a = 0; a < subgraphsEEFCHAIN[l].idx2Arc.size(); a++) {
					ySub[l][a] = model.addVar(0, 1 * (inst.L > 1), 0, GRB_BINARY, "y_" + to_string(a));
					d = subgraphsEEFCHAIN[l].idx2Arc[a][0]; r = subgraphsEEFCHAIN[l].idx2Arc[a][1]; w = subgraphsEEFCHAIN[l].idx2Arc[a][2];
					totalWeight += w * ySub[l][a];
					numberOfTransplants += (w>0) * ySub[l][a];
					yOut[d] += ySub[l][a];
					yOutSub[l][d] += ySub[l][a];
					yIn[r] += ySub[l][a];
					yInSub[l][r] += ySub[l][a];
					arcsPerSubgraphChains[l] += ySub[l][a];
				}
			}

			if (sol.modelChain == "EEF-CHAIN-EXP" || sol.modelChain == "EEF-CHAIN-CUTSET" || sol.modelChain == "EEF-CHAIN-MTZ") {
				subgraphEFCHAIN = createSubgraphEFCHAIN(inst);
			}

			if (sol.modelChain == "EEF-CHAIN-EXP" || sol.modelChain == "EEF-CHAIN-CUTSET") { 
				shortCycles = findShortCycles(inst, subgraphEFCHAIN.A, inst.L - 2);
				int a;
				// print2DVector(shortCycles, "shortCycles", "cycle");
				
				if (sol.modelChain == "EEF-CHAIN-EXP") {
					// Find the arcs used in all small cycles
					arcsUsedOnShortCycles.resize(shortCycles.size());
					for (int c = 0; c < shortCycles.size(); c++) {
						for (int i = 0; i < shortCycles[c].size(); i++) {
							d = shortCycles[c][i]; r = shortCycles[c][(i + 1) % shortCycles[c].size()];
							for (int l = 0; l < inst.nN; l++) {
								a = subgraphsEEFCHAIN[l].arc2Idx[d][r];
								if (a != -1) { arcsUsedOnShortCycles[c] += ySub[l][a]; }
							}
						}
					}
				}

				else if (sol.modelChain == "EEF-CHAIN-CUTSET") {
					vector<int> S, notS;
					arcsIntoCycles.resize(shortCycles.size(), 0);
					for (int c = 0; c < shortCycles.size(); c++) {
						S = shortCycles[c];
						notS = findComplement(S, inst.nR + inst.nN);
						for (int d : notS) {
							for (int r : S) {
								for (int l = 0; l < inst.nN; l++) {
									a = subgraphsEEFCHAIN[l].arc2Idx[d][r];
									if (a != -1) { arcsIntoCycles[c] += ySub[l][a]; }
								}
							}
						}
					}
				}
			}

			else if (sol.modelChain == "EEF-CHAIN-MTZ") {
				// for r in R: t_r is the time-stamp of RDP r
				if (inst.L > 1) {
					t.resize(inst.nR);
					for (int r = 0; r < inst.nR; r++) {
						t[r] = model.addVar(1, inst.L - 1, 0, GRB_CONTINUOUS, "t_" + to_string(r));
					}
				}

				// find sum_{l in N: a in A^l} y_a^l for every a in A
				arcsSummedSubgraphs.resize(subgraphEFCHAIN.idx2Arc.size());
				int al;
				for (int a = 0; a < subgraphEFCHAIN.idx2Arc.size(); a++) {
					d = subgraphEFCHAIN.idx2Arc[a][0]; r = subgraphEFCHAIN.idx2Arc[a][1];
					for (int l = 0; l < inst.nN; l++) {
						al = subgraphsEEFCHAIN[l].arc2Idx[d][r];
						if (al != -1) { arcsSummedSubgraphs[a] += ySub[l][al]; }
					}
				}

			}

			// If constraint generation is used, set the required callback attributes
			else if (sol.modelChain == "EEF-CHAIN-EXP-CG") {
				cb.setChainCallback(subgraphsEEFCHAIN, ySub);
			}
			else if (sol.modelChain == "EEF-CHAIN-CUTSET-CG") {
				cb.setChainCallback(subgraphsEEFCHAIN, ySub);
			}
		}

		else if (sol.modelChain == "PIEF-CHAIN") {

			// Create the position indexed subgraph
			subgraphPIEFCHAIN = createSubgraphPIEFCHAIN(inst);

			yL.resize(inst.L); // seperate set of variables for all positions
			int d, r, w;
			yOutL.resize(inst.nR+inst.nN, vector<GRBLinExpr>(inst.L, 0));
			yInL.resize(inst.nR, vector<GRBLinExpr>(inst.L, 0));
			// for a in A and k in K(a): y_ka = 1 iff arc a is used in position k
			for (int k = 1; k <= inst.L-1; k++) {
				yL[k].resize(subgraphPIEFCHAIN[k].idx2Arc.size());
				for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {
					yL[k][a] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(a));
					d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2];
					// cout << "l = " << l << ", k = " << k << ", a = " << a << ", d = " << d << ", r = " << r <<  ", w = " << w << "\n";
					totalWeight += w * yL[k][a];
					numberOfTransplants += (w>0) * yL[k][a];
					yOut[d] += yL[k][a];
					yOutL[d][k] += yL[k][a];
					yIn[r] += yL[k][a];
					yInL[r][k] += yL[k][a];
				}
			}
		}

		else if (sol.modelChain == "") {
			// placeholder in case a hybrid model is used
		}

		else {
			cout << "The chosen chain model (" << sol.modelChain << ") does not exist. Typo?\n";
			throw invalid_argument("The chosen chain model (" + sol.modelChain + ") does not exist. Typo?\n");
		}

	}

	//---------------------------DEFINING THE CHOSEN HYBRID MODEL---------------------------
	if (sol.modelCycle == "" && sol.modelChain == "" && (inst.K > 0 || inst.L > 0)) {
		if (sol.modelHybrid == "EF-HYBRID" || sol.modelHybrid == "EF-HYBRID-CG" || sol.modelHybrid == "EF-HYBRID-IMPROVED" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") {

			// Throw an error if the improved version of the model is used, but L < K
			if ((sol.modelHybrid == "EF-HYBRID-IMPROVED" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") && inst.L < inst.K) {
				cout << "EF-HYBRID-IMPROVED(-CG) has not been implemented for the case in which L < K.\n";
				throw invalid_argument("EF-HYBRID-IMPROVED(-CG) has not been implemented for the case in which L < K.\n");
			}

			// Reduce A, and create mappings from the 2D-arcs to the arc indices, and vice versa
			subgraphEFHYBRID = createSubgraphEFHYBRID(inst);
			// printA(subgraphEFHYBRID.A, inst.wToTau, "Ared");

			// for a in A: y_a = 1 iff arc a is used
			y.resize(subgraphEFHYBRID.idx2Arc.size());
			int d, r, w;
			for (int a = 0; a < subgraphEFHYBRID.idx2Arc.size(); a++) {
				y[a] = model.addVar(0, 1, 0, GRB_BINARY, "y_" + to_string(a));
				d = subgraphEFHYBRID.idx2Arc[a][0]; r = subgraphEFHYBRID.idx2Arc[a][1]; w = subgraphEFHYBRID.idx2Arc[a][2];
				totalWeight += w * y[a];
				numberOfTransplants += (w>0) * y[a];
				yOut[d] += y[a];
				yIn[r] += y[a];
			}

			if (sol.modelHybrid == "EF-HYBRID") { // if no constraint generation is used
				// Create cycles of size more than K
				longCycles = findLongCycles(inst, subgraphEFHYBRID.A, inst.K);
				// print2DVector(longCycles, "longCycles", "cycle");
				arcsUsedOnLongCycle.resize(longCycles.size());
				int a;
				for (int c = 0; c < longCycles.size(); c++) {
					for (int i = 0; i < longCycles[c].size() - 1; i++) {
						d = longCycles[c][i]; r = longCycles[c][i + 1]; a = subgraphEFHYBRID.arc2Idx[d][r];
						arcsUsedOnLongCycle[c] += y[a];
					}
					d = longCycles[c].back(); r = longCycles[c].front(); a = subgraphEFHYBRID.arc2Idx[d][r];
					arcsUsedOnLongCycle[c] += y[a];
				}

				// Create simple paths consisting of L vertices from R, where the first vertex must be adjacent to at least one NDD
				minInfeasibleChains = findMinInfeasibleChains(inst, subgraphEFHYBRID.A, true);
				// print2DVector(minInfeasibleChains, "minInfeasibleChains", "path");
				arcsUsedOnMinInfeasibleChain.resize(minInfeasibleChains.size());
				for (int c = 0; c < minInfeasibleChains.size(); c++) {
					for (int i = 0; i < minInfeasibleChains[c].size() - 1; i++) {
						d = minInfeasibleChains[c][i]; r = minInfeasibleChains[c][i + 1]; a = subgraphEFHYBRID.arc2Idx[d][r];
						arcsUsedOnMinInfeasibleChain[c] += y[a];
					}
					r = minInfeasibleChains[c].front();
					for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
						a = subgraphEFHYBRID.arc2Idx[n][r];
						if (a != -1) { arcsUsedOnMinInfeasibleChain[c] += y[a]; }
					}
				}
			}

			else if (sol.modelHybrid == "EF-HYBRID-IMPROVED") {
				// In the improved version of EF-HYBRID, we distinguish between three cases

				// Case 1:
				if (inst.L == inst.K) {
					// Find the set of maximum feasible paths: consisting of K vertices from R and none from N
					maxFeasiblePaths = findPaths(inst, subgraphEFHYBRID.A, inst.K - 1);
					// print2DVector(maxFeasiblePaths, "maxFeasiblePaths", "path");
					arcsUsedOnMaxFeasiblePath.resize(maxFeasiblePaths.size(), 0);
					int a;
					for (int p = 0; p < maxFeasiblePaths.size(); p++) {
						for (int i = 0; i < maxFeasiblePaths[p].size() - 1; i++) {
							d = maxFeasiblePaths[p][i]; r = maxFeasiblePaths[p][i + 1]; a = subgraphEFHYBRID.arc2Idx[d][r];
							arcsUsedOnMaxFeasiblePath[p] += y[a];
						}
						a = subgraphEFHYBRID.arc2Idx[maxFeasiblePaths[p].back()][maxFeasiblePaths[p].front()];
						if (a != -1) { arcsUsedOnMaxFeasiblePath[p] -= y[a]; } // Note the minus sign here
					}
				}

				// Case 2:
				else if (inst.L == inst.K + 1) {
					// Create the set of minimal infeasible paths: consisting of K+1 vertices from R and none from N
					minInfeasiblePaths = findPaths(inst, subgraphEFHYBRID.A, inst.K);
					// print2DVector(minInfeasiblePaths, "minInfeasiblePaths", "path");
					arcsUsedOnMinInfeasiblePath.resize(minInfeasiblePaths.size(), 0);
					for (int p = 0; p < minInfeasiblePaths.size(); p++) {
						for (int v = 0; v < inst.K; v++) {
							int d = minInfeasiblePaths[p][v]; int r = minInfeasiblePaths[p][v + 1];
							arcsUsedOnMinInfeasiblePath[p] += y[subgraphEFHYBRID.arc2Idx[d][r]];
						}
					}
				}

				// Case 3:
				else if (inst.L > inst.K + 1) {
					// Create simple paths consisting of L vertices from R (where the first vertex does NOT need to be adjacent to at least one NDD)
					minInfeasibleChains = findMinInfeasibleChains(inst, subgraphEFHYBRID.A, false);
					// print2DVector(minInfeasibleChains, "minInfeasibleChains", "path");
					arcsUsedOnMinInfeasibleChain.resize(minInfeasibleChains.size());
					int a;
					for (int c = 0; c < minInfeasibleChains.size(); c++) {
						for (int i = 0; i < minInfeasibleChains[c].size() - 1; i++) {
							d = minInfeasibleChains[c][i]; r = minInfeasibleChains[c][i + 1]; a = subgraphEFHYBRID.arc2Idx[d][r];
							arcsUsedOnMinInfeasibleChain[c] += y[a];
						}
					}
					// Create cycles of size more than K but less than L
					mediumCycles = findLongCycles(inst, subgraphEFHYBRID.A, inst.K, inst.L);
					// print2DVector(mediumCycles, "mediumCycles", "cycle");
					arcsUsedOnMediumCycle.resize(mediumCycles.size());
					for (int c = 0; c < mediumCycles.size(); c++) {
						for (int i = 0; i < mediumCycles[c].size() - 1; i++) {
							d = mediumCycles[c][i]; r = mediumCycles[c][i + 1]; a = subgraphEFHYBRID.arc2Idx[d][r];
							arcsUsedOnMediumCycle[c] += y[a];
						}
						d = mediumCycles[c].back(); r = mediumCycles[c].front(); a = subgraphEFHYBRID.arc2Idx[d][r];
						arcsUsedOnMediumCycle[c] += y[a];
					}



				}
			}

			else if (sol.modelHybrid == "EF-HYBRID-CG" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") { // if constraint generation is used
				cb.setChainCallback(subgraphEFHYBRID, y); // we put the callback under the chain callbacks
			}
		}

		else {
			cout << "The chosen hybrid model (" << sol.modelHybrid << ") does not exist. Typo?\n";
			throw invalid_argument("The chosen hybrid model (" + sol.modelHybrid + ") does not exist. Typo?\n");
		}
	}

	// add arc weights of arcs to tau
	if (inst.L > 0) {
		for (int d = 0; d < inst.nR; d++) {
			totalWeight += inst.wToTau[d] * (yIn[d] - yOut[d]);
			numberOfTransplants += (inst.wToTau[d] > 0) * (yIn[d] - yOut[d]);
		}
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
			totalWeight += inst.wToTau[d] * (1 - yOut[d]);
			numberOfTransplants += (inst.wToTau[d] > 0) * (1 - yOut[d]);
		}
	}

	model.update();

	//---------------------------GENERAL OBJECTIVE AND CONSTRAINTS---------------------------

	// Set the objective: maximize the sum of cycles weights (= number of transplants in the unweighted case)
	if (not solveHierarchically) { model.setObjective(totalWeight, GRB_MAXIMIZE); }
	// In the hierarchical case: first maximize the number of transplants
	else if (solveHierarchically) { model.setObjective(numberOfTransplants, GRB_MAXIMIZE); }

	//  Constraint for every model 1:  every NDD can donate at most once
	for (int d = inst.nR; d < inst.nR + inst.nN; d++) { // loop over NDDs
		if (yOut[d].size() > 0) {model.addConstr(yOut[d] <= 1);}
	}

	// Constraint for every model 2: every RDP can be donated to at most once
	// - note that these are the only contraints linking the cycle and chain component of the model
	for (int r = 0; r < inst.nR; r++) { // loop over RDPs
		if (xIn[r].size() + yIn[r].size() > 0) {model.addConstr(xIn[r] + yIn[r] <= 1);}
	}

	//---------------------------CYCLE MODEL-SPECIFIC CONSTRAINTS---------------------------
	if (inst.K > 0) {
		// No additional constraints needed for CF-CYCLE

		if (sol.modelCycle == "HCF-CYCLE") {
			// Additional constraints 1 for HCF-CYCLE: matching half-cycles
			for (int s = 0; s < inst.nR - 1; s++) {
				for (int e = s + 1; e < inst.nR; e++) {
					if (halfCyclesPerStartEnd[s][e].size() >= 0) {model.addConstr(halfCyclesPerStartEnd[s][e] == halfCyclesPerStartEnd[e][s]);}
				}
			}
		}

		else if (sol.modelCycle == "EF-CYCLE" || sol.modelCycle == "EF-CYCLE-CG" || sol.modelCycle == "EF-CYCLE-OLD" || sol.modelCycle == "EF-CYCLE-OLD-CG" || sol.modelCycle == "EF-CYCLE-STRONG" || sol.modelCycle == "EF-CYCLE-STRONG-CG") {
			// Additional constraints 1 for EF-CYCLE: flow conservation
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (xIn[r].size() + xOut[r].size() > 0) { model.addConstr(xIn[r] == xOut[r]); }
			}

			if (sol.modelCycle == "EF-CYCLE" || sol.modelCycle == "EF-CYCLE-STRONG") {	// if no constraint generation is used
				// Additional constraints 2 for EF2/3-CYCLE: exclude minimal infeasible paths
				for (int p = 0; p < maxFeasiblePaths.size(); p++) { // loop over maximally cardinality feasible paths
					if (arcsUsedOnMaxFeasiblePath[p].size() > 0) { model.addConstr(arcsUsedOnMaxFeasiblePath[p] <= inst.K - 2); }
				}
			}

			else if (sol.modelCycle == "EF-CYCLE-OLD") {	// if no constraint generation is used and we use the original cardinality constraints
				// Additional constraints 2 for EF-CYCLE-OLD: exclude minimal infeasible paths
				for (int p = 0; p < minInfeasiblePaths.size(); p++) { // loop over minimal infeasible paths
					if (arcsUsedOnMinInfeasiblePath[p].size() > 0) { model.addConstr(arcsUsedOnMinInfeasiblePath[p] <= inst.K - 1); }
				}
			}

		}

		else if (sol.modelCycle == "EEF-CYCLE" || sol.modelCycle == "EEF-CYCLE-OLD" || sol.modelCycle == "EEF-CYCLE-STRONG") {
			// Additional constraints 1 for EEF-CYCLE: flow conservation per subgraph
			for (int l = 0; l < inst.nR; l++) {					// loop over subgraphs
				for (int r = 0; r < inst.nR; r++) {				// loop over RDPs
					if (xInSub[l][r].size() + xOutSub[l][r].size() > 0) { model.addConstr(xInSub[l][r] == xOutSub[l][r]); }
				}
			}

			// Additional constraints 2 for EEF-CYCLE: cycle cardinality
			for (int l = 0; l < inst.nR; l++) { // loop over subgraphs
				if (sol.modelCycle == "EEF-CYCLE") { // improved version of the constraints
					if (arcsPerSubgraph[l].size() > 0) { model.addConstr(arcsPerSubgraph[l] <= inst.K * xOutSub[l][l]); }
				}
				else { // original version of the constraints
					if (arcsPerSubgraph[l].size() > 0) { model.addConstr(arcsPerSubgraph[l] <= inst.K); }
				}
			}

			// Additional constraints 3 for EEF-CYCLE: only use subgraph l if vertex l is used in that subgraph
			if (sol.modelCycle == "EEF-CYCLE-OLD" || sol.modelCycle == "EEF-CYCLE-STRONG") {
				for (int l = 0; l < inst.nR; l++) {					// loop over subgraphs
					for (int d = 0; d < inst.nR; d++) {				// loop over RDPs
						if (xOutSub[l][d].size() > 0) { model.addConstr(xOutSub[l][d] <= xOutSub[l][l]); }
					}
				}
			}
		}

		else if (sol.modelCycle == "PIEF-CYCLE" || sol.modelCycle == "PIEF-CYCLE-REDUCED") {
			// Additional constraints 1 for PIEF-CYCLE: flow conservation per subgraph per index
			for (int l = 0; l < inst.nR; l++) {						// loop over subgraphs
				for (int r = l + 1; r < inst.nR; r++) {				// loop over RDPs
					for (int k = 1 + 1 * (sol.modelCycle == "PIEF-CYCLE-REDUCED"); k <= inst.K-1 - 1 * (sol.modelCycle == "PIEF-CYCLE-REDUCED"); k++) {	// loop over arc indices
						if (xInSubK[l][r][k].size() + xOutSubK[l][r][k+1].size() > 0) { model.addConstr(xInSubK[l][r][k] == xOutSubK[l][r][k + 1]); }
					}
				}
			}
		}
	}

	//---------------------------CHAIN MODEL-SPECIFIC CONSTRAINTS---------------------------
	if (inst.L > 0) {
		// No additional constraints needed for CF-CHAIN

		if (sol.modelChain == "HCF-CHAIN") {
			// Additional constraints 1 for HCF-CHAIN: matching half-chains
			for (int r = 0; r < inst.nR; r++) {
				if (firstHalfChainsPerEnd[r].size() + secondHalfChainsPerStart[r].size() > 0) { model.addConstr(firstHalfChainsPerEnd[r] == secondHalfChainsPerStart[r]); }
			}
		}

		else if (sol.modelChain == "EF-CHAIN-EXP" || sol.modelChain == "EF-CHAIN-EXP-CG" || sol.modelChain == "EF-CHAIN-CUTSET" || sol.modelChain == "EF-CHAIN-CUTSET-CG" || sol.modelChain == "EF-CHAIN-MTZ") {
			// Additional constraints 0 for EF-CHAIN: flow conservation
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (yOut[r].size() + yIn[r].size() > 0) { model.addConstr(yOut[r] <= yIn[r]); }
			}

			if (inst.L > 1 && inst.L < inst.nR) {
				if (sol.modelChain == "EF-CHAIN-EXP" || sol.modelChain == "EF-CHAIN-CUTSET") {
					// Additional constraints 1 for EF-CHAIN-EXP/EF-CHAIN-CUTSET: long chain elimination constraints
					for (int c = 0; c < minInfeasibleChains.size(); c++) {
						if (arcsUsedOnMinInfeasibleChain[c].size() > 0) { model.addConstr(arcsUsedOnMinInfeasibleChain[c] <= inst.L - 2); }
					}
				}
			}

			if (sol.modelChain == "EF-CHAIN-EXP") {
				// Additional constraints 2 for EF-CHAIN-EXP: simple cycle elimination constraints
				for (int c = 0; c < shortCycles.size(); c++) {
					if (arcsUsedOnShortCycles[c].size() > 0) { model.addConstr(arcsUsedOnShortCycles[c] <= shortCycles[c].size() - 1); }
				}
			}
			else if (sol.modelChain == "EF-CHAIN-CUTSET") {
				// Alternative additional constraints 2 for EF-CHAIN-CUTSET: cutset cycle elimination constraints
				for (int c = 0; c < shortCycles.size(); c++) {
					for (int r : shortCycles[c]) {
						if (yIn[r].size() + arcsIntoCycles[c].size() > 0) { model.addConstr(yIn[r] <= arcsIntoCycles[c]); }
					}
				}
			}

			else if (sol.modelChain == "EF-CHAIN-MTZ") {
				// Additional constraints 1 for EF-CHAIN-MTZ: MTZ constraints
				if (inst.L > 1) {
					int d, r;
					for (int a = 0; a < subgraphEFCHAIN.idx2Arc.size(); a++) {
						d = subgraphEFCHAIN.idx2Arc[a][0]; r = subgraphEFCHAIN.idx2Arc[a][1];
						if (!inst.isN[d]) {
							int aBack = subgraphEFCHAIN.arc2Idx[r][d];
							if (aBack != -1) { model.addConstr(t[d] - t[r] + (inst.L - 1) * y[a] + (inst.L-3)*y[aBack] <= inst.L - 2); }
							else { model.addConstr(t[d] - t[r] + (inst.L - 1) * y[a] <= inst.L - 2); }
						}

						// if (!inst.isN[d]) { model.addConstr(t[d] - t[r] + inst.L * y[a] <= inst.L - 1); }    // old
						// if (!inst.isN[d]) { model.addConstr(t[d] - t[r] + (inst.L-1) * y[a] <= inst.L - 2); }
					}
				}
			}
		}

		else if (sol.modelChain == "EEF-CHAIN-EXP" || sol.modelChain == "EEF-CHAIN-EXP-CG" || sol.modelChain == "EEF-CHAIN-CUTSET" || sol.modelChain == "EEF-CHAIN-CUTSET-CG" || sol.modelChain == "EEF-CHAIN-MTZ") {
			// Additional constraints 1 for EEF-CHAIN: flow conservation per subgraph
			for (int l = 0; l < inst.nN; l++) {					// loop over subgraphs
				for (int r = 0; r < inst.nR; r++) {				// loop over RDPs
					if (yInSub[l][r].size() + yOutSub[l][r].size() > 0) { model.addConstr(yInSub[l][r] >= yOutSub[l][r]); }
				}
			}

			// Additional constraints 2 for EEF-CHAIN: chain length at most L and only use subgraph l if vertex l is used in that subgraph
			for (int l = 0; l < inst.nN; l++) {					// loop over subgraphs
				if (arcsPerSubgraphChains[l].size() > 0) { model.addConstr(arcsPerSubgraphChains[l] <= (inst.L-1)*yOutSub[l][inst.nR + l]); }
			}

			// Original 2 seperate sets of constraints
			//// Additional constraints 2 for EEF-CHAIN: only use subgraph l if vertex l is used in that subgraph
			//for (int l = 0; l < inst.nN; l++) {					// loop over subgraphs
			//	for (int d = 0; d < inst.nR; d++) {				// loop over RDPs
			//		if (yOutSub[l][d].size() + yOutSub[l][inst.nR + l].size() > 0) { model.addConstr(yOutSub[l][d] <= yOutSub[l][inst.nR + l]); }
			//	}
			//}

			//// Additional constraints 2.5 for EEF-CHAIN: chain length
			//if (inst.L <= inst.nR) {
			//	for (int l = 0; l < inst.nN; l++) { // loop over subgraphs
			//		if (arcsPerSubgraphChains[l].size() > 0) {model.addConstr(arcsPerSubgraphChains[l] <= inst.L - 1); }
			//	}
			//}


			if (sol.modelChain == "EEF-CHAIN-EXP") { // if no constraint generation is used
				// Additional constraints 3 for EEF-CHAIN-EXP: standard cycle elimination constraints
				for (int c = 0; c < shortCycles.size(); c++) {
					if (arcsUsedOnShortCycles[c].size() > 0) { model.addConstr(arcsUsedOnShortCycles[c] <= shortCycles[c].size() - 1); }
				}
			}

			else if (sol.modelChain == "EEF-CHAIN-CUTSET") { // if no constraint generation is used
				// Additional constraints 3 for EEF-CHAIN-CUTSET: cutset-type cycle elimination constraints
				for (int c = 0; c < shortCycles.size(); c++) {
					for (int r : shortCycles[c]) {
						if (yIn[r].size() + arcsIntoCycles[c].size() > 0) { model.addConstr(yIn[r] <= arcsIntoCycles[c]); }
					}
				}
			}

			else if (sol.modelChain == "EEF-CHAIN-MTZ") {
				// Additional constraints 3 for EEF-CHAIN-MTZ: MTZ constraints
				if (inst.L > 1) {
					int d, r;
					for (int a = 0; a < subgraphEFCHAIN.idx2Arc.size(); a++) {
						d = subgraphEFCHAIN.idx2Arc[a][0]; r = subgraphEFCHAIN.idx2Arc[a][1];
						if (!inst.isN[d]) {
							int aBack = subgraphEFCHAIN.arc2Idx[r][d];
							if (aBack != -1) { model.addConstr(t[d] - t[r] + (inst.L - 1) * arcsSummedSubgraphs[a] + (inst.L - 3) * arcsSummedSubgraphs[aBack] <= inst.L - 2); }
							else { model.addConstr(t[d] - t[r] + (inst.L - 1) * arcsSummedSubgraphs[a] <= inst.L - 2); }
						}
					}
				}
			}
		}
		
		else if (sol.modelChain == "PIEF-CHAIN") {
			// Additional constraints 1 for PIEF-CHAIN: flow conservation per index
			for (int r = 0; r < inst.nR; r++) {						// loop over RDPs
				for (int k = 1; k <= inst.L - 2; k++) {				// loop over arc position indices
					if (yInL[r][k].size() + yOutL[r][k + 1].size() > 0) { model.addConstr(yInL[r][k] >= yOutL[r][k + 1]); }
				}
			}
		}
	}

	//---------------------------HYBRID MODEL-SPECIFIC CONSTRAINTS---------------------------
	if (inst.K > 0 || inst.L > 0) {
		if (sol.modelHybrid == "EF-HYBRID" || sol.modelHybrid == "EF-HYBRID-CG" || sol.modelHybrid == "EF-HYBRID-IMPROVED" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") {
			// Additional constraints 1 for EF-HYBRID: flow conservation
			for (int r = 0; r < inst.nR; r++) { // loop over RDPs
				if (yOut[r].size() + yIn[r].size() > 0) { model.addConstr(yOut[r] <= yIn[r]); }
			}

			if (sol.modelHybrid == "EF-HYBRID") { // if no constraint generation is used
				// Additional constraints 2 for EF-HYBRID: long cycle elimination constraints
				for (int c = 0; c < longCycles.size(); c++) {
					if (arcsUsedOnLongCycle[c].size() > 0) { model.addConstr(arcsUsedOnLongCycle[c] <= longCycles[c].size() - 1); }
				}

				// Additional constraints 3 for EF-HYBRID: long chain elimination constraints
				if (inst.L <= inst.nR) {
					for (int c = 0; c < arcsUsedOnMinInfeasibleChain.size(); c++) {
						if (arcsUsedOnMinInfeasibleChain[c].size() > 0) { model.addConstr(arcsUsedOnMinInfeasibleChain[c] <= inst.L - 1); }
					}
				}
			}

			else if (sol.modelHybrid == "EF-HYBRID-IMPROVED") { // if the improved version of EF-HYBRID is used
				// Alternative constraints for EF-HYBRID-IMPROVED: long cycle and long chain elimination constraints
				if (inst.L == inst.K) {
					for (int p = 0; p < arcsUsedOnMaxFeasiblePath.size(); p++) {
						model.addConstr(arcsUsedOnMaxFeasiblePath[p] <= inst.K - 2);
					}
				}
				else if (inst.L == inst.K + 1) {
					for (int p = 0; p < minInfeasiblePaths.size(); p++) { // loop over minimal infeasible paths
						if (arcsUsedOnMinInfeasiblePath[p].size() > 0) { model.addConstr(arcsUsedOnMinInfeasiblePath[p] <= inst.K - 1); }
					}
				}
				else if (inst.L > inst.K + 1){
					// long chain elimination constraints that also exclude cycles of size more than L
					if (inst.L <= inst.nR) {
						for (int c = 0; c < arcsUsedOnMinInfeasibleChain.size(); c++) {
							if (arcsUsedOnMinInfeasibleChain[c].size() > 0) { model.addConstr(arcsUsedOnMinInfeasibleChain[c] <= inst.L - 2); }
						}
					}
					// cycle elimination constraints for cycles of sizes more than K but less than L
					for (int c = 0; c < mediumCycles.size(); c++) {
						if (arcsUsedOnMediumCycle[c].size() > 0) { model.addConstr(arcsUsedOnMediumCycle[c] <= mediumCycles[c].size() - 1); }
					}
				}
			}
		}
	}

	model.update();

	// Print the model size
	// cout << "Model size: " << model.get(GRB_IntAttr_NumVars) << ", " << model.get(GRB_IntAttr_NumConstrs) << ", " << model.get(GRB_IntAttr_NumNZs);

	//---------------------------RUNNING THE METHOD---------------------------

	// Save the preprocessing time
	sol.timeP = getCPUTime() - start;

	// Change some settings
	model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
	model.getEnv().set(GRB_IntParam_Threads, 1);
	model.getEnv().set(GRB_IntParam_Method, methodLP);
	if (timeLimit - sol.timeP < 0) { sol.timeT = timeLimit; return sol; } // return an empty solution if the time limit has already been exceeded in pre-processing
	model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP);

	// If constraint generation is used, set up a callback 
	if (sol.modelCycle == "EF-CYCLE-CG" || sol.modelCycle == "EF-CYCLE-OLD-CG" || sol.modelCycle == "EF-CYCLE-STRONG-CG" || sol.modelChain == "EF-CHAIN-EXP-CG" || sol.modelChain == "EF-CHAIN-CUTSET-CG" || sol.modelChain == "EEF-CHAIN-EXP-CG" || sol.modelChain == "EEF-CHAIN-CUTSET-CG" || sol.modelHybrid == "EF-HYBRID-CG" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") {
		model.set(GRB_IntParam_LazyConstraints, 1); // indicate that we want to add Lazy Constraints
		model.setCallback(&cb);						// link the callback to the model
	}

	// Find the optimal solution
	if (useRCVF) {					// solve using reduced cost variable fixing
		int nIterRCVF = 0;			// count the number of iterations required by RCVF
		if (printRCVFLog) { cout << "We use RCVF for " << sol.modelFull << "\n"; }
		if (printRCVFLog) { cout << "timeP = " << sol.timeP << ", curTime = " << getCPUTime() - start << "\n"; }

		// Solve the LP relaxation
		GRBModel modelRelaxed = model.relax();
		modelRelaxed.getEnv().set(GRB_IntParam_Crossover, 0); // crossover can be turned off
		if (printRCVFLog) { cout << "Start solving initial LP for RCVF\n"; }
		modelRelaxed.optimize();
		double valLP = modelRelaxed.get(GRB_DoubleAttr_ObjVal);
		sol.LPrel = valLP;
		if (printRCVFLog) { cout << "Finished solving initial LP for RCVF, valLP = " << valLP << "\n"; }

		// If the time limit has already been exceeded after solving the LP relaxation, simply return an empty solution
		if (timeLimit + start - getCPUTime() < 0) { sol.timeT = timeLimit; return sol; }
		if (printRCVFLog) { cout << "curTime after solving relaxed model = " << getCPUTime() - start << "\n"; }
		model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit + start - getCPUTime());
		sol.timeLP = getCPUTime() - start;
		if (printRCVFLog) { cout << "Updated time limit = " << timeLimit + start - getCPUTime() << "\n"; }

		// Find the variable values and their reduced costs
		int numVars = modelRelaxed.get(GRB_IntAttr_NumVars);
		vector<double> solLP(numVars, 0); vector<double> RC(numVars, 0);
		GRBVar* vars = model.getVars();
		GRBVar* varsRelaxed = modelRelaxed.getVars();
		for (int i = 0; i < numVars; i++) {
			solLP[i] = varsRelaxed[i].get(GRB_DoubleAttr_X);
			if (solLP[i] < EPSILON) {
				RC[i] = varsRelaxed[i].get(GRB_DoubleAttr_RC);
			}
		}
		modelRelaxed.reset(1); // clear up the memory
		modelRelaxed.update();

		// Set the initial target
		int T = floor(valLP + EPSILON);
		int nVarRemoved = numVars + 1; int nVarRemovedPrevious;
		int valILP = 0;

		while (true) {

			// Set variables with RC < T - valLP to 0 
			nVarRemovedPrevious = nVarRemoved;
			nVarRemoved = 0;
			vector<bool> isActive(numVars, true);
			for (int i = 0; i < numVars; i++) {
				if (RC[i] <= T - valLP - EPSILON && vars[i].get(GRB_CharAttr_VType) == GRB_BINARY) {
					vars[i].set(GRB_DoubleAttr_UB, 0.0);
					isActive[i] = false;
					nVarRemoved++;
				}
				else if (vars[i].get(GRB_CharAttr_VType) == GRB_BINARY) { // (re-)activate the other variables
					vars[i].set(GRB_DoubleAttr_UB, 1.0);
				}
			}

			// If no variables were reactivated, no need to solve the ILP again
			if (nVarRemoved == nVarRemovedPrevious) {
				if (printRCVFLog) { cout << "\nNo variables were reactivated for T = " << T << ", decrease T to " << T - 1 << "\n"; }
				sol.RCVFNSkips++;	// count the number of skips
				T--;
				if (valILP == T) { // if the best solution reaches the new target, it is optimal
					if (printRCVFLog) { cout << "Target T = " << T << " reached--> solution is optimal.\n"; }
					break;
				}
			}

			// Optimize the model 
			else {
				if (printRCVFLog) { cout << "Start solving ILP model, iter = " << nIterRCVF+1 << "\n"; }
				model.optimize();
				if (printRCVFLog) { cout << "curTime after solving ILP model = " << getCPUTime() - start << "\n"; }

				// Increment the number of iterations by 1
				nIterRCVF++;
				if (printRCVFLog) { cout << "\nIter " << nIterRCVF << ":\n"; }
				if (printRCVFLog) { cout << "LP relaxation value valLP = " << valLP << ", Target value T = " << T << "\n"; }
				if (printRCVFLog) { cout << "Number of variables removed = " << nVarRemoved << ", out of " << model.get(GRB_IntAttr_NumVars) << " originally\n"; }

				// If a feasible solution was found
				if (model.get(GRB_IntAttr_SolCount) >= 1) {
					valILP = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);
					if (printRCVFLog) { cout << "Value reduced ILP = " << valILP << "\n"; }

					// If the target was reached, the solution is optimal
					if (valILP >= T) {
						if (printRCVFLog) { cout << "Target T = " << T << " reached --> solution is optimal.\n"; }
						break;
					}

					// If the target was not reached, but no variables were deactivated, the solution is optimal
					if (nVarRemoved == 0) {
						if (printRCVFLog) { cout << "No variables were deactivated --> solution is optimal.\n"; }
						break;
					}

					// If the target was not reached, but a feasible solution with value T-1 was found, the solution is also optimal
					else if ((model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) && (valILP >= T - 1)) {
						sol.RCVFNSkips++;	// count the number of skips
						if (printRCVFLog) { cout << "Target T = " << "shown to be impossible, but next target: T-1 = " << T - 1 << " was reached --> solution is optimal.\n"; }
						break;
					}

					// If the time limit was reached
					else if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
						// The final T is an UB on the obj. val.
						boundRCVF = T;
						if (printRCVFLog) { cout << "Time limit reached --> UB set to final T = " << T << "\n"; }
						break;
					}

					// Otherwise, decrease T and try again
					else {
						T--;
						if (printRCVFLog) { cout << "Current value of T = " << T+1 << " impossible --> decrease T to " << T << "\n"; }

						// Update the timelimit
						model.getEnv().set(GRB_DoubleParam_TimeLimit, max(timeLimit + start - getCPUTime(), double(0.01)));
						// if (printRCVFLog) { cout << "Updated time limit = " << max(timeLimit + start - getCPUTime(), double(0.01)) << "\n"; }
					}
				}
				// If no feasible solution was found
				else {
					break;
				}
			}
		}
		sol.RCVFNIter = nIterRCVF;
		sol.RCVFNVarDeact = nVarRemoved;
	}
	else {							// solve using standard way
		model.optimize();
	}

	//---------------------------TRANSLATE AND SAVE THE SOLUTION---------------------------

	// Store the results in a Solution object
	if (not solveHierarchically) {
		sol.UB = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);	// the best upper bound
		if (sol.UB < 0) { sol.UB = 1000000; }							// if no upper bound was found yet, set it to 1000000
		if (useRCVF && boundRCVF > -1) { sol.UB = boundRCVF; }			// if RCVF was used and the model ran out of time, return the lowest target

		// If constraint generation is used, save the number of cuts
		if (sol.modelCycle == "EF-CYCLE-CG" || sol.modelCycle == "EF-CYCLE-OLD-CG" || sol.modelCycle == "EF-CYCLE-STRONG-CG" || sol.modelChain == "EF-CHAIN-EXP-CG" || sol.modelChain == "EF-CHAIN-CUTSET-CG" || sol.modelChain == "EEF-CHAIN-EXP-CG" || sol.modelChain == "EEF-CHAIN-CUTSET-CG" || sol.modelHybrid == "EF-HYBRID-CG" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") {
			sol.nCutsCycle += cb.getNCuts(true);
			sol.nCutsChain += cb.getNCuts(false);
		}

		// If the instance is infeasible
		if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
			sol.feas = false;											// indicate that the instance is infeasible
		}

		// If a solution has been found within the time limit
		else if (model.get(GRB_IntAttr_SolCount) >= 1) {

			sol.feas = true;											// indicate that the instance is feasible
			sol.LB = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// the best lower bound

			// If the solution is optimal
			if (sol.LB == sol.UB) {
				sol.opt = true;											// indicate that an optimal solution was found
			}
		}
	}

	// If the hierarchical variant is solved, proceed by maximizing the weight given the maximum number of transplants
	else if (solveHierarchically) {
		sol.timeUnweighted = getCPUTime() - start - sol.timeP;					// save the time spent in the first phase
		sol.UBUnweighted = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);	// the best upper bound
		if (sol.UBUnweighted < 0) { sol.UBUnweighted = 1000000; }				// if no upper bound was found yet, set it to 1000000

		// If a solution has been found within the time limit
		if (model.get(GRB_IntAttr_SolCount) >= 1) {
			sol.feas = true;														// indicate that the instance is feasible
			sol.LBUnweighted = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// the best lower bound
			// If the solution is optimal
			if (sol.LBUnweighted == sol.UBUnweighted) {
				sol.optUnweighted = true;											// indicate that an optimal solution was found
			}
		}

		// Only continue if the first stage problem has been solved to optimality
		if (sol.optUnweighted) {

			// Print the first stage solution
			// cout << "The first stage problem has been solved to optimality.\n- Optimal number of transplants = " << sol.LBUnweighted << "\n- Time used = " << sol.timeP + sol.timeUnweighted << "s\n";
			// cout << "Start maximizing the weight given the maximum number of transplants...\n";

			// change the objective to maximizing the weight
			model.setObjective(totalWeight, GRB_MAXIMIZE);

			// add a constraint to enforce the maximum number of transplants
			model.addConstr(numberOfTransplants == sol.LBUnweighted);

			// update the time limit
			model.getEnv().set(GRB_DoubleParam_TimeLimit, timeLimit - sol.timeP - sol.timeUnweighted);

			// re-optimize the model
			model.optimize();

			// return the solution
			sol.UB = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
			if (sol.UB < 0) { sol.UB = 1000000; }				// if no upper bound was found yet, set it to 1000000

			// ff a solution has been found within the time limit
			if (model.get(GRB_IntAttr_SolCount) >= 1) {
				sol.LB = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	// the best lower bound

				// If the solution is optimal
				if (sol.LB == sol.UB) {
					sol.opt = true;						// indicate that an optimal solution was found
				}

			}

			// cout << "\n-------------------------\nSolution second stage: status = " << sol.opt << ", bounds = [" << sol.LB << ", " << sol.UB << "]\n-------------------------\n";
		}

		else {
			// cout << "The first problem has not been solved to optimality within the time limit.\n";
		}

		sol.timeWeighted = getCPUTime() - start - sol.timeP - sol.timeUnweighted;	// save time spent on second stage problem
	}

	// Decompose the solution into the selected cycles and chains
	if (model.get(GRB_IntAttr_SolCount) >= 1) {
		int xval, yval;
		sol.cyclesAndChains.resize(2);


		//---------------------------DECOMPOSING THE CYCLES---------------------------
		if (inst.K > 0) {
			if (sol.modelCycle == "CF-CYCLE") {
				for (int c = 0; c < cycles.size(); c++) {				// loop over all cycles
					xval = ceil(x[c].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
					if (xval == 1) {									// if the cycle is selected
						sol.cyclesAndChains[0].push_back(cycles[c]);	// add it
					}
				}
			}

			else if (sol.modelCycle == "HCF-CYCLE") {
				vector<vector<vector<vector<int>>>> halfCyclesUsedPerStartEnd(inst.nR, vector<vector<vector<int>>>(inst.nR));
				vector<vector<int>> h, h1, h2, c;
				int s, e;
				for (int i = 0; i < halfCycles.size(); i++) {						// loop over all half-cycles
					xval = ceil(x[i].get(GRB_DoubleAttr_X) - EPSILON);				// find variable value
					if (xval == 1) {												// if the half-cycle is selected
						h = halfCycles[i]; s = h[0].front(); e = h[0].back();		// find the half-cycle, it's start and end
						halfCyclesUsedPerStartEnd[s][e] = h;						// save the half-cycle as a half-cycle starting at s and ending at e
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

			else if (sol.modelCycle == "EF-CYCLE" || sol.modelCycle == "EF-CYCLE-CG" || sol.modelCycle == "EF-CYCLE-OLD" || sol.modelCycle == "EF-CYCLE-OLD-CG" || sol.modelCycle == "EF-CYCLE-STRONG" || sol.modelCycle == "EF-CYCLE-STRONG-CG" || sol.modelCycle == "EEF-CYCLE" || sol.modelCycle == "EEF-CYCLE-OLD" || sol.modelCycle == "EEF-CYCLE-STRONG" || sol.modelCycle == "PIEF-CYCLE" || sol.modelCycle == "PIEF-CYCLE-REDUCED") {
				vector<vector<int>> donationPerDonor(inst.nR);			// array containing [r, w_dr] for each donor d that donates
				int d, r, w;
				if (sol.modelCycle == "EEF-CYCLE" || sol.modelCycle == "EEF-CYCLE-OLD" || sol.modelCycle == "EEF-CYCLE-STRONG") {	// for EEF-CYCLE we need to account for the subgraphs and we need to reset the vertex order
					for (int l = 0; l < inst.nR; l++) {							// loop over all subgraphs
						for (int a = 0; a < subgraphsEEF[l].idx2Arc.size(); a++) {	// loop over all arcs
							xval = ceil(xSub[l][a].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
							if (xval == 1) {									// if the arc is selected
								d = vertexOrder[subgraphsEEF[l].idx2Arc[a][0]]; r = vertexOrder[subgraphsEEF[l].idx2Arc[a][1]]; w = subgraphsEEF[l].idx2Arc[a][2]; // decompose the arc
								donationPerDonor[d] = { r, w };					// add the arc to the list of donations
							}
						}
					}
				}
				else if (sol.modelCycle == "PIEF-CYCLE" || sol.modelCycle == "PIEF-CYCLE-REDUCED") {		// for PIEF-CYCLE we need to account for the subgraphs and position indices indices and we need to reset the vertex order
					for (int l = 0; l < inst.nR; l++) {											// loop over all subgraphs
						for (int k = 1+1*(sol.modelCycle == "PIEF-CYCLE-REDUCED"); k <= inst.K-1*(sol.modelCycle== "PIEF-CYCLE-REDUCED"); k++) {	// loop over all position indices
							for (int a = 0; a < subgraphsPIEF[l][k].idx2Arc.size(); a++) {		// loop over all arcs
								xval = ceil(xSubK[l][k][a].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
								if (xval == 1) {												// if the arc is selected
									d = subgraphsPIEF[l][k].idx2Arc[a][0]; r = subgraphsPIEF[l][k].idx2Arc[a][1]; w = subgraphsPIEF[l][k].idx2Arc[a][2]; // decompose the arc
									donationPerDonor[vertexOrder[d]] = { vertexOrder[r], w };	// add the arc to the list of donations

									if (sol.modelCycle == "PIEF-CYCLE-REDUCED") {
										// Account for the implicit arcs (l,d) on position 1
										if (k == 2) {donationPerDonor[vertexOrder[l]] = { vertexOrder[d],arcToWeight[l][d] };	}
										// Account for the implicit arcs (r,l) on position K
										if (k == inst.K-1 && r != l) { donationPerDonor[vertexOrder[r]] = { vertexOrder[l],arcToWeight[r][l]}; }
									}
								}
							}
						}
					}
				}

				else {														// for all variations of EF there are no subgraphs
					for (int a = 0; a < subgraphEF.idx2Arc.size(); a++) {	// loop over all arcs
						xval = ceil(x[a].get(GRB_DoubleAttr_X) - EPSILON);	// find variable value
						if (xval == 1) {									// if the arc is selected
							d = subgraphEF.idx2Arc[a][0]; r = subgraphEF.idx2Arc[a][1]; w = subgraphEF.idx2Arc[a][2]; // decompose the arc
							donationPerDonor[d] = { r, w };					// add the arc to the list of donations
						}
					}
				}
				// print2DVector(donationPerDonor, "donationPerDonor", "donor");
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
				// print2DVector(donationPerDonor, "donationPerDonor", "donor");
			}
		}

		//---------------------------DECOMPOSING THE CHAINS---------------------------
		if (inst.L > 0) {
			if (sol.modelChain == "CF-CHAIN") {
				for (int c = 0; c < chains.size(); c++) {
					yval = ceil(y[c].get(GRB_DoubleAttr_X) - EPSILON); // find variable value
					if (yval == 1) {
						chains[c][1][0] += inst.wToTau[chains[c][0].back()];
						sol.cyclesAndChains[1].push_back(chains[c]);
					}
				}
				// Also add trivial n-tau chains for unused NDDs n
				for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
					if (ceil(yOut[d].getValue() - EPSILON) == 0) {
						sol.cyclesAndChains[1].push_back({ {d}, {inst.wToTau[d]} });
					}
				}
			}

			else if (sol.modelChain == "HCF-CHAIN") {
				vector<vector<vector<int>>> firstHalfChainUsedPerEnd(inst.nR);
				vector<vector<vector<int>>> secondHalfChainUsedPerStart(inst.nR);
				vector<vector<int>> h, h1, h2, c;
				int s, e;
				for (int i = 0; i < halfChains.size(); i++) {						// loop over all half-chains
					yval = ceil(y[i].get(GRB_DoubleAttr_X) - EPSILON);				// find variable value
					if (yval == 1) {												// if the half-chain is selected
						h = halfChains[i]; s = h[0].front(); e = h[0].back();		// find the half-chain, it's start and end
						if (inst.isN[s]) { firstHalfChainUsedPerEnd[e] = h; }		// if it's a first half-chain, save under first half-chains ending at e
						else { secondHalfChainUsedPerStart[s] = h; }				// if it's a second half-chain, save under second half-chains starting at s
					}
				}
				//print3DVector(firstHalfChainUsedPerEnd, "firstHalfChainUsedPerEnd", "end");
				//print3DVector(secondHalfChainUsedPerStart, "secondHalfChainUsedPerStart", "start");

				for (int r = 0; r < inst.nR; r++) {									// loop over all half-way points	
					if (firstHalfChainUsedPerEnd[r].size() > 0) {					// if there is a first half-chain ending at e
						h1 = firstHalfChainUsedPerEnd[r];							// find the half-chain
						c.clear(); c.resize(2);										// initialize a chain that will be the combination of both half-chains
						for (int d : h1[0]) { c[0].push_back(d); }					// add the vertices of the first half-chain
						c[1] = { h1[1][0] };										// add the weight of the first half-chain
						if (secondHalfChainUsedPerStart[r].size() > 0) {			// if there is a matching second half-chain
							h2 = secondHalfChainUsedPerStart[r];					// find the matching second half-chain
							h2[0].erase(h2[0].begin());								// remove the double vertex
							for (int d : h2[0]) { c[0].push_back(d); }				// add the vertices of the second half-chain
							c[1][0] += h2[1][0];									// add the weight of the second half-chain
						}
						c[1][0] += inst.wToTau[c[0].back()];						// add the weight of the final vertex to tau
						sol.cyclesAndChains[1].push_back(c);						// add the completed chain
					}
				}

				// Also add trivial n-tau chains for unused NDDs n
				for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
					if (ceil(yOut[d].getValue() - EPSILON) == 0) {
						sol.cyclesAndChains[1].push_back({ {d}, {inst.wToTau[d]} });
					}
				}

			}

			else if (sol.modelChain == "EF-CHAIN-EXP" || sol.modelChain == "EF-CHAIN-EXP-CG" || sol.modelChain == "EF-CHAIN-CUTSET" || sol.modelChain == "EF-CHAIN-CUTSET-CG" || sol.modelChain == "EF-CHAIN-MTZ" || sol.modelChain == "EEF-CHAIN-EXP" || sol.modelChain == "EEF-CHAIN-EXP-CG" || sol.modelChain == "EEF-CHAIN-CUTSET" || sol.modelChain == "EEF-CHAIN-CUTSET-CG" || sol.modelChain == "EEF-CHAIN-MTZ" || sol.modelChain == "PIEF-CHAIN") {
				vector<vector<int>> donationPerDonor(inst.nR + inst.nN);		// array containing [r, w_dr] for each donor d that donates
				int d, r, w;

				// First find the donation per donor
				if (sol.modelChain == "EF-CHAIN-EXP" || sol.modelChain == "EF-CHAIN-EXP-CG" || sol.modelChain == "EF-CHAIN-CUTSET" || sol.modelChain == "EF-CHAIN-CUTSET-CG" || sol.modelChain == "EF-CHAIN-MTZ") {
					for (int a = 0; a < subgraphEFCHAIN.idx2Arc.size(); a++) {	// loop over all arcs
						yval = ceil(y[a].get(GRB_DoubleAttr_X) - EPSILON);		// find variable value
						if (yval == 1) {										// if the arc is selected
							d = subgraphEFCHAIN.idx2Arc[a][0]; r = subgraphEFCHAIN.idx2Arc[a][1]; w = subgraphEFCHAIN.idx2Arc[a][2]; // decompose the arc
							donationPerDonor[d] = { r, w };						// add the arc to the list of donations
						}
					}
				}

				else if (sol.modelChain == "EEF-CHAIN-EXP" || sol.modelChain == "EEF-CHAIN-EXP-CG" || sol.modelChain == "EEF-CHAIN-CUTSET" || sol.modelChain == "EEF-CHAIN-CUTSET-CG" || sol.modelChain == "EEF-CHAIN-MTZ") {
					for (int l = 0; l < inst.nN; l++) {
						for (int a = 0; a < subgraphsEEFCHAIN[l].idx2Arc.size(); a++) {	// loop over all arcs
							yval = ceil(ySub[l][a].get(GRB_DoubleAttr_X) - EPSILON);		// find variable value
							if (yval == 1) {										// if the arc is selected
								d = subgraphsEEFCHAIN[l].idx2Arc[a][0]; r = subgraphsEEFCHAIN[l].idx2Arc[a][1]; w = subgraphsEEFCHAIN[l].idx2Arc[a][2]; // decompose the arc
								donationPerDonor[d] = { r, w };						// add the arc to the list of donations
							}
						}
					}
				}

				else if (sol.modelChain == "PIEF-CHAIN") {
					for (int k = 1; k <= inst.L - 1; k++) {								// loop over all arc positions
						for (int a = 0; a < subgraphPIEFCHAIN[k].idx2Arc.size(); a++) {	// loop over all arcs
							yval = ceil(yL[k][a].get(GRB_DoubleAttr_X) - EPSILON);		// find variable value
							if (yval == 1) {										// if the arc is selected
								d = subgraphPIEFCHAIN[k].idx2Arc[a][0]; r = subgraphPIEFCHAIN[k].idx2Arc[a][1]; w = subgraphPIEFCHAIN[k].idx2Arc[a][2]; // decompose the arc
								donationPerDonor[d] = { r, w };						// add the arc to the list of donations
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
				// print2DVector(donationPerDonor, "donationPerDonor", "donor");

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
				// print2DVector(donationPerDonor, "donationPerDonor", "donor");


			}
		}

		//---------------------------DECOMPOSING THE CYCLES AND CHAINS---------------------------
		if (inst.K > 0 || inst.L > 0) {
			if (sol.modelHybrid == "EF-HYBRID" || sol.modelHybrid == "EF-HYBRID-CG" || sol.modelHybrid == "EF-HYBRID-IMPROVED" || sol.modelHybrid == "EF-HYBRID-IMPROVED-CG") {
				// Decompose the solution
				vector<vector<int>> donationPerDonor(inst.nR + inst.nN);						// array containing the recipient r for each donor d that donates
				int yval, d, r, w;
				for (int a = 0; a < subgraphEFHYBRID.idx2Arc.size(); a++) {						// loop over all arcs
					yval = ceil(y[a].get(GRB_DoubleAttr_X) - EPSILON);							// find variable value
					if (yval == 1) {															// if the arc is selected
						d = subgraphEFHYBRID.idx2Arc[a][0]; r = subgraphEFHYBRID.idx2Arc[a][1];	w = subgraphEFHYBRID.idx2Arc[a][2]; // decompose the arc
						donationPerDonor[d] = { r, w };											// add the arc to the list of donations
					}
				}
				// Also find the donations to tau
				if (inst.L > 0) {
					for (int d = 0; d < inst.nR; d++) {
						if (ceil(yIn[d].getValue() - EPSILON) - ceil(yOut[d].getValue() - EPSILON) == 1) {
							donationPerDonor[d] = { -1, inst.wToTau[d] };
						}
					}
					for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
						if (ceil(yOut[d].getValue() - EPSILON) == 0) {
							donationPerDonor[d] = { -1, inst.wToTau[d] };
						}
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
				// Find the cycles
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
		}

		// If the problem was solved hierarchically, but the first problem was only solved to suboptimality, find the obj. val. of the suboptimal solution
		if (solveHierarchically && not sol.optUnweighted) {
			for (int c = 0; c < sol.cyclesAndChains[0].size(); c++) { // loop over all cycles
				sol.LB += sol.cyclesAndChains[0][c][1][0];
			}
			for (int c = 0; c < sol.cyclesAndChains[1].size(); c++) { // loop over all chains
				sol.LB += sol.cyclesAndChains[1][c][1][0];
			}
		}
	}

	// Save the model size and total time
	sol.nVar = model.get(GRB_IntAttr_NumVars);						// number of variables
	sol.nConstr = model.get(GRB_IntAttr_NumConstrs);				// number of constraints
	sol.nCoeff = model.get(GRB_IntAttr_NumNZs);						// number of non-zero coefficients
	sol.timeT = getCPUTime() - start;
	sol.timeW = model.get(GRB_DoubleAttr_Work);

	// Optionally, seperately re-solve the LP relaxation
	if (solveLP && not solveHierarchically) {
		if (printGurobiLog) { cout << "\n----------------------------------\nSeperately resolve the LP relaxation:\n"; }
		double startLP = getCPUTime();
		GRBModel modelRelaxed = model.relax();
		model.reset(1);
		modelRelaxed.getEnv().set(GRB_IntParam_Crossover, 0); // crossover can be turned off
		modelRelaxed.optimize();
		if (modelRelaxed.get(GRB_IntAttr_Status) == GRB_OPTIMAL) { sol.LPrel = modelRelaxed.get(GRB_DoubleAttr_ObjVal); }
		sol.timeLP = getCPUTime() - startLP;
	}

	//---------------------------RETURNING THE SOLUTION---------------------------

	// Return Solution object
	return sol;
}

//---------------------------FUNCTIONS RELATING TO THE INSTANCE STRUCT---------------------------

Instance readInstance(string path, string filename) {
	// This function reads an instance from a .JSON file and saves it as an Instance struct

	// initialize Instance object
	Instance inst;
	inst.name = filename;

	// open the file
	ifstream file(path + "/" + filename);
	using json = nlohmann::json;
	json data = json::parse(file);
	vector<vector<vector<int>>> A; // version of inst.A before changing the donor indices
	for (const auto& [key, donor] : data["data"].items()) {

		// Have to do RDPs first
		if (donor.contains("altruistic") && donor["altruistic"] == true) {
			continue;
		}
		inst.V.push_back(stoi(key));
		A.push_back({});
		inst.isN.push_back(false);
		inst.nR++;
		for (const auto& match : donor["matches"]) {
			auto recipient = match["recipient"].get<int>();
			auto weight = match["score"].get<int>();
			A.back().push_back({ recipient, weight });
		}
	}
	for (const auto& [key, donor] : data["data"].items()) {
		// Have to do NDDs second
		if (!(donor.contains("altruistic") && donor["altruistic"] == true)) {
			continue;
		}
		inst.V.push_back(stoi(key));
		A.push_back({});
		inst.isN.push_back(true);
		inst.nN++;
		for (const auto& match : donor["matches"]) {
			auto recipient = match["recipient"].get<int>();
			auto weight = match["score"].get<int>();
			A.back().push_back({ recipient, weight });
		}
	}

	// change the donor indices to {0, 1, ..., nR + nN - 1}
	vector<int> idxMap(*max_element(inst.V.begin(), inst.V.end()) + 1, -1); // idxMap(d) gives the new index of the donor that originally had index d
	for (int d = 0; d < inst.V.size(); d++) {
		idxMap[inst.V[d]] = d;
	}
	for (int d = 0; d < A.size(); d++) {
		inst.A.push_back({});
		for (int r = 0; r < A[d].size(); r++) {
			inst.A[d].push_back({ idxMap[A[d][r][0]],A[d][r][1] });
		}
	}

	inst.wToTau.resize(inst.V.size(), 0);

	// Count the number of arcs
	for (int d = 0; d < inst.nR + inst.nN; d++) {       // loop over all donors
		for (int i = 0; i < inst.A[d].size(); i++) {    // loop over all recipients compatible with the donor
			inst.nA++;
			if (!inst.isN[d]) { inst.nAR++; }
		}
	}

	// Find the shortest path distance between every pair of vertices
	// inst.floyd = floydsAlgorithm(inst, inst.A);
	return inst;
}

void Instance::print(bool full) {
	// This function prints information about the Instance struct
	cout << "\n------------------------------------------------------\n";
	cout << "Instance information (\"" << name << "\"):\n";
	cout << "- maximum cycle length: K = " << K << "\n";
	cout << "- maximum chain length: L = " << L << "\n";
	cout << "- number of RDPs: nR = " << nR << '\n';
	cout << "- number of NDDs: nN = " << nN << '\n';
	cout << "- number of arcs: nA = " << nA << '\n';
	if (full) {
		printVector(isN, "- isN");
		printA(A, wToTau, "- A");
	}
	cout << "------------------------------------------------------\n";
}

void Instance::nonWeightedVariant(bool useWeightsToTau) {
	// This function sets all weights to 1
	for (int v = 0; v < V.size(); v++) {
		for (int a = 0; a < A[v].size(); a++) {
			A[v][a][1] = 1;
		}
		if (useWeightsToTau) { wToTau[v] = 1; }
	}
}

void Instance::generateRandomWToTau() {
	// This function randomly generates weights from each vertex to tau
	
	// Set the random seed for reproducibility
	srand(100);

	// Find the minimum and maximum weight in the instance
	int wMin = numeric_limits<int>::max(); int wMax = numeric_limits<int>::min();
	for (int v = 0; v < V.size(); v++) {
		for (int a = 0; a < A[v].size(); a++) {
			if (A[v][a][1] < wMin) {
				wMin = A[v][a][1];
			}
			else if (A[v][a][1] > wMax) {
				wMax = A[v][a][1];
			}
		}
	}

	// Set the weights
	for (int v = 0; v < V.size(); v++) {
		wToTau[v] = rand() % (wMax - wMin + 1) + wMin;
	}
}

//---------------------------FUNCTIONS RELATING TO THE SOLUTION STRUCT---------------------------
void Solution::print(int detail) {
	// Prints a Solution struct in a nice format

	// if detail is set to -1 or -2, the main metrics are printed in one line
	if (detail == -1 || detail == -2) {
		string model = modelFull + ":";
		int originalSize = model.size();
		for (int i = 0; i < 53 - originalSize; i++) { model += " "; }
		cout << " " << model;
		if (not solvedHierarchically) {
			// for option -2, the LP relaxation value is not printed
			if (detail == -2) {
				if (opt) { cout << LB; }
				else { cout << "[" << LB << ", " << UB << "]"; }
			}
			// for option -1, the LP relaxation value is printed
			else if (detail == -1) {
				if (opt) { cout << LB << " {" << LPrel << "}"; }
				else { cout << "[" << LB << ", " << UB << " {" << LPrel << "}]"; }
			}
		}
		if (solvedHierarchically) {
			if (opt) {cout << "#Don = " << LBUnweighted << ", W = " << LB; }
			else if (optUnweighted) {cout << "#Don = " << LBUnweighted << ", W = [" << LB << ", " << UB << "]";}
			else { cout << "#Don = [" << LBUnweighted << ", " << UBUnweighted << "], W = [" << LB << ", " << UB << "]"; }
		}
		cout << " (" << timeT << "s)\n";
	}

	// otherwise, the full solution info is printed
	else {
		// print parameters and model
		cout << "\n------------------------------------------------------\n";
		cout << "Model:\n";
		if (modelHybrid == "") {
			cout << "- model used for cycles: " << modelCycle << "\n";
			cout << "- model used for chains: " << modelChain << "\n";
		}
		else {
			cout << "- model used for both cycles and chains: " << modelHybrid << "\n";
		}

		// print solution status
		if (not solvedHierarchically) {
			cout << "Solution status:\n";
			if (opt) {
				cout << "- optimal solution\n";
				cout << "- obj.value: " << LB << "\n";
				cout << "- LP rel.: " << LPrel << "\n";
			}
			else if (!feas) {
				cout << "- no feasible solution found\n";
			}
			else {
				cout << "- suboptimal solution\n";
				cout << "- LB: " << LB << "\n";
				cout << "- UB: " << UB << "\n";
				cout << "- LP rel.: " << LPrel << "\n";
			}
		}
		else {
			cout << "Solution status:\n";
			if (opt) {
				cout << "- optimal solution\n";
				cout << "- obj.value: " << LB << "\n";
				cout << "- number of transplants: " << LBUnweighted << "\n";
			}
			else if (optUnweighted) {
				cout << "- first stage problem solved to optimality, but suboptimal solution for second stage problem\n";
				cout << "- LB: " << LB << "\n";
				cout << "- UB: " << UB << "\n";
				cout << "- number of transplants: " << LBUnweighted << "\n";
			}
			else if (!feas) {
				cout << "- no feasible solution found\n";
			}
			else {
				cout << "- first stage problem not solved to optimality\n";
				cout << "- LB transplants: " << LBUnweighted << "\n";
				cout << "- UB transplants: " << UBUnweighted << "\n";
				cout << "- LB weight: " << LB << "\n";
				cout << "- UB weight: " << UB << "\n";
			}
		}

		// print running time
		cout << "Running time:\n";
		cout << "- total: " << timeT << "s\n";
		cout << "- preprocessing: " << timeP << "s\n";
		if (solvedHierarchically) { cout << "- first stage: " << timeUnweighted << "s\n"; }
		if (solvedHierarchically) { cout << "- second stage: " << timeWeighted << "s\n"; }
		cout << "- time LP rel.: " << timeLP << "s\n";

		// print model size
		cout << "Model size:\n";
		cout << "- number of variables: " << nVar << "\n";
		cout << "- number of constraints: " << nConstr << "\n";
		if (modelCycle == "EF-CYCLE-CG" || modelCycle == "EF-CYCLE-OLD-CG" || modelCycle == "EF-CYCLE-STRONG-CG" || modelChain == "EF-CHAIN-EXP-CG" || modelChain == "EF-CHAIN-CUTSET-CG" || modelChain == "EEF-CHAIN-EXP-CG" || modelChain == "EEF-CHAIN-CUTSET-CG" || modelHybrid == "EF-HYBRID-CG" || modelHybrid == "EF-HYBRID-IMPROVED-CG") {
			cout << "- number of cuts: " << nCutsCycle + nCutsChain;
			if (nCutsCycle + nCutsChain > 0) {
				if (((modelCycle == "EF-CYCLE-CG" || modelCycle == "EF-CYCLE-OLD-CG" || modelCycle == "EF-CYCLE-STRONG-CG") && (modelChain == "EF-CHAIN-EXP-CG" || modelChain == "EF-CHAIN-CUTSET-CG" || modelChain == "EEF-CHAIN-EXP-CG" || modelChain == "EEF-CHAIN-CUTSET-CG")) || (modelHybrid == "EF-HYBRID-CG" || modelHybrid == "EF-HYBRID-IMPROVED-CG")) {
					cout << " (" << nCutsCycle << ", " << nCutsChain << ")";
				}
			}
			cout << "\n";
		}
		cout << "- number of non-zero coefficients: " << nCoeff << "\n";

		// print information regarding RCVF
		if (RCVFUsed) {
			cout << "RCVF:\n";
			cout << "- number of iterations: " << RCVFNIter << " (iterations skipped: " << RCVFNSkips << ")\n";
			cout << "- number of deactivated variables: " << RCVFNVarDeact << "\n";
			cout << "- number of remaining variables: " << nVar-RCVFNVarDeact << "\n";
		}

		// print feasible solution if it was found and you want to print it
		if (detail == 1 && feas == 1) {
			if (opt) cout << "Optimal solution:\n";
			else cout << "Feasible solution:\n";
			printCyclesOrChains(cyclesAndChains[0], "- Cycles", "cycle");
			printCyclesOrChains(cyclesAndChains[1], "- Chains", "chain");
		}
		cout << "------------------------------------------------------\n";
	}
}

bool Solution::check(const Instance& inst) {
	// This function checks for a given solution if:
	// - The maximum cycle and chain lengths are not exceeded
	// - Every donor is involved in at most one exchange
	// - The objective value is correct (we only partly check this: we assume that the value of each individual cycle and chain is correct)

	bool correct = true;
	if (not feas) {			// skip the check if the model is infeasible
		return correct;
	}
	vector<int> cycle, chain;
	int d, obj = 0;
	string message;
	if (modelHybrid == "") {message = "ERRORS in " + modelCycle + " + " + modelChain + ":\n";}
	else { message = "ERRORS in " + modelHybrid + ":\n"; }
	vector<int> isUsed(inst.nR + inst.nN, 0);
	for (int c = 0; c < cyclesAndChains[0].size(); c++) { // loop over all cycles
		cycle = cyclesAndChains[0][c][0];
		if (cycle.size() > inst.K) { message += " - Cycle " + to_string(c) + " is longer than K.\n"; correct = false; }
		for (int i = 0; i < cycle.size(); i++) {
			d = cycle[i];
			isUsed[d]++;
			if (isUsed[d] > 1) { message += " - Donor " + to_string(d) + " is involved in more than one exchange.\n"; correct = false; }
		}
		obj += cyclesAndChains[0][c][1][0];
	}
	for (int c = 0; c < cyclesAndChains[1].size(); c++) { // loop over all chains
		chain = cyclesAndChains[1][c][0];
		if (chain.size() > inst.L) { message += " - Chain " + to_string(c) + " is longer than L.\n"; correct = false; }
		for (int i = 0; i < chain.size(); i++) {
			d = chain[i];
			isUsed[d]++;
			if (isUsed[d] > 1) { message += " - Donor " + to_string(d) + " is involved in more than one exchange.\n"; correct = false; }
		}
		obj += cyclesAndChains[1][c][1][0];
	}
	if (obj != LB) { message += " - The objective value is incorrect: obj = " + to_string(obj) + " != " + to_string(LB) + " = sol.LB.\n"; correct = false; }

	if (not correct) { cout << message; }

	return correct;
}

//---------------------------FUNCTIONS RELATING TO CALLBACKS---------------------------
Callback::Callback(const Instance& inst, string modelCycle, string modelChain, string modelHybrid) {
	// Initialize the Callback object
	this->inst = inst;					// the instance
	this->modelCycle = modelCycle;		// the name of the cycle model
	this->modelChain = modelChain;		// the name of the chain model
	this->modelHybrid = modelHybrid;	// the name of the chain model
	this->inst = inst;					// the instance
	this->nCutsCycle = 0;				// the number of cuts due to the cycle model
	this->nCutsChain = 0;				// the number of cuts due to the chain model
}

void Callback::setCycleCallback(const Subgraph& subgraphCycle, const vector<GRBVar>& x) {
	// Sets the attributes of the Callback object needed to be able to handle the lazy constraints for the cycle model
	this->subgraphCycle = subgraphCycle;	// the reduced set of arcs
	this->x = x;							// the variables that the callback acts on
}

void Callback::setChainCallback(const Subgraph& subgraphChain, const vector<GRBVar>& y) {
	// Sets the attributes of the Callback object needed to be able to handle the lazy constraints for the EF-CHAIN and the EF-HYBRID model
	this->subgraphChain = subgraphChain;	// the reduced set of arcs
	this->y = y;							// the variables that the callback acts on
}

void Callback::setChainCallback(const vector<Subgraph>& subgraphsChain, const vector<vector<GRBVar>>& ySub) {
	// Sets the attributes of the Callback object needed to be able to handle the lazy constraints for the EEF-CHAIN model
	this->subgraphsChain = subgraphsChain;	// the reduced set of arcs
	this->ySub = ySub;						// the variables that the callback acts on
}

int Callback::getNCuts(bool ofCycleModel) {
	if (ofCycleModel) { return this->nCutsCycle; }	// return the number of cuts due to the cycle model
	else { return this->nCutsChain; }				// return the number of cuts due to the chain model
}

void Callback::callback() {
	// This function adds the lazy constraints
	if (where == GRB_CB_MIPSOL) { // if a new incumbent is found
		// cout << "A new incumbent has been found!\n";

		// Callback for the cycle model
		if (inst.K > 0) {
			if (modelCycle == "EF-CYCLE-OLD-CG") {
				vector<vector<int>> donationPerDonor(inst.nR);				// array containing the recipient r for each donor d that donates
				int xval, d, r;
				for (int a = 0; a < subgraphCycle.idx2Arc.size(); a++) {	// loop over all arcs
					xval = ceil(getSolution(x[a]) - EPSILON);				// find variable value
					if (xval == 1) {										// if the arc is selected
						d = subgraphCycle.idx2Arc[a][0]; r = subgraphCycle.idx2Arc[a][1]; // decompose the arc
						donationPerDonor[d] = { r };						// add the arc to the list of donations
					}
				}
				vector<int> cycle;							// array containing the arcs in the current cycle
				GRBLinExpr arcsUsedOnMinInfeasiblePath;		// the variables for the current minimal infeasible path
				for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
					if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
						cycle = {};							// start the cycle
						d = s;								// initialize the current donor
						while (true) {						// keep looping until the cycle is closed
							r = donationPerDonor[d][0]; 					// find recipient
							donationPerDonor[d].clear();					// remove the used arc
							cycle.push_back(subgraphCycle.arc2Idx[d][r]);	// add the arc to the cycle
							if (r == s) { break; }							// stop once the cycle is closed
							else { d = r; }							// else move to the next point
						}
						// Add lazy constraints if the cycle is too long
						if (cycle.size() > inst.K) {
							// printVector(cycle, "Cycle");
							arcsUsedOnMinInfeasiblePath = 0;
							for (int i = 0; i < inst.K; i++) {
								arcsUsedOnMinInfeasiblePath += x[cycle[i]];
							}
							addLazy(arcsUsedOnMinInfeasiblePath <= inst.K - 1);
							nCutsCycle++;
							for (int i = 0; i < cycle.size() - 1; i++) {
								arcsUsedOnMinInfeasiblePath -= x[cycle[i]];
								arcsUsedOnMinInfeasiblePath += x[cycle[(i + inst.K) % cycle.size()]];
								addLazy(arcsUsedOnMinInfeasiblePath <= inst.K - 1);
								nCutsCycle++;
							}
						}
					}
				}
			}

			else if (modelCycle == "EF-CYCLE-CG" || modelCycle == "EF-CYCLE-STRONG-CG") {
				vector<vector<int>> donationPerDonor(inst.nR);					// array containing the recipient r for each donor d that donates
				int xval, d, r, a;
				for (int a = 0; a < subgraphCycle.idx2Arc.size(); a++) {		// loop over all arcs
					xval = ceil(getSolution(x[a]) - EPSILON);					// find variable value
					if (xval == 1) {											// if the arc is selected
						d = subgraphCycle.idx2Arc[a][0]; r = subgraphCycle.idx2Arc[a][1]; // decompose the arc
						donationPerDonor[d] = { r };							// add the arc to the list of donations
					}
				}
				// print2DVector(donationPerDonor, "donationPerDonor", "donor");
				vector<int> cycle;							// array containing the vertices in the current cycle
				vector<int> maxFeasiblePath;				// the vertices in the current maximally cardinality feasible path
				GRBLinExpr arcsUsedOnMaxFeasiblePath;		// the variables for the current maximally cardinality feasible path
				for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
					if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
						cycle = { s };						// start the cycle
						d = s;								// initialize the current donor
						while (true) {						// keep looping until the cycle is closed
							r = donationPerDonor[d][0]; 	// find recipient
							donationPerDonor[d].clear();	// remove the used arc
							if (r != s) {					// if the new arc did not close the cycle yet
								d = r;						// move to the new point
								cycle.push_back(r);			// add the new point to the cycle
							}
							else { break; }					// stop once the cycle is closed
						}
						// Add lazy constraints if the cycle is too long
						if (cycle.size() > inst.K) {
							// printVector(cycle, "Cycle");
							for (int i = 0; i < cycle.size(); i++) {			// loop over all embedded paths of K vertices
								maxFeasiblePath = {};							// find the path starting at i
								arcsUsedOnMaxFeasiblePath = 0;
								for (int k = 0; k < inst.K; k++) {
									maxFeasiblePath.push_back(cycle[(i + k) % cycle.size()]);
								}
								// printVector(maxFeasiblePath, "MaxFeasiblePath");

								if (modelCycle == "EF-CYCLE-CG") { // non-lifted version
									for (int i = 0; i < maxFeasiblePath.size() - 1; i++) {
										d = maxFeasiblePath[i];
										r = maxFeasiblePath[i + 1];
										a = subgraphCycle.arc2Idx[d][r];
										arcsUsedOnMaxFeasiblePath += x[a];
									}
								}
								else if (modelCycle == "EF-CYCLE-STRONG-CG") { // lifted version
									for (int iD = 0; iD < inst.K - 1; iD++) {
										d = maxFeasiblePath[iD];
										for (int iR = iD + 1; iR < inst.K; iR++) {
											r = maxFeasiblePath[iR];
											a = subgraphCycle.arc2Idx[d][r];
											if (a != -1) { arcsUsedOnMaxFeasiblePath += x[a]; }
										}
									}
								}
								a = subgraphCycle.arc2Idx[maxFeasiblePath.back()][maxFeasiblePath.front()];
								if (a != -1) { arcsUsedOnMaxFeasiblePath -= x[a]; } // Note the minus sign here
								addLazy(arcsUsedOnMaxFeasiblePath <= inst.K - 2);
								nCutsCycle++;
							}
						}
					}
				}
			}
		}
		// Callback for the chain model
		if (inst.L > 0) {
			if (modelChain == "EF-CHAIN-EXP-CG" || modelChain == "EF-CHAIN-CUTSET-CG") {
				// Decompose the solution
				vector<vector<int>> donationPerDonor(inst.nR + inst.nN);					// array containing the recipient r for each donor d that donates
				int yval, d, r, a;
				for (int a = 0; a < subgraphChain.idx2Arc.size(); a++) {					// loop over all arcs
					yval = ceil(getSolution(y[a]) - EPSILON);								// find variable value
					if (yval == 1) {														// if the arc is selected
						d = subgraphChain.idx2Arc[a][0]; r = subgraphChain.idx2Arc[a][1];	// decompose the arc
						donationPerDonor[d] = { r };										// add the arc to the list of donations
					}
				}

				// Identify the cycles and the chains
				vector<int> chain;							// array containing the current chain
				GRBLinExpr arcsOnChain = 0;
				for (int s = inst.nR; s < inst.nR + inst.nN; s++) { // loop over all non-directed donors
					chain = { s };				// start the chain
					d = s;								// initialize the current donor
					while (donationPerDonor[d].size() > 0) { // keep looping until the cycle is closed
						r = donationPerDonor[d][0]; 	// find recipient and weight
						donationPerDonor[d].clear();	// remove the used arc
						chain.push_back(r);				// add the recipient to the chain
						d = r;							// the new donor is the recipient's donor
					}
					if (chain.size() > inst.L) {
						// printVector(chain, "Chain to be excluded");
						arcsOnChain = 0;
						for (int i = 1; i < inst.L; i++) {
							d = chain[i]; r = chain[i + 1]; a = subgraphChain.arc2Idx[d][r];
							arcsOnChain += y[a];
						}
						addLazy(arcsOnChain <= inst.L - 2);
						nCutsChain++;
						for (int i = 1; i < chain.size() - inst.L; i++) {
							d = chain[i]; r = chain[i + 1]; a = subgraphChain.arc2Idx[d][r];
							arcsOnChain -= y[a];
							d = chain[i + inst.L - 1]; r = chain[i + inst.L]; a = subgraphChain.arc2Idx[d][r];
							arcsOnChain += y[a];
							addLazy(arcsOnChain <= inst.L - 2);
							nCutsChain++;
						}
					}
				}

				vector<int> cycle;							// array containing the vertices in the current cycle
				GRBLinExpr arcsOnCycle = 0;
				GRBLinExpr arcsIntoCycle = 0;
				GRBLinExpr arcsIntoVertex = 0;
				for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
					if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
						cycle = { s };						// start the cycle
						d = s;								// initialize the current donor
						while (true) {						// keep looping until the cycle is closed
							r = donationPerDonor[d][0]; 	// find recipient
							donationPerDonor[d].clear();	// remove the used arc
							if (r != s) {					// if the new arc did not close the cycle yet
								d = r;						// move to the new point
								cycle.push_back(r);			// add the new point to the cycle
							}
							else { break; }					// stop once the cycle is closed
						}
						// Long cycles can be excluded using long chain elimination constraints
						if (cycle.size() >= inst.L) {
							// printVector(cycle, "Cycle to be excluded using chain size constraints");
							arcsOnCycle = 0;
							for (int i = 0; i < inst.L - 1; i++) {
								d = cycle[i]; r = cycle[i + 1]; a = subgraphChain.arc2Idx[d][r];
								arcsOnCycle += y[a];
							}
							addLazy(arcsOnCycle <= inst.L - 2);
							nCutsChain++;
							for (int i = 0; i < cycle.size() - 1; i++) {
								d = cycle[i]; r = cycle[i + 1]; a = subgraphChain.arc2Idx[d][r];
								arcsOnCycle -= y[a];
								d = cycle[(i + inst.L - 1) % cycle.size()]; r = cycle[(i + inst.L) % cycle.size()]; a = subgraphChain.arc2Idx[d][r];
								arcsOnCycle += y[a];
								addLazy(arcsOnCycle <= inst.L - 2);
								nCutsChain++;
							}
						}
						// Short cycles must be excluded with cycle elimination constraints specifically,
						// the exact way depends on the chosen model
						else if (modelChain == "EF-CHAIN-EXP-CG") {
							// printVector(cycle, "Cycle to be excluded using standard cycle constraint");
							arcsOnCycle = 0;
							for (int i = 0; i < cycle.size() - 1; i++) {
								d = cycle[i]; r = cycle[i + 1]; a = subgraphChain.arc2Idx[d][r];
								arcsOnCycle += y[a];
							}
							d = cycle.back(); r = cycle.front(); a = subgraphChain.arc2Idx[d][r];
							arcsOnCycle += y[a];
							addLazy(arcsOnCycle <= cycle.size() - 1);
							nCutsChain++;
						}

						else if (modelChain == "EF-CHAIN-CUTSET-CG") {
							// printVector(cycle, "Cycle to be excluded using cutset cycle constraint");
							vector<int> notCycle = findComplement(cycle, inst.nR + inst.nN);
							arcsIntoCycle = 0;
							// printA(subgraphEFCHAIN.A, inst.wToTau, "A_EFCHAIN");
							// print2DVector(subgraphEFCHAIN.arc2Idx, "A_EFCHAIN", "donor");
							// printVector(S, "S");
							// printVector(notS, "notS");
							for (int d : notCycle) {
								for (int r : cycle) {
									a = subgraphChain.arc2Idx[d][r];
									// cout << "d = " << d << ", r = " << r << ", a = " << a << "\n";
									if (a != -1) { arcsIntoCycle += y[a]; }
								}
							}
							for (int r : cycle) {
								arcsIntoVertex = 0;
								for (int d = 0; d < inst.nR + inst.nN; d++) {
									a = subgraphChain.arc2Idx[d][r];
									if (a != -1) { arcsIntoVertex += y[a]; }
								}
								addLazy(arcsIntoVertex <= arcsIntoCycle);
								nCutsChain++;
							}
						}
					}
				}
			}

			else if (modelChain == "EEF-CHAIN-EXP-CG" || modelChain == "EEF-CHAIN-CUTSET-CG") {
				// Decompose the solution
				vector<vector<int>> donationPerDonor(inst.nR + inst.nN);				// array containing the recipient r for each donor d that donates
				int yval, d, r, a;
				for (int l = 0; l < inst.nN; l++) {								// loop over all subgraphs
					for (int a = 0; a < subgraphsChain[l].idx2Arc.size(); a++) {			// loop over all arcs
						yval = ceil(getSolution(ySub[l][a]) - EPSILON);				// find variable value
						if (yval == 1) {										// if the arc is selected
							d = subgraphsChain[l].idx2Arc[a][0]; r = subgraphsChain[l].idx2Arc[a][1]; // decompose the arc
							donationPerDonor[d] = { r };						// add the arc to the list of donations
						}
					}
				}

				// First remove all chains
				for (int s = inst.nR; s < inst.nR + inst.nN; s++) { // loop over all non-directed donors
					d = s;								// initialize the current donor
					while (donationPerDonor[d].size() > 0) { // keep looping until the cycle is closed
						r = donationPerDonor[d][0]; 	// find recipient and weight
						donationPerDonor[d].clear();	// remove the used arc
						d = r;							// the new donor is the recipient's donor
					}
				}

				vector<int> cycle;							// array containing the vertices in the current cycle
				GRBLinExpr arcsOnCycle;
				GRBLinExpr arcsIntoCycle;
				GRBLinExpr arcsIntoVertex;
				for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
					if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
						cycle = { s };						// start the cycle
						d = s;								// initialize the current donor
						while (true) {						// keep looping until the cycle is closed
							r = donationPerDonor[d][0]; 	// find recipient
							donationPerDonor[d].clear();	// remove the used arc
							if (r != s) {					// if the new arc did not close the cycle yet
								d = r;						// move to the new point
								cycle.push_back(r);			// add the new point to the cycle
							}
							else { break; }					// stop once the cycle is closed
						}

						if (modelChain == "EEF-CHAIN-EXP-CG") {
							// printVector(cycle, "Cycle to be excluded using standard cycle constraint");
							arcsOnCycle = 0;
							for (int i = 0; i < cycle.size(); i++) {
								d = cycle[i]; r = cycle[(i + 1) % cycle.size()];
								for (int l = 0; l < inst.nN; l++) {
									a = subgraphsChain[l].arc2Idx[d][r];
									if (a != -1) { arcsOnCycle += ySub[l][a]; }
								}
							}
							addLazy(arcsOnCycle <= cycle.size() - 1);
							nCutsChain++;
						}

						else if (modelChain == "EEF-CHAIN-CUTSET-CG") {
							// printVector(cycle, "Cycle to be excluded using cutset cycle constraint");
							vector<int> notCycle = findComplement(cycle, inst.nR + inst.nN);
							arcsIntoCycle = 0;
							// printA(subgraphEFCHAIN.A, inst.wToTau, "A_EFCHAIN");
							// print2DVector(subgraphEFCHAIN.arc2Idx, "A_EFCHAIN", "donor");
							// printVector(S, "S");
							// printVector(notS, "notS");
							for (int d : notCycle) {
								for (int r : cycle) {
									for (int l = 0; l < inst.nN; l++) {
										a = subgraphsChain[l].arc2Idx[d][r];
										if (a != -1) { arcsIntoCycle += ySub[l][a]; }
									}
								}
							}
							for (int r : cycle) {
								arcsIntoVertex = 0;
								for (int d = 0; d < inst.nR + inst.nN; d++) {
									for (int l = 0; l < inst.nN; l++) {
										a = subgraphsChain[l].arc2Idx[d][r];
										if (a != -1) { arcsIntoVertex += ySub[l][a]; }
									}
								}
								addLazy(arcsIntoVertex <= arcsIntoCycle);
								nCutsChain++;
							}
						}
					}
				}
			}
		}

		// Callback for a hybrid model
		if (inst.K > 0 || inst.L > 0) {
			if (modelHybrid == "EF-HYBRID-CG" || modelHybrid == "EF-HYBRID-IMPROVED-CG") {
				// Decompose the solution
				vector<vector<int>> donationPerDonor(inst.nR + inst.nN);					// array containing the recipient r for each donor d that donates
				int yval, d, r, a;
				for (int a = 0; a < subgraphChain.idx2Arc.size(); a++) {					// loop over all arcs
					yval = ceil(getSolution(y[a]) - EPSILON);								// find variable value
					if (yval == 1) {														// if the arc is selected
						d = subgraphChain.idx2Arc[a][0]; r = subgraphChain.idx2Arc[a][1];	// decompose the arc
						donationPerDonor[d] = { r };										// add the arc to the list of donations
					}
				}
				// Identify the cycles and the chains
				vector<int> chain;							// array containing the current chain
				GRBLinExpr arcsOnChain = 0;
				GRBLinExpr startingArcs = 0;

				vector<int> maxFeasiblePath;				// the vertices in the current maximally cardinality feasible path
				GRBLinExpr arcsUsedOnMaxFeasiblePath;		// the variables for the current maximally cardinality feasible path
				vector<int> minInfeasiblePath;				// the vertices in the current minimal infeasible path
				GRBLinExpr arcsUsedOnMinInfeasiblePath;		// the variables for the current minimal infeasible path

				bool canStart;
				for (int s = inst.nR; s < inst.nR + inst.nN; s++) { // loop over all non-directed donors
					if (donationPerDonor[s].size() > 0) {
						chain = { s };						// start the chain
						d = s;								// initialize the current donor
						while (donationPerDonor[d].size() > 0) { // keep looping until the cycle is closed
							r = donationPerDonor[d][0]; 	// find recipient and weight
							donationPerDonor[d].clear();	// remove the used arc
							chain.push_back(r);				// add the recipient to the chain
							d = r;							// the new donor is the recipient's donor
						}
						// Exclude the long chains
						if (chain.size() > inst.L) {
							// printVector(chain, "Chain to be excluded");

							if (modelHybrid == "EF-HYBRID-CG") {
								arcsOnChain = 0;
								startingArcs = 0;
								for (int i = 1; i < inst.L; i++) { // loop over the first L-1 vertices (skipping the NDD vertex at the start)
									d = chain[i]; r = chain[i + 1]; a = subgraphChain.arc2Idx[d][r];
									arcsOnChain += y[a];
								}
								r = chain[1];
								for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
									a = subgraphChain.arc2Idx[n][r];
									if (a != -1) {
										startingArcs += y[a];
									}
								}
								addLazy(arcsOnChain + startingArcs <= inst.L - 1);
								nCutsChain++;

								// Optionally, add more than one cut
								for (int i = 1; i < chain.size() - inst.L; i++) {
									d = chain[i]; r = chain[i + 1]; a = subgraphChain.arc2Idx[d][r];
									arcsOnChain -= y[a];
									d = chain[i + inst.L - 1]; r = chain[i + inst.L]; a = subgraphChain.arc2Idx[d][r];
									arcsOnChain += y[a];
									r = chain[i + 1];
									canStart = false; startingArcs = 0;
									for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
										a = subgraphChain.arc2Idx[n][r];
										if (a != -1) {
											startingArcs += y[a];
											canStart = true;
										}
									}
									if (canStart) {
										addLazy(arcsOnChain + startingArcs <= inst.L - 1);
										nCutsChain++;
									}
								}
							}
							else if (modelHybrid == "EF-HYBRID-IMPROVED-CG") {

								if (inst.L == inst.K) { // Case 1
									for (int i = 1; i <= chain.size() - inst.L; i++) {	// loop over all embedded paths of L vertices
										maxFeasiblePath = {};							// find the path starting at i
										arcsUsedOnMaxFeasiblePath = 0;
										for (int l = 0; l < inst.L; l++) {
											maxFeasiblePath.push_back(chain[i + l]);
										}
										// printVector(maxFeasiblePath, "MaxFeasiblePath");

										for (int i = 0; i < maxFeasiblePath.size() - 1; i++) {
											d = maxFeasiblePath[i];
											r = maxFeasiblePath[i + 1];
											a = subgraphChain.arc2Idx[d][r];
											arcsUsedOnMaxFeasiblePath += y[a];
										}

										a = subgraphChain.arc2Idx[maxFeasiblePath.back()][maxFeasiblePath.front()];
										if (a != -1) { arcsUsedOnMaxFeasiblePath -= y[a]; } // Note the minus sign here
										addLazy(arcsUsedOnMaxFeasiblePath <= inst.L - 2);
										nCutsChain++;
									}
								}
								else if (inst.L >= inst.K + 1) { // Cases 2 and 3
									for (int i = 1; i <= chain.size() - inst.L; i++) {	// loop over all embedded paths of L vertices
										minInfeasiblePath = {};							// find the path starting at i
										arcsUsedOnMinInfeasiblePath = 0;
										for (int l = 0; l < inst.L; l++) {
											minInfeasiblePath.push_back(chain[i + l]);
										}
										// printVector(minInfeasiblePath, "MinInfeasiblePath");

										for (int i = 0; i < minInfeasiblePath.size() - 1; i++) {
											d = minInfeasiblePath[i];
											r = minInfeasiblePath[i + 1];
											a = subgraphChain.arc2Idx[d][r];
											arcsUsedOnMinInfeasiblePath += y[a];
										}
										addLazy(arcsUsedOnMinInfeasiblePath <= inst.L - 2);
										nCutsChain++;
									}

								}
							}
						}
					}
				}

				vector<int> cycle;							// array containing the vertices in the current cycle
				GRBLinExpr arcsOnCycle = 0;
				GRBLinExpr arcsIntoCycle = 0;
				GRBLinExpr arcsIntoVertex = 0;
				for (int s = 0; s < inst.nR; s++) {			// loop over all possible starting donors
					if (donationPerDonor[s].size() > 0) {	// if the donor actually donated
						cycle = { s };						// start the cycle
						d = s;								// initialize the current donor
						while (true) {						// keep looping until the cycle is closed
							r = donationPerDonor[d][0]; 	// find recipient
							donationPerDonor[d].clear();	// remove the used arc
							if (r != s) {					// if the new arc did not close the cycle yet
								d = r;						// move to the new point
								cycle.push_back(r);			// add the new point to the cycle
							}
							else { break; }					// stop once the cycle is closed
						}

						// Exclude the long cycles
						if (cycle.size() > inst.K) {
							// printVector(cycle, "Cycle to be excluded");

							if (modelHybrid == "EF-HYBRID-CG" || (modelHybrid == "EF-HYBRID-IMPROVED-CG" && inst.L > inst.K+1 && cycle.size() < inst.L)) {
								arcsOnCycle = 0;
								for (int i = 0; i < cycle.size() - 1; i++) {
									d = cycle[i]; r = cycle[i + 1]; a = subgraphChain.arc2Idx[d][r];
									arcsOnCycle += y[a];
								}
								d = cycle.back(); r = cycle.front(); a = subgraphChain.arc2Idx[d][r];
								arcsOnCycle += y[a];
								addLazy(arcsOnCycle <= cycle.size() - 1);
								nCutsCycle++;
							}

							else if (modelHybrid == "EF-HYBRID-IMPROVED-CG") {
								if (inst.L == inst.K) { // Case 1
									for (int i = 0; i < cycle.size(); i++) {	// loop over all embedded paths of K vertices
										maxFeasiblePath = {};					// find the path starting at i
										arcsUsedOnMaxFeasiblePath = 0;
										for (int k = 0; k < inst.K; k++) {
											maxFeasiblePath.push_back(cycle[(i + k) % cycle.size()]);
										}
										// printVector(maxFeasiblePath, "MaxFeasiblePath");

										for (int i = 0; i < maxFeasiblePath.size() - 1; i++) {
											d = maxFeasiblePath[i];
											r = maxFeasiblePath[(i + 1) % cycle.size()];
											a = subgraphChain.arc2Idx[d][r];
											arcsUsedOnMaxFeasiblePath += y[a];
										}

										a = subgraphChain.arc2Idx[maxFeasiblePath.back()][maxFeasiblePath.front()];
										if (a != -1) { arcsUsedOnMaxFeasiblePath -= y[a]; } // Note the minus sign here
										addLazy(arcsUsedOnMaxFeasiblePath <= inst.K - 2);
										nCutsCycle++;
									}
								}
								else if (inst.L == inst.K + 1 || inst.L > inst.K + 1 && cycle.size() >= inst.L) { // Case 2 and case 3 for large cycles
									for (int i = 0; i < cycle.size(); i++) {	// loop over all embedded paths of K+1 vertices
										minInfeasiblePath = {};					// find the path starting at i
										arcsUsedOnMinInfeasiblePath = 0;
										for (int k = 0; k < inst.L; k++) {
											minInfeasiblePath.push_back(cycle[(i + k) % cycle.size()]);
										}
										// printVector(minInfeasiblePath, "MinInfeasiblePath");

										for (int i = 0; i < minInfeasiblePath.size() - 1; i++) {
											d = minInfeasiblePath[i];
											r = minInfeasiblePath[(i + 1) % cycle.size()];
											a = subgraphChain.arc2Idx[d][r];
											arcsUsedOnMinInfeasiblePath += y[a];
										}
										addLazy(arcsUsedOnMinInfeasiblePath <= inst.L - 2);
										nCutsCycle++;
									}
									// for case 3 for medium cycles, see code at standard EF-HYBRID-CG
								}
							}
						}
					}
				}
			}
		}

	}
}

//---------------------------FUNCTIONS RELATING TO SUBGRAPH CREATION---------------------------
Subgraph createSubgraphEFCYCLE(const Instance& inst) {
	// Creates the subgraph required for EF-CYCLE

	// Find the shortest path distance between every pair of nodes
	vector<vector<int>> dist = floydsAlgorithm(inst, inst.A, false, false);

	Subgraph subgraph;
	int idx = 0;
	int r, w;
	subgraph.A.resize(inst.nR);
	subgraph.arc2Idx.resize(inst.nR, vector<int>(inst.nR, -1));
	for (int d = 0; d < inst.nR; d++) {
		for (int i = 0; i < inst.A[d].size(); i++) {
			r = inst.A[d][i][0]; w = inst.A[d][i][1];
			if (1 + dist[r][d] <= inst.K) {
				subgraph.A[d].push_back({ r,w });
				subgraph.arc2Idx[d][r] = idx;
				subgraph.idx2Arc.push_back({ d, r, w });
				idx++;
			}
		}
	}
	return subgraph;
}

Subgraph createSubgraphEEFCYCLE(const Instance& inst, const vector<vector<vector<int>>>& A, int l) {
	// Creates subgraph l for EEF-CYCLE
	// We do not need the .A and .arc2Idx attributes

	int r, w;
	vector<vector<vector<int>>> Al(inst.nR);                // the set of arcs induced by {v in R: v >= l}
	for (int d = l; d < inst.nR; d++) {						// start from l
		for (int i = 0; i < A[d].size(); i++) {
			r = A[d][i][0]; w = A[d][i][1];
			if (r >= l) {									// only vertices that are at least l
				Al[d].push_back({ r,w });
			}
		}
	}

	// Find the shortest distance from l to any other point and from any point to l
	vector<int> distFromStart = shortestDistancesFromPoint(inst, Al, l);
	vector<int> distToStart = shortestDistancesToPoint(inst, Al, l);

	Subgraph subgraph;
	// int idx = 0;
	// subgraph.A.resize(inst.nR);
	// subgraph.arc2Idx.resize(inst.nR, vector<int>(inst.nR, -1));
	for (int d = l; d < inst.nR; d++) {
		for (int i = 0; i < Al[d].size(); i++) {
			r = Al[d][i][0]; w = Al[d][i][1];
			if (distFromStart[d] + 1 + distToStart[r] <= inst.K) {
				// subgraph.A[d].push_back({ r,w });
				// subgraph.arc2Idx[d][r] = idx;
				subgraph.idx2Arc.push_back({ d, r, w });
				// idx++; // only needed for arc2idx attribute
			}
		}
	}
	return subgraph;
}

vector<Subgraph> createSubgraphPIEFCYCLE(const Instance& inst, const vector<vector<vector<int>>>& A, int l) {
	// Creates subgraph l for PIEF-CYCLE
	// We do not need the .A and .arc2Idx attributes

	int r, w;
	vector<vector<vector<int>>> Al(inst.nR);                // the set of arcs induced by {v in R: v >= l}
	for (int d = l; d < inst.nR; d++) {						// loop over all donors starting from l
		for (int i = 0; i < A[d].size(); i++) {             // loop over all arcs leaving d
			r = A[d][i][0]; w = A[d][i][1];                 // find the recipient and weight
			if (r >= l) {									// only vertices that are at least l
				Al[d].push_back({ r,w });                   // save the arc
			}
		}
	}

	// Find the shortest distance from l to any other point and from any point to l
	vector<int> distToStart = shortestDistancesToPoint(inst, Al, l);

	vector<Subgraph> subgraphs(inst.K + 1); // create a subgraph object for every possible position index
	vector<bool> activeDonors(inst.nR, false); activeDonors[l] = true;
	vector<bool> newActiveDonors(inst.nR, false);
	for (int k = 1; k <= inst.K; k++) {                 // loop over all possible position indices
		// printVector(activeDonors, "active donors at iteration " + to_string(k));
		for (int d = l; d < inst.nR; d++) {             // loop over all donors d activated in the previous layer
			if (activeDonors[d]) {
				for (int i = 0; i < Al[d].size(); i++) {    // loop over all arcs leaving d
					r = Al[d][i][0]; w = Al[d][i][1];       // find the recipient and the weight
					// cout << d << ", " << r << "\n";
					if (r == l) {										// if we are back at the start 
						subgraphs[k].idx2Arc.push_back({ d, r, w });    // add the arc, but don't activate the vertex
					}
					else if (distToStart[r] <= inst.K - k) {            // if we can still go back to the start from the new vertex
						subgraphs[k].idx2Arc.push_back({ d, r, w });    // add the arc
						newActiveDonors[r] = true;						// activate the vertex for next round
					}
				}
			}
		}
		activeDonors = newActiveDonors;
		fill(newActiveDonors.begin(), newActiveDonors.end(), false);
	}

	return subgraphs;
}

Subgraph createSubgraphEFCHAIN(const Instance& inst) {
	// Creates the subgraph required for EF-CHAIN

	// Find the shortest path distance between every pair of nodes
	vector<vector<int>> dist = floydsAlgorithm(inst, inst.A);

	Subgraph subgraph;
	int idx = 0;
	int r, w;
	vector<int> distN(inst.nR, inst.nR + 1);
	for (int r = 0; r < inst.nR; r++) {
		for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
			distN[r] = min(distN[r], dist[n][r]);
		}
	}
	subgraph.A.resize(inst.nR + inst.nN);
	subgraph.arc2Idx.resize(inst.nR + inst.nN, vector<int>(inst.nR, -1));
	if (inst.L > 1) {
		for (int d = 0; d < inst.nR; d++) {
			for (int i = 0; i < inst.A[d].size(); i++) {
				r = inst.A[d][i][0]; w = inst.A[d][i][1];
				if (distN[d] + 1 <= inst.L - 1) {
					subgraph.A[d].push_back({ r,w });
					subgraph.arc2Idx[d][r] = idx;
					subgraph.idx2Arc.push_back({ d, r, w });
					idx++;
				}
			}
		}
		for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
			for (int i = 0; i < inst.A[n].size(); i++) {
				r = inst.A[n][i][0]; w = inst.A[n][i][1];
				subgraph.A[n].push_back({ r,w });
				subgraph.arc2Idx[n][r] = idx;
				subgraph.idx2Arc.push_back({ n, r, w });
				idx++;
			}
		}
	}
	return subgraph;
}

Subgraph createSubgraphEEFCHAIN(const Instance& inst, int l) {
	// Creates subgraph l for EEF-CHAIN
	// We do not need the .A attribute

	// Find the shortest distance from l to all RDPs
	vector<int> distFromRoot = shortestDistancesFromPoint(inst, inst.A, l);

	Subgraph subgraph;
	int idx = 0;
	int r, w;
	// subgraph.A.resize(inst.nR+inst.nN);
	subgraph.arc2Idx.resize(inst.nR + inst.nN, vector<int>(inst.nR, -1));
	for (int d = 0; d < inst.nR; d++) {
		if (distFromRoot[d] + 1 <= inst.L - 1) {
			for (int i = 0; i < inst.A[d].size(); i++) {
				r = inst.A[d][i][0]; w = inst.A[d][i][1];
				// subgraph.A[d].push_back({ r,w });
				subgraph.arc2Idx[d][r] = idx;
				subgraph.idx2Arc.push_back({ d, r, w });
				idx++;
			}
		}
	}
	if (inst.L > 1) {
		for (int i = 0; i < inst.A[l].size(); i++) {
			r = inst.A[l][i][0]; w = inst.A[l][i][1];
			// subgraph.A[l].push_back({ r,w });
			subgraph.arc2Idx[l][r] = idx;
			subgraph.idx2Arc.push_back({ l, r, w });
			idx++;
		}
	}
	return subgraph;
}

vector<Subgraph> createSubgraphPIEFCHAIN(const Instance& inst) {
	// Creates the position indexed subgraph for PIEF-CHAIN (one subgraph per position)
	// We do not need the .A and .arc2Idx attributes

	vector<Subgraph> subgraphs(inst.L); // create a subgraph object for every possible position index
	vector<bool> activeDonors(inst.nR+inst.nN, false);
	vector<bool> newActiveDonors(inst.nR + inst.nN, false);
	// We start with all NDD's active
	for (int n = inst.nR; n < inst.nR + inst.nN; n++) {activeDonors[n] = true;}

	int r, w;
	for (int k = 1; k <= inst.L - 1; k++) {                 // loop over all possible position indices k
		// printVector(activeDonors, "active donors at iteration " + to_string(k));
		for (int d = 0; d < inst.nR + inst.nN; d++) {       // loop over all donors d activated in the previous layer
			if (activeDonors[d]) {
				for (int i = 0; i < inst.A[d].size(); i++) {    // loop over all arcs leaving d
					r = inst.A[d][i][0]; w = inst.A[d][i][1];   // find the recipient and the weight
					// cout << d << ", " << r << "\n";
					subgraphs[k].idx2Arc.push_back({ d, r, w });	 // add the arc
					newActiveDonors[r] = true;			             // activate the vertex for next round
				}
			}
		}
		activeDonors = newActiveDonors;
		fill(newActiveDonors.begin(), newActiveDonors.end(), false);
	}
	return subgraphs;
}

Subgraph createSubgraphEFHYBRID(const Instance& inst) {
	// Creates the subgraph required for EF-HYBRID

	// Find the shortest path distance between every pair of nodes
	vector<vector<int>> dist = floydsAlgorithm(inst, inst.A);

	Subgraph subgraph;
	int idx = 0;
	int r, w;
	vector<int> distN(inst.nR, inst.nR + 1);
	for (int r = 0; r < inst.nR; r++) {
		for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
			distN[r] = min(distN[r], dist[n][r]);
		}
	}
	subgraph.A.resize(inst.nR + inst.nN);
	subgraph.arc2Idx.resize(inst.nR + inst.nN, vector<int>(inst.nR, -1));
	for (int d = 0; d < inst.nR; d++) {
		for (int i = 0; i < inst.A[d].size(); i++) {
			r = inst.A[d][i][0]; w = inst.A[d][i][1];
			if (1 + dist[r][d] <= inst.K || distN[d] + 1 <= inst.L - 1) {
				subgraph.A[d].push_back({ r,w });
				subgraph.arc2Idx[d][r] = idx;
				subgraph.idx2Arc.push_back({ d, r, w });
				idx++;
			}
		}
	}
	if (inst.L > 0) {
		for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
			for (int i = 0; i < inst.A[n].size(); i++) {
				r = inst.A[n][i][0]; w = inst.A[n][i][1];
				subgraph.A[n].push_back({ r,w });
				subgraph.arc2Idx[n][r] = idx;
				subgraph.idx2Arc.push_back({ n, r, w });
				idx++;
			}
		}
	}
	return subgraph;
}

//---------------------------FUNCTIONS RELATING TO CONSTRUCTING LISTS OF CYCLES AND CHAINS---------------------------
vector<vector<vector<int>>> findCycles(const Instance& inst) {
	// This function returns all cycles of length at most K, including their weights
	// Every cycle is represented in the form {{d_1, d_2, ..., d_k}, rho_c}

	// Find the shortest path distance between every pair of nodes
	vector<vector<int>> dist = floydsAlgorithm(inst, inst.A);

	// Create cycles
	vector<vector<vector<int>>> cycles;								// list of all cycles of length at most K
	vector<vector<vector<int>>> curC;								// list of current partial cycles starting at s
	vector<vector<vector<int>>> newC;								// set of new partial cycles that will contain an additional vertex
	int d, r, w, w_c;
	bool add;
	for (int s = 0; s < inst.nR; s++) {								// for all vertices s, enumerate cycles starting at s with all other vertices > s
		curC.push_back({ {s}, {0} });								// start with just {s}
		for (int k = 1; k <= inst.K; k++) {							// we can add up to K points
			newC.clear();
			for (int c = 0; c < curC.size(); c++) {					// loop over all current partial cycles containing k vertices (so length k-1)
				d = curC[c][0].back();								// find the final donor d in the current partial cycle
				w_c = curC[c][1][0];								// find the weight of the current partial cycle
				for (int l = 0; l < inst.A[d].size(); l++) {		// for all recipients compatible with the donor d			
					r = inst.A[d][l][0];							// find the current recipient r
					w = inst.A[d][l][1];							// find the weight of arc (d, r)
					if (r == s) {									// if we are back at the start then this is a cycle
						cycles.push_back({ curC[c][0], {w_c + w} });// add the cycle that we thus created (don't re-add the initial vertex)
					}
					else if (r > s && dist[r][s] <= inst.K - k) {	// else, only consider recipients r with index more than s such that the distance from r back to s is small enough
						add = true;									// consider adding recipient r to the partial cycle
						for (int p = 1; p < k; p++) {				// loop over all k predecessors of recipient r
							if (curC[c][0][p] == r) {				// if recipient r is already present in the current partial cycle
								add = false;						// don't add recipient r after all
								break;
							}
						}
						if (add) {									// if we decided to add recipient r
							newC.push_back({ curC[c] });			// add the new partial cycle containing k+1 vertices (so length k)
							newC.back()[0].push_back(r);			// including recipient r as the k+1'th vertex
							newC.back()[1] = { w_c + w };			// also add the weight of the new arc
						}
					}
				}
			}
			curC = newC;											// the current set of partial cycles now consists of the partial cycles containing j+1 vertex (so length j)
		}
	}
	return cycles;
}

vector<vector<vector<int>>> findNonDominatedCycles(const Instance& inst) {
	// This function returns all non-dominated cycles of length at most K, including their weights
	// Every cycle is represented in the form {{d_1, d_2, ..., d_k}, rho_c}

	// Find the shortest path distance between every pair of nodes
	vector<vector<int>> dist = floydsAlgorithm(inst, inst.A);

	// Create cycles				
	vector<vector<vector<int>>> cycles;								// list of all non-dominated cycles of length at most K
	map<vector<int>, vector<vector<int>>> cycleMap;					// map containing all non-dominated cycles of length at most K
	vector<vector<vector<int>>> curC;								// list of current partial cycles starting at s
	vector<vector<vector<int>>> newC;								// set of new partial cycles that will contain an additional vertex
	vector<int> vertInCycle;										// ordered set of vertices present in current cycle
	int d, r, w, w_c;
	bool add;
	for (int s = 0; s < inst.nR; s++) {								// for all vertices s, enumerate cycles starting at s with all other vertices > s
		curC.push_back({ {s}, {0} });								// start with just {s}
		for (int k = 1; k <= inst.K; k++) {							// we can add up to K points
			newC.clear();
			for (int c = 0; c < curC.size(); c++) {					// loop over all current partial cycles containing k vertices (so length k-1)
				d = curC[c][0].back();								// find the final donor d in the current partial cycle
				w_c = curC[c][1][0];								// find the weight of the current partial cycle
				for (int l = 0; l < inst.A[d].size(); l++) {		// for all recipients compatible with the donor d			
					r = inst.A[d][l][0];							// find the current recipient r
					w = inst.A[d][l][1];							// find the weight of arc (d, r)
					if (r == s) {									// if we are back at the start then this is a cycle

						// original line:
						// cycles.push_back({ curC[c][0], {w_c + w} });// add the cycle that we thus created (don't re-add the initial vertex)
						
						vertInCycle = curC[c][0];							// find the vertices that are in the cycle
						sort(vertInCycle.begin(), vertInCycle.end());		// sort the vertices from smallest to largest
						auto it = cycleMap.find(vertInCycle);				// determine if there was already a cycle with the same set of vertices
						if (it != cycleMap.end()) {							// if there was already a cycle with the same set of vertices
							if (w_c + w > it->second[1][0]) {				// if the new cycle has a higher weight
								it->second = { curC[c][0], { w_c + w } };	// save the new cycle instead
							}
						}
						else {													// if there was no cycle yet with the current set of vertices
							cycleMap[vertInCycle] = { curC[c][0], {w_c + w} };	// save the new cycle
						}

					}
					else if (r > s && dist[r][s] <= inst.K - k) {	// else, only consider recipients r with index more than s such that the distance from r back to s is small enough
						add = true;									// consider adding recipient r to the partial cycle
						for (int p = 1; p < k; p++) {				// loop over all k predecessors of recipient r
							if (curC[c][0][p] == r) {				// if recipient r is already present in the current partial cycle
								add = false;						// don't add recipient r after all
								break;
							}
						}
						if (add) {									// if we decided to add recipient r
							newC.push_back({ curC[c] });			// add the new partial cycle containing k+1 vertices (so length k)
							newC.back()[0].push_back(r);			// including recipient r as the k+1'th vertex
							newC.back()[1] = { w_c + w };			// also add the weight of the new arc
						}
					}
				}
			}
			curC = newC;											// the current set of partial cycles now consists of the partial cycles containing j+1 vertex (so length j)
		}
	}

	// save the set of cycles as a vector instead
	// map<vector<int>, vector<vector<int>>>::iterator it;
	for (map<vector<int>, vector<vector<int>>>::iterator it = cycleMap.begin(); it != cycleMap.end(); it++) {
		cycles.push_back(it->second);
	}


	return cycles;
}

vector<vector<vector<int>>> findHalfCycles(const Instance& inst, const vector<vector<vector<int>>>& A) {
	// This function returns all half-cycles based on max cycle length K
	// Every half-cycle is represented in the form {{d_1, d_2, ..., d_k}, rho_h}

	// A half-cycles <i, ..., j> containing k vertices must satisfy:
	// a) 2 <= k <= Max = 1 + ceil(K/2)
	// b) either i or j is the lowest indexed vertex (it can be only i when K is odd and k = Max)
	// c) there exists a half-cycle <j, ..., i> containing k or k-1 vertices (if i<j) and k or k+1 vertices (if i>j) 

	// Find shortest distance from every vertex to every other vertex
	vector<vector<int>> floyd = floydsAlgorithm(inst, A, false, false);
	// print2DVector(floyd, "distances", "vertex");

	// Create an adjacency matrix containing weights
	int r, w;
	vector<vector<int>> Aw(inst.nR, vector<int>(inst.nR));
	for (int d = 0; d < inst.nR; d++) {
		for (int i = 0; i < A[d].size(); i++) {
			r = A[d][i][0]; w = A[d][i][1];
			Aw[d][r] = w;
		}
	}

	// Step 1: Create half-cycles according to condition a) and b) (and partly c))
	vector<vector<int>> curH;					// list of potential half-cycles containing k vertices
	vector<vector<int>> newH;					// list of potential half-cycles containing k+1 vertices
	vector<vector<int>> allH;					// list of all potential half-cycles
	vector<vector<vector<bool>>> canBeClosed(3 + ceil(double(inst.K) / 2), vector<vector<bool>>(inst.nR, vector<bool>(inst.nR, false)));	// boolean array such that canBeClosed[k][s][e] == true iff there is a half-cycle containing k vertices starting at s and ending at e

	for (int s = 0; s < inst.nR; s++) {			// all vertices s could be starting points of half-cycles;
		if (floyd[s][s] <= inst.K) {			// but only if the shortest path distance back to itself is at most K
			curH.push_back({ s });
		}
	}
	// print2DVector(curH, "curH", "half-cycle");

	vector<int> h; int s, e, minPos, minVer; bool add;

	for (int k = 1; k <= ceil(double(inst.K) / 2); k++) {	// add at most ceil(inst.K /2) points
		for (int i = 0; i < curH.size(); i++) {				// loop over all half-cycles of size k
			h = curH[i]; s = h.front(); e = h.back();		// find the current half-cycle h, its starting point s and ending point e
			for (int j = 0; j < A[e].size(); j++) {			// loop over all recipients of donor e
				r = A[e][j][0];								// call the recipient r
				if (floyd[r][s] <= inst.K - k) {			// only consider adding r if the distance back to s isn't too long
					add = true;								// only add r if it wasn't in the half-cycle yet
					minPos = k;	minVer = r;					// also save the position and index of the lowest indexed vertex in the half-cycle
					for (int p = 0; p < k; p++) {			// loop over all k predecessors of recipient r
						if (h[p] == r) {					// if recipient r is already present in the current partial cycle
							add = false;					// don't add recipient r after all
							break;
						}
						else if (h[p] < minVer) {			// if the current vertex index is lower than the current minimum
							minPos = p;	minVer = h[p];		// update the minimum
						}
					}
					if (add) {
						newH.push_back(h); 						// add the half-cycle to the list of half cycles containing k+1 vertices...
						newH.back().push_back(r);				// .. including r
						if (minPos == 0 || (minPos == k && (k < ceil(double(inst.K) / 2) || inst.K % 2 == 0))) { // if condition b is satisfied
							allH.push_back(newH.back());		// ... also add the new half-cycle to the list of all half-cycles
							canBeClosed[k + 1][s][r] = true;
						}
					}
				}
			}
		}
		curH = newH;
		newH.clear();
		// print2DVector(curH, "curH", "half-cycle");
	}

	// print2DVector(allH, "allH", "half-cycle");

	//for (int k = 2; k < canBeClosed.size()-1; k++) {
	//	print2DVector(canBeClosed[k], "canBeClosed[" + to_string(k) + "]", "s");
	//}

	// Step 2: Only keep half-cycles that satisfy condition c) and find the weight of those half-cycles
	vector<vector<vector<int>>> halfCycles;							// reduced list of all half-cycles, containing also weights
	int k, rho;														// size and weight of currently considered half-cycle
	for (int i = 0; i < allH.size(); i++) {							// loop over all potential half-cycles
		h = allH[i]; s = h.front(); e = h.back(); k = h.size();		// decompose the half-cycle
		if (canBeClosed[k][e][s] || (s < e && canBeClosed[k - 1][e][s]) || (s > e && canBeClosed[k + 1][e][s])) { // if the half-cycle can be closed by another half-cycle of appropriate length
			rho = 0;							// find the weight of the half-cycle
			for (int j = 0; j < k - 1; j++) {	// loop over the half-cycle
				rho += Aw[h[j]][h[j + 1]];		// add the weights of the arcs
			}
			halfCycles.push_back({ h, {rho} });	// add the half-cycle to the list of all half-cycles
		}
	}

	return halfCycles;
}

vector<vector<vector<int>>> findChains(const Instance& inst) {
	// This function returns all chains of length at most L
	// Every cycle is represented in the form {{n_0, d_1, d_2, ..., d_k}, rho_c}

	vector<vector<vector<int>>> chains;					// list of all chains of length at most L
	vector<vector<vector<int>>> curC;					// all chains of length l (initially l = 1)
	if (inst.L >= 1) {
		for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
			// chains.push_back({ {n}, {inst.wToTau[n]} });
			// chains.push_back({ {n}, {0} });
			curC.push_back({ {n}, {0} });
		}
	}
	if (inst.L >= 2) {
		int d, r, w, w_c;
		bool add;
		vector<vector<vector<int>>> newC;						// all chains of length l+1 (initially l = 1)
		vector<int> newChain;									// new chain
		for (int l = 1; l <= inst.L - 1; l++) {					// we can add up to L-1 points (a terminal node is implicitly added)
			for (int c = 0; c < curC.size(); c++) {				// loop over all current chains c
				d = curC[c][0].back();							// find the final donor d in the current chain c
				w_c = curC[c][1][0];							// find the weight of the current chain
				for (int i = 0; i < inst.A[d].size(); i++) {	// for all recipients compatible with the donor d			
					r = inst.A[d][i][0];						// find the current recipient r
					w = inst.A[d][i][1];						// find the weight of arc (d, r)
					add = true;									// consider adding recipient r to the chain c
					for (int p = 1; p < l; p++) {				// loop over all l predecessors of recipient r
						if (curC[c][0][p] == r) {				// if recipient r is already present in the current chain c
							add = false;						// don't add recipient r after all
							break;
						}
					}
					if (add) {
						newChain = curC[c][0]; newChain.push_back(r);					// find the new chain obtained by adding r
						// chains.push_back({ newChain, {w_c + w + inst.wToTau[r]} });	// add the new chain to the set of all chains, also add the weight from the final vertex to tau
						chains.push_back({ newChain, {w_c + w} });						// add the new chain to the set of all chains
						newC.push_back({ newChain, {w_c + w} });						// add the new chain to the set of all chains of length l+1
					}
				}
			}
			curC = newC;			// update the set of chains of length l to the set of length l+1
			newC.clear();
		}
	}
	return chains;
}

vector<vector<vector<int>>> findHalfChains(const Instance& inst, bool useLateSplit, bool useAlternative) {
	// This function returns all half-chains based on max chain length L
	// Every half-chain is represented in the form {{d_1, d_2, ..., d_l}, rho_h}

	vector<vector<int>> curH;					// list of potential half-chains containing l vertices
	vector<vector<int>> newH;					// list of potential half-chains containing l+1 vertices
	vector<vector<int>> allH1;					// list of all potential first half-chains
	vector<vector<int>> allH2;					// list of all potential second half-chains
	vector<bool> isReachable(inst.nR);			// contains for each RDP whether it is reachable by a first half-chain
	vector<vector<vector<int>>> halfChains;		// (reduced) list of all half-chains, containing also weights
	if (inst.L <= 1) {return halfChains;}		// return an empty list if L is trivially low

	int limitLenH1;
	if (useLateSplit) { limitLenH1 = ceil(double(inst.L) / 2) + 1; }
	else { limitLenH1 = floor(double(inst.L) / 2) + 1; }
	int limitLenH2;
	if (useLateSplit) { limitLenH2 = floor(double(inst.L) / 2); }
	else { limitLenH2 = ceil(double(inst.L) / 2); }
	vector<vector<bool>> canEnd(limitLenH1+2, vector<bool>(inst.nR, false));	// indicates for each RDP whether a first half-chain with l vertices can end at that RDP
	vector<vector<bool>> canStart(limitLenH2+2, vector<bool>(inst.nR, false));	// indicates for each RDP whether a second half-chain with l vertices can start at that RDP
	vector<int> h;
	int s, e, r;
	bool add;

	// Create an adjacency matrix containing weights
	int w;
	vector<vector<int>> Aw(inst.nR+inst.nN, vector<int>(inst.nR));
	for (int d = 0; d < inst.nR+inst.nN; d++) {
		for (int i = 0; i < inst.A[d].size(); i++) {
			r = inst.A[d][i][0]; w = inst.A[d][i][1];
			Aw[d][r] = w;
		}
	}

	// Step 1. Construct first halves (starting at vertices in N and containing at most floor(L/2)+1 vertices and floor(L/2) arcs)
	//	when a late split is used, the first half can contain at most ceil(L/2)+1 vertices and ceil(L/2) arcs
	for (int n = inst.nR; n < inst.nR + inst.nN; n++) {
		curH.push_back({ n });
	}
	for (int l = 1; l <= limitLenH1-1; l++) {	// add at most limitLenH1-1 extra vertices
		for (int i = 0; i < curH.size(); i++) {				// loop over all half-chains of size l
			h = curH[i]; e = h.back();						// find the current half-chain h, and its ending point e
			for (int j = 0; j < inst.A[e].size(); j++) {	// loop over all recipients of donor e
				r = inst.A[e][j][0];						// call the recipient r
				add = true;									// only add r if it wasn't in the half-chain yet
				for (int p = 1; p < l; p++) {				// loop over all l predecessors of recipient r
					if (h[p] == r) {						// if recipient r is already present in the current partial chain
						add = false;						// don't add recipient r after all
						break;
					}
				}
				if (add) {
					newH.push_back(h); 						// add the half-chain to the list of half chains containing l+1 vertices...
					newH.back().push_back(r);				// .. including r
					allH1.push_back(newH.back());			// ... also add the new half-chain to the list of all first half-chains
					canEnd[l + 1][r] = true;				// r can be the l+1'th vertex of a first half-chain
					if (not useAlternative || l == limitLenH1-1) {
						isReachable[r] = true;					// save that r can be reached by a first half-chain
					}
				}
			}
		}
		curH = newH;	// Update list of half-chains containing l' = l+1 vertices 
		newH.clear();	// Update list of half-chains containing l'+1 = l+2 vertices
	}


	// Step 2. Construct second halves (starting at reachable vertices in R and containing at most ceil(L/2) vertices and ceil(L/2)-1 arcs (not counting tau))
	//	if a late split is used, the second half can contain at most floor(L/2) vertices and floor(L/2)-1 arcs
	curH.clear();
	for (int d = 0; d < inst.nR; d++) {
		if (isReachable[d]) {
			curH.push_back({ d });
			allH2.push_back({ d });
			canStart[1][d] = true;							// d can be the first vertex of a second half-chain
		}
	}
	for (int l = 1; l <= limitLenH2 - 1; l++) {				// add at most limitLenH2-1 extra vertices
		for (int i = 0; i < curH.size(); i++) {				// loop over all half-chains of size l
			h = curH[i]; s = h.front(); e = h.back();		// find the current half-chain h, and its ending point e
			for (int j = 0; j < inst.A[e].size(); j++) {	// loop over all recipients of donor e
				r = inst.A[e][j][0];						// call the recipient r
				add = true;									// only add r if it wasn't in the half-chain yet
				for (int p = 0; p < l; p++) {				// loop over all l predecessors of recipient r
					if (h[p] == r) {						// if recipient r is already present in the current partial chain
						add = false;						// don't add recipient r after all
						break;
					}
				}
				if (add) {
					newH.push_back(h); 						// add the half-chain to the list of half chains containing l+1 vertices...
					newH.back().push_back(r);				// .. including r
					allH2.push_back(newH.back());			// ... also add the new half-chain to the list of all second half-chains
					canStart[l + 1][s] = true;				// s can be the first vertex of a second half-chain of length 
				}
			}
		}
		curH = newH;	// Update list of half-chains containing l' = l+1 vertices 
		newH.clear();	// Update list of half-chains containing l'+1 = l+2 vertices
	}

	//print2DVector(allH1, "allH1", "half-chain");
	//print2DVector(allH2, "allH2", "half-chain");
	//print2DVector(canEnd, "firstHalfCanEnd", "size");
	//print2DVector(canStart, "secondHalfCanStart", "size");

	// trivial half-chains NDD-tau should be dealt with seperately (i.e., add the w if NDD not used)

	// in the standard version, the first half must be combined with second half of same number of vertices or one less (not counting tau)
	int l, rho;										// size and weight of currently considered half-chain
	for (int i = 0; i < allH1.size(); i++) {		// loop over all potential first half-chains
		h = allH1[i]; e = h.back(); l = h.size();	// decompose the half-chain
		if ((not(useAlternative) && not(useLateSplit) && (canStart[l][e] || canStart[l - 1][e])) 
			|| (not(useAlternative) && useLateSplit && (canStart[l - 1][e] || canStart[l - 2][e]))
			|| useAlternative) { // if the half-chain can be continued with a second half-chain of appropriate length (or the alternative method is used)		
			rho = 0;								// find the weight of the half-chain
			for (int j = 0; j < l - 1; j++) {		// loop over the half-chain
				rho += Aw[h[j]][h[j + 1]];			// add the weights of the arcs
			}
			halfChains.push_back({ h, {rho} });		// add the half-chain to the list of all first half-chains
		}
	}
	for (int i = 0; i < allH2.size(); i++) {		// loop over all potential second half-chains
		h = allH2[i]; s = h.front(); l = h.size();	// decompose the half-chain
		if ((not(useAlternative) && not(useLateSplit) && (canEnd[l][s] || canEnd[l + 1][s]))
			|| (not(useAlternative) && useLateSplit && (canEnd[l + 1][s] || canEnd[l + 2][s]))
			|| (useAlternative && l > 1)) {			// if the half-chain can be started by a first half-chain of appropriate length (or the alternative method is used)
			rho = 0;								// find the weight of the half-chain
			for (int j = 0; j < l - 1; j++) {		// loop over the half-chain
				rho += Aw[h[j]][h[j + 1]];			// add the weights of the arcs
			}
			// rho += inst.wToTau[h.back()];			// also add the weight of the terminal arc
			halfChains.push_back({ h, {rho} });		// add the half-chain to the list of all second half-chains
		}
	}

	return halfChains;
}

vector<vector<int>> findMinInfeasibleChains(const Instance& inst, const vector<vector<vector<int>>>& A, bool limitedStart) {
	// This function returns all chains that
	// - start at some vertex in R, and if limitedStart=true: that vertex must be adjacent to a vertex in N 
	// - contain L vertices (excluding the starting vertex from N and tau) 
	// - contain L-1 arcs (excluding the starting and ending arc)

	vector<vector<int>> curC;							// all chains of l vertices (initially l = 1)
	vector<bool> isPossibleStart(inst.nR, true*(1-limitedStart));
	int r;
	if (inst.L >= 1) {
		if (limitedStart) {
			for (int n = inst.nR; n < inst.nR + inst.nN; n++) { // loop over all NDDs
				for (int i = 0; i < A[n].size(); i++) {			// loop over all recipients of n
					r = A[n][i][0];								// find recipient
					isPossibleStart[r] = true;					// the recipient can 'start' a chain
				}
			}
		}
		for (int d = 0; d < inst.nR; d++) {
			if (isPossibleStart[d]) {
				curC.push_back({ d });
			}
		}
	}
	if (inst.L >= 2) {
		int d, r;
		bool add;
		vector<vector<int>> newC;								// all chains of length l+1 (initially l = 1)
		vector<int> newChain;									// new chain
		for (int l = 1; l <= inst.L - 1; l++) {					// we can add up to L-1 points (a terminal node is implicitly added)
			for (int c = 0; c < curC.size(); c++) {				// loop over all current chains c
				d = curC[c].back();								// find the final donor d in the current chain c
				for (int i = 0; i < A[d].size(); i++) {	// for all recipients compatible with the donor d			
					r = A[d][i][0];						// find the current recipient r
					add = true;									// consider adding recipient r to the chain c
					for (int p = 0; p < l; p++) {				// loop over all l predecessors of recipient r
						if (curC[c][p] == r) {					// if recipient r is already present in the current chain c
							add = false;						// don't add recipient r after all
							break;
						}
					}
					if (add) {
						newChain = curC[c]; newChain.push_back(r);	// find the new chain obtained by adding r
						newC.push_back(newChain);		// add the new chain to the set of all chains of length l+1
					}
				}
			}
			curC = newC;			// update the set of chains of length l to the set of length l+1
			newC.clear();
		}
	}
	// print2DVector(curC, "chains", "chain");
	return curC;
}

vector<vector<int>> findShortCycles(const Instance& inst, const vector<vector<vector<int>>>& A, int len) {
	// This function returns all cycles of length at most len (len=L-1 for EF-CHAIN and l=L-2 for EEF-CHAIN)
	// Every cycle is represented in the form {d_1, d_2, ..., d_k}

	vector<vector<int>> floyd = floydsAlgorithm(inst, A, false, false);
	// print2DVector(floyd, "distances", "vertex");

	// Create cycles
	vector<vector<int>> cycles;								// list of all cycles of length at most len
	vector<vector<int>> curC;								// list of current partial cycles starting at s
	vector<vector<int>> newC;								// set of new partial cycles that will contain an additional vertex
	int d, r;
	bool add;
	for (int s = 0; s < inst.nR; s++) {							// for all vertices s, enumerate cycles starting at s with all other vertices > s
		curC.push_back({s});									// start with just {s}
		for (int k = 1; k <= len+1; k++) {						// we can add up to len points
			newC.clear();
			for (int c = 0; c < curC.size(); c++) {				// loop over all current partial cycles containing k vertices (so length k-1)
				d = curC[c].back();								// find the final donor d in the current partial cycle
				for (int l = 0; l < A[d].size(); l++) {			// for all recipients compatible with the donor d			
					r = A[d][l][0];								// find the current recipient r
					if (r == s) {								// if we are back at the start then this is a cycle
						cycles.push_back(curC[c]);				// add the cycle that we thus created (don't re-add the initial vertex)
					}
					else if (r > s && floyd[r][s] <= len- k) {	// else, only consider recipients r with index more than s such that the distance from r back to s is small enough
						add = true;									// consider adding recipient r to the partial cycle
						for (int p = 1; p < k; p++) {				// loop over all k predecessors of recipient r
							if (curC[c][p] == r) {					// if recipient r is already present in the current partial cycle
								add = false;						// don't add recipient r after all
								break;
							}
						}
						if (add) {								// if we decided to add recipient r
							newC.push_back({ curC[c] });		// add the new partial cycle containing k+1 vertices (so length k)
							newC.back().push_back(r);			// including recipient r as the k+1'th vertex
						}
					}
				}
			}
			curC = newC;											// the current set of partial cycles now consists of the partial cycles containing j+1 vertex (so length j)
		}
	}
	return cycles;
}

vector<vector<int>> findLongCycles(const Instance& inst, const vector<vector<vector<int>>>& A, int len, int lenMax) {
	// This function returns all cycles of length more than len (and possibly less than lenMax)
	// Every cycle is represented in the form {d_1, d_2, ..., d_k}

	if (lenMax == 0) { lenMax = inst.nR + 1; }
	vector<vector<int>> floyd = floydsAlgorithm(inst, A, false, false);
	// print2DVector(floyd, "distances", "vertex");

	// Create cycles
	vector<vector<int>> cycles;								// list of all cycles of length more than len and less than lenMax
	if (lenMax <= len + 1) { return cycles; }				
	vector<vector<int>> curC;								// list of current partial cycles starting at s
	vector<vector<int>> newC;								// set of new partial cycles that will contain an additional vertex
	int d, r;
	bool add;
	for (int s = 0; s < inst.nR; s++) {							// for all vertices s, enumerate cycles starting at s with all other vertices > s
		curC.push_back({ s });									// start with just {s}
		for (int k = 1; k <= lenMax-2; k++) {					// we can add up to lenMax-2 points (so a cycle will have size less than lenMax)
			newC.clear();
			for (int c = 0; c < curC.size(); c++) {				// loop over all current partial cycles containing k vertices (so length k-1)
				d = curC[c].back();								// find the final donor d in the current partial cycle
				for (int l = 0; l < A[d].size(); l++) {			// for all recipients compatible with the donor d			
					r = A[d][l][0];								// find the current recipient r
					if (r == s && k > len) {					// if we are back at the start then this is a cycle, add it if its length is more than len
						cycles.push_back(curC[c]);				// add the cycle that we thus created (don't re-add the initial vertex)
					}
					else if (r > s && floyd[r][s] <= inst.nR - k) {	// else, only consider recipients r with index more than s such that the distance from r back to s is small enough
						add = true;									// consider adding recipient r to the partial cycle
						for (int p = 1; p < k; p++) {				// loop over all k predecessors of recipient r
							if (curC[c][p] == r) {					// if recipient r is already present in the current partial cycle
								add = false;						// don't add recipient r after all
								break;
							}
						}
						if (add) {								// if we decided to add recipient r
							newC.push_back({ curC[c] });		// add the new partial cycle containing k+1 vertices (so length k)
							newC.back().push_back(r);			// including recipient r as the k+1'th vertex
						}
					}
				}
			}
			curC = newC;									// the current set of partial cycles now consists of the partial cycles containing j+1 vertex (so length j)
		}
		for (int c = 0; c < curC.size(); c++) {				// loop over all partial cycles of size lenMax-1
			d = curC[c].back();								// find the final donor d in the current partial cycle
			for (int l = 0; l < A[d].size(); l++) {			// for all recipients compatible with the donor d			
				r = A[d][l][0];								// find the current recipient r
				if (r == s) {								// if we are back at the start then this is a cycle
					cycles.push_back(curC[c]);				// add the cycle that we thus created (don't re-add the initial vertex)
				}
			}
		}
		curC.clear();
	}
	return cycles;
}

vector<vector<int>> findPaths(const Instance& inst, const vector<vector<vector<int>>>& A, int length) {
	// This function finds the set of all simple paths of length ``length'' (i.e., containing "length" arcs and "length"+1 vertices, not repeating vertices)
	// If we set length = K, we obtain the set of minimally infeasible paths
	// If we set length = K-1, we obtain the set of maximally cardinality feasible paths 
	vector<vector<int>> curP;								// all paths of length k (initially k = 0)
	if (length > 0) {
		for (int d = 0; d < inst.nR; d++) {					// loop over all RDPs
			curP.push_back({ d });							// each RDP can be the start of a path
		}
	}
	int d, r;
	bool add;
	vector<vector<int>> newP;								// all paths of length k+1 (initially k = 0)
	for (int k = 1; k <= length; k++) {						// we add ``length'' points
		for (int p = 0; p < curP.size(); p++) {				// loop over all current paths p
			d = curP[p].back();								// find the final donor d in the current path p
			for (int i = 0; i < A[d].size(); i++) {			// for all recipients compatible with the donor d			
				r = A[d][i][0];								// find the current recipient r
				add = true;									// consider adding recipient r to the path p
				for (int l = 0; l < k; l++) {				// loop over all k predecessors of recipient r
					if (curP[p][l] == r) {					// if recipient r is already present in the current path p
						add = false;						// don't add recipient r after all
						break;
					}
				}
				if (add) {
					newP.push_back(curP[p]);		// add the new chain to the set of all chains of length k+1
					newP.back().push_back(r);
				}
			}
		}
		curP = newP;			// update the set of chains of length k to the set of length k+1
		newP.clear();
	}

	return curP;
}

//---------------------------FUNCTIONS RELATING TO SHORTEST PATHS---------------------------
vector<int> shortestDistancesFromPoint(const Instance& inst, const vector<vector<vector<int>>>& A, int l) {
	// For a given l, this function finds the shortest (l,i) distances for all i using BFS
	vector<int> dist(inst.nR + inst.nN, inst.nR+1);
	int u, v;
	vector<int> V = { l };
	dist[l] = 0;
	while (V.size() > 0) {
		u = V.front(); V.erase(V.begin() + 0);
		for (int i = 0; i < A[u].size(); i++) {
			v = A[u][i][0];
			if (dist[v] == inst.nR+1) {  // if still unprocessed
				dist[v] = dist[u] + 1;
				V.push_back(v);
			}
		}
	}
	return dist;
}

vector<int> shortestDistancesToPoint(const Instance& inst, const vector<vector<vector<int>>>& A, int l) {
	// For a given l, this function finds the shortest (i,l) distance for all i using BFS
	vector<int> dist(inst.nR, inst.nR);
	int u, v;


	// Reverse all arcs first
	vector<vector<int>> Arev(inst.nR);
	int r;
	for (int d = l; d < inst.nR; d++) {
		for (int i = 0; i < A[d].size(); i++) {
			r = A[d][i][0];
			Arev[r].push_back(d);
		}
	}

	// Apply BFS
	vector<int> V = { l };
	dist[l] = 0;
	while (V.size() > 0) {
		u = V.front(); V.erase(V.begin() + 0);
		for (int i = 0; i < Arev[u].size(); i++) {
			v = Arev[u][i];
			if (dist[v] == inst.nR) {  // if still unprocessed
				dist[v] = dist[u] + 1;
				V.push_back(v);
			}
		}
	}
	return dist;
}

vector<vector<int>> floydsAlgorithm(const Instance& inst, const vector<vector<vector<int>>>& A, bool zeroDiagonal, bool nondirected) {
	// This function creates the Floyd's matrix: floyd[d][r] is the shortest path distance from d to r (in terms of arcs), 
	//	and simply nR+1 if no (d,r)-path exists (note that no path can be longer than nR arcs)
	vector<vector<int>> floyd(inst.nR + (inst.nN) * nondirected, vector<int>(inst.nR, inst.nR + 1 * nondirected));
	int r;
	for (int d = 0; d < inst.nR; d++) {
		for (int i = 0; i < A[d].size(); i++) {
			r = A[d][i][0];
			floyd[d][r] = 1 * (d != r);
		}
	}
	for (int k = 0; k < inst.nR; k++) {
		for (int d = 0; d < inst.nR; d++) {
			for (int r = 0; r < inst.nR; r++)
				floyd[d][r] = min(floyd[d][r], floyd[d][k] + floyd[k][r]);
		}
	}
	if (zeroDiagonal) {
		for (int d = 0; d < inst.nR; d++) {
			floyd[d][d] = 0;
		}
	}
	if (nondirected) {
		for (int d = inst.nR; d < inst.nR + inst.nN; d++) {
			for (int r = 0; r < inst.nR; r++) {
				for (int i = 0; i < A[d].size(); i++) {
					if (A[d][i][0] == r) { floyd[d][r] = 1; break; }
					else { floyd[d][r] = min(floyd[d][r], 1 + floyd[A[d][i][0]][r]); }
				}
			}
		}
	}
	return floyd;
}

//---------------------------FUNCTIONS RELATING TO VERTEX ORDER---------------------------
vector<int> findVertexOrder(const Instance& inst, bool ascending) {
	// This function finds the order of the vertices when sorting them according to ascending (if "ascending"=true) or descending (if "ascending"=false) total degree
	// - vertexOrder[i] gives the original index of the vertex that is on index i after sorting
	// - This function only finds the order; the ordering is implemented in "changeOrder()"

	// First count the total degree of every vertex
	// We use a matrix with two columns, where column 1 simply contains the index of each vertex and column 2 contains the corresponding degree
	int r;
	vector<vector<int>> degree(inst.nR, vector<int>(2, 0));
	for (int d = 0; d < inst.nR; d++) {
		degree[d][0] = d;
		for (int i = 0; i < inst.A[d].size(); i++) {
			r = inst.A[d][i][0];
			degree[d][1]++; degree[r][1]++;
		}
	}

	// Sort according to the second column (containing the degrees)
	if (ascending) { sort(degree.begin(), degree.end(), sortOnSecondColumnAscending); }
	else { sort(degree.begin(), degree.end(), sortOnSecondColumnDescending); }

	// The vertex ordering is now simply the first column
	vector<int> vertexOrder(inst.nR);
	for (int d = 0; d < inst.nR; d++) {
		vertexOrder[d] = degree[d][0];
	}
	return vertexOrder;
}

void changeOrder(const Instance& inst, const vector<int>& vertexOrder, vector<vector<vector<int>>>& ANew, vector<int>& wToTauNew) {
	// This funciton changes the indices in A and wToTau according to the new vertex ordering
	vector<int> sortedIndex(inst.nR); for (int i = 0; i < inst.nR; i++) { sortedIndex[vertexOrder[i]] = i; }
	// - sortedIndex[i] gives the new index of the vertex that was originally at index i
	// printVector(sortedIndex, "sortedIndex");
	ANew.resize(inst.nR); wToTauNew.resize(inst.nR);
	int dOrig, rOrig, r;
	for (int d = 0; d < inst.nR; d++) {
		dOrig = vertexOrder[d];
		wToTauNew[d] = inst.wToTau[dOrig];
		for (int i = 0; i < inst.A[dOrig].size(); i++) {
			rOrig = inst.A[dOrig][i][0]; r = sortedIndex[rOrig];
			ANew[d].push_back({ r, inst.A[dOrig][i][1] });
		}
	}
}

//---------------------------FUNCTIONS RELATING TO PRINTING---------------------------
void printA(const vector<vector<vector<int>>>& A, const vector<int> wToTau, string name) {
	// This function prints a compatibility graph
	cout << name << ":\n";
	for (size_t d = 0; d < A.size(); d++) {
		cout << "   " << "donor" << " " << d << " (" << wToTau[d] << "): ";
		if (A[d].size() == 0) {
			cout << "-\n";
			continue;
		}
		cout << "[";
		for (size_t r = 0; r < A[d].size(); r++) {
			cout << A[d][r][0] << "(" << A[d][r][1] << ")";
			if (r != A[d].size() - 1) cout << ", ";
		}
		cout << "]\n";
	}
}

void printCyclesOrChains(const vector<vector<vector<int>>>& cyclesOrChains, string name, string rowname) {
	// This function prints a set of cycles or chains
	cout << name << ":\n";
	for (int c = 0; c < cyclesOrChains.size(); c++) {
		cout << "   " << rowname << " " << c << ": [";
		for (size_t i = 0; i < cyclesOrChains[c][0].size(); i++) {
			cout << cyclesOrChains[c][0][i];
			if (i != cyclesOrChains[c][0].size() - 1) { cout << " "; }
		}
		cout << "] (" << cyclesOrChains[c][1][0] << ")\n";
	}
}

//---------------------------OTHER---------------------------
int selectLPMethod(const Solution& sol) {
	// This function is used to select an LP solution method to use: 0 = primal simplex, 1 = dual simplex, 2 = barrier
	int methodLP;
	if (sol.modelChain == "EF-CHAIN-EXP" or sol.modelChain == "EF-CHAIN-EXP-CG" or sol.modelHybrid == "EF-HYBRID" or sol.modelHybrid == "EF-HYBRID-CG") { methodLP = 0; }
	else if (sol.modelChain == "HCF-CHAIN" or sol.modelChain == "EEF-CHAIN-EXP" or sol.modelChain == "EEF-CHAIN-EXP-CG") { methodLP = 2; }
	else if (sol.modelCycle == "EF-CYCLE" or sol.modelCycle == "EF-CYCLE-CG") { methodLP = 0; }
	else if (sol.modelCycle == "EEF-CYCLE") { methodLP = 2; }
	else if (sol.modelChain == "EEF-CHAIN-MTZ") { methodLP = 0; }
	else { methodLP = 2; }
	return methodLP;
}