#ifndef MAIN_FUNCTIONS_H
#define MAIN_FUNCTIONS_H

#include "helper_functions.h"
#include "json.hpp"

struct Instance {
	// The Instance struct stores all information regarding the instance that is to be solved

	// Instance attributes
	string name;					// name of the instance
	int K, L;						// the maximum cycle and chain length, respectively
	int nR = 0;						// number of recipient-donor pairs (RDPs) 
	int nN = 0;						// number of non-directed donors (NDDs) 
	int nAR = 0;					// number of arcs between vertices in R only
	int nA = 0;						// number of arcs (excluding those to the terminal vertex tau)
	vector<int> V;                  // array giving the original index of each donor
	vector<bool> isN;				// boolean array indicating for each donor is they are non-directed (true) or not (false)			  
	vector<vector<vector<int>>> A;  // adjacency matrix, containing for each donor a list of tuples [[r,w] for each compatible recipient r and corresponding weight w]
	vector<int> wToTau;				// the weight from each vertex to the terminal vertex tau
	// vector<vector<int>> floyd;		// shortest path distance between every pair of vertices

	// Instance methods
	void print(bool full = false);  // prints the Instance object
	void nonWeightedVariant(bool useWeightsToTau);	// sets all weights to 1
	void generateRandomWToTau();	// generates random weights from each vertex to the terminal vertex tau
};

struct Solution {
	// The Solution struct stores all information and statistics regarding the method used to solve the instance and the results of that method

	// Solution attributes
	string modelCycle, modelChain, modelHybrid;				// the names of the models used to model cycles and chains (or hybrid model), respectively
	string modelFull;										// the full name of the model
	bool feas = false, opt = false;							// whether a feasible and optimal solution have been found, respectively
	int LB = 0, UB = 1000000;								// the best lower and upper bound found by the model, respectively 
	double LPrel = 1000000;									// the LP relaxation value found by the model
	int nVar = 0, nConstr = 0, nCoeff = 0;					// the number of variables, constraints and non-zero coefficients, respectively (before pre-processing)
	int nCutsCycle = 0, nCutsChain = 0;						// the number of cuts added within a contraint generation framework due to the cycle and chain model, respectively
	double timeP = 0, timeLP = 0, timeT = 0, timeW = 0; 	// the processing time, solving time for LP relaxation, total computation time (excluding initial LP relaxation time) and Gurobi time in work units, respectively
	vector<vector<vector<vector<int>>>> cyclesAndChains;	// the set of cycles and chains that are selected in the best solution found within the time limit
	
	// Attributes that have to do with RCVF
	bool RCVFUsed = false;									// boolean indicating whether reduced cost variable fixing is used
	int RCVFNIter = 0, RCVFNSkips = 0, RCVFNVarDeact = 0;	// the number of iterations required by RCVF, the number of times that an iteration was skipped and the number of variables that are deactivated in the final iteration
	
	// Attributes that have to do with hierarchical optimization
	bool solvedHierarchically = false;
	bool optUnweighted = false;
	int LBUnweighted = 0, UBUnweighted = 1000000;
	double timeUnweighted = 0, timeWeighted = 0;

	// Solution methods
	void print(int detail = 0);								// method used to print the Solution object
	bool check(const Instance& inst);						// method used to check if the solution is indeed feasible
};

struct Subgraph {
	// The Subgraph struct stores the set of arcs in multiple formats for some subgraph, for example:
	// - in EF, we construct one subgraph, which is the reduced version of the original graph
	// - in EEF, we construct one subgraph per layer, corresponding to the sets A^l
	// - in PIEF, we construct one subgraph per layer and arc position index

	// Subgraph attributes
	// - note: not all three attributes are constructed and used by all methods
	vector<vector<vector<int>>> A;	// A[d][i] gives the i'th arc leaving donor d, in the format [r,w], with r the recipient and w the arc's weight
	vector<vector<int>> arc2Idx;	// arc2Idx[d][r] gives the integer index of the arc (d,r), with value -1 if the arc does not exist
	vector<vector<int>> idx2Arc;	// idx2Arc[i] gives the i'th arc, in the format [d, r, w], with d the donor, r the recipient and w the arc's weight
};

class Callback : public GRBCallback {
	// The Callback object contains all code required for constraint generation and stores all information required for this

	public:
		// General
		Instance inst;
		string modelCycle;
		string modelChain;
		string modelHybrid;
		int nCutsCycle, nCutsChain;

		// Cycle model (EF variants)
		Subgraph subgraphCycle;
		vector<GRBVar> x;

		// Chain model (EF-CHAIN variants)
		Subgraph subgraphChain;
		vector<GRBVar> y;

		// Chain model (EEF-CHAIN variants)
		vector<Subgraph> subgraphsChain;
		vector<vector<GRBVar>> ySub;
	
		// Callback methods
		Callback(const Instance& inst, string modelCycle, string modelChain, string modelHybrid);
		void setCycleCallback(const Subgraph& subgraphCycle, const vector<GRBVar>& x);
		void setChainCallback(const Subgraph& subgraphChain, const vector<GRBVar>& y);
		void setChainCallback(const vector<Subgraph>& subgraphsChain, const vector<vector<GRBVar>>& ySub);
		int getNCuts(bool ofCycleModel);

	protected:
		void callback();
};

//---------------------------MAIN FUNCTION---------------------------
Solution solveKEP(const Instance& inst, string modelCycle, string modelChain, string modelHybrid, double timeLimit=3600, bool printGurobiLog=true, bool solveLP=false, bool useRCVF=false, bool printRCVFLog=true, bool solveHierarchically=false);

//---------------------------FUNCTIONS RELATING TO THE INSTANCE STRUCT---------------------------
Instance readInstance(string path, string filename);

//---------------------------FUNCTIONS RELATING TO SUBGRAPH CREATION---------------------------
Subgraph createSubgraphEFCYCLE(const Instance& inst);
Subgraph createSubgraphEEFCYCLE(const Instance& inst, const vector<vector<vector<int>>>& A, int l);
vector<Subgraph> createSubgraphPIEFCYCLE(const Instance& inst, const vector<vector<vector<int>>>& A, int l);
Subgraph createSubgraphEFCHAIN(const Instance& inst);
Subgraph createSubgraphEEFCHAIN(const Instance& inst, int l);
vector<Subgraph> createSubgraphPIEFCHAIN(const Instance& inst);
Subgraph createSubgraphEFHYBRID(const Instance& inst);

//---------------------------FUNCTIONS RELATING TO CONSTRUCTING LISTS OF CYCLES AND CHAINS---------------------------
vector<vector<vector<int>>> findCycles(const Instance& inst);
vector<vector<vector<int>>> findNonDominatedCycles(const Instance& inst);
vector<vector<vector<int>>> findHalfCycles(const Instance& inst, const vector<vector<vector<int>>>& A);
vector<vector<vector<int>>> findChains(const Instance& inst);
vector<vector<vector<int>>> findHalfChains(const Instance& inst, bool useLateSplit = false, bool useAlternative = false);
vector<vector<int>> findMinInfeasibleChains(const Instance& inst, const vector<vector<vector<int>>>& A, bool limitedStart=false);
vector<vector<int>> findShortCycles(const Instance& inst, const vector<vector<vector<int>>>& A, int len);
vector<vector<int>> findLongCycles(const Instance& inst, const vector<vector<vector<int>>>& A, int len, int lenMax=0);
vector<vector<int>> findPaths(const Instance& inst, const vector<vector<vector<int>>>& A, int length);

//---------------------------FUNCTIONS RELATING TO SHORTEST PATHS---------------------------
vector<int> shortestDistancesFromPoint(const Instance& inst, const vector<vector<vector<int>>>& A, int l);
vector<int> shortestDistancesToPoint(const Instance& inst, const vector<vector<vector<int>>>& A, int l);
vector<vector<int>> floydsAlgorithm(const Instance& inst, const vector<vector<vector<int>>>& A, bool zeroDiagonal = false, bool nondirected = true);

//---------------------------FUNCTIONS RELATING TO VERTEX ORDER---------------------------
vector<int> findVertexOrder(const Instance& inst, bool ascending);
void changeOrder(const Instance& inst, const vector<int>& vertexOrder, vector<vector<vector<int>>>& ANew, vector<int>& wToTauNew);

//---------------------------FUNCTIONS RELATING TO PRINTING---------------------------
void printA(const vector<vector<vector<int>>>& A, const vector<int> wToTau, string name);
void printCyclesOrChains(const vector<vector<vector<int>>>& cyclesOrChains, string name = "Cycles", string rowname = "cycle");

//---------------------------OTHER---------------------------
int selectLPMethod(const Solution& sol);

#endif