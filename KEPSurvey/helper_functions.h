#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

//---------------------------IMPORT PACKAGES---------------------------
using namespace std;
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream> 
#include "gurobi_c++.h"

//---------------------------SET GENERAL CONSTANTS---------------------------
const double EPSILON = 0.001;   // small constant
const int M = 1000000;			// big constant

//---------------------------SIMPLE GENERAL PURPOSE FUNCTIONS---------------------------
double getCPUTime();
void removeLine();
void printVector(const vector<int>& vec, string name);
void printVector(const vector<bool>& vec, string name);
void printVector(const vector<double>& vec, string name);
void print2DVector(const vector<vector<int>>& vec, string name, string rowname);
void print2DVector(const vector<vector<bool>>& vec, string name, string rowname);
void print2DVector(const vector<vector<double>>& vec, string name, string rowname);
void print3DVector(const vector<vector<vector<int>>>& vec, string name, string rowname);
void print3DVector(const vector<vector<vector<bool>>>& vec, string name, string rowname);
void print3DVector(const vector<vector<vector<bool>>>& vec, string name, string rowname);
bool sortOnSecondColumnAscending(const vector<int>& v1, const vector<int>& v2);
bool sortOnSecondColumnDescending(const vector<int>& v1, const vector<int>& v2);
vector<int> findComplement(const vector<int>& vec, int len);

#endif
