#include "helper_functions.h"

//---------------------------SIMPLE GENERAL PURPOSE FUNCTIONS---------------------------
double getCPUTime() {
    // This function is used for timing code
    return (double)clock() / CLOCKS_PER_SEC;
}

void removeLine() {
    // This function is used to remove a message printed by Gurobi
    #ifdef _WIN32
        cout << "\x1b[1F\x1b[2K"; 
        cout << "\x1b[1F\x1b[2K";
    #endif
}

void printVector(const vector<int>& vec, string name) {
    // This function prints a 1-dimensional vector containing integers
    cout << name << ": ";

    if (vec.size() == 0) {
        cout << "[]\n";
    }
    else {
        cout << "[";
        for (size_t i = 0; i < vec.size(); i++) {
            cout << vec[i];
            if (i != vec.size() - 1) {
                cout << " ";
            }
        }
        cout << "]\n";
    }
}

void printVector(const vector<bool>& vec, string name) {
    // This function prints a 1-dimensional vector containing boolean values
    cout << name << ": ";

    if (vec.size() == 0) {
        cout << "[]\n";
    }
    else {
        cout << "[";
        for (size_t i = 0; i < vec.size(); i++)
        {
            cout << vec[i];
            if (i != vec.size() - 1) cout << " ";
        }
        cout << "]\n";
    }
}

void printVector(const vector<double>& vec, string name) {
    // This function prints a 1-dimensional vector containing doubles
    cout << name << ": ";

    if (vec.size() == 0) {
        cout << "[]\n";
    }
    else {
        cout << "[";
        for (size_t i = 0; i < vec.size(); i++)
        {
            cout << vec[i];
            if (i != vec.size() - 1) cout << " ";
        }
        cout << "]\n";
    }
}

void print2DVector(const vector<vector<int>>& vec, string name, string rowname) {
    // This function prints a 2-dimensional vector containing integers
    cout << name << ": \n";
    for (size_t i = 0; i < vec.size(); i++)
    {
        printVector(vec[i], "   " + rowname + " " + to_string(i));
    }
}

void print2DVector(const vector<vector<bool>>& vec, string name, string rowname) {
    // This function prints a 2-dimensional vector containing booleans
    cout << name << ": \n";
    for (size_t i = 0; i < vec.size(); i++)
    {
        printVector(vec[i], "   " + rowname + " " + to_string(i));
    }
}

void print2DVector(const vector<vector<double>>& vec, string name, string rowname) {
    // This function prints a 2-dimensional vector containing doubles
    cout << name << ": \n";
    for (size_t i = 0; i < vec.size(); i++)
    {
        printVector(vec[i], "   " + rowname + " " + to_string(i));
    }
}

void print3DVector(const vector<vector<vector<int>>>& vec, string name, string rowname) {
    // This function prints a 3-dimensional vector containing integers
    cout << name << ": \n";
    for (size_t i = 0; i < vec.size(); i++) {
        cout << "   " << rowname << " " << i << ": ";
        if (vec[i].size() == 0) {
            cout << "-\n";
            continue;
        }
        cout << "[";
        for (size_t j = 0; j < vec[i].size(); j++) {
            cout << "(";
            for (size_t k = 0; k < vec[i][j].size(); k++) {
                cout << vec[i][j][k];
                if (k != vec[i][j].size() - 1) cout << ",";
            }
            cout << ")";
            if (j != vec[i].size() - 1) cout << ", ";
        }
        cout << "]\n";
    }
}

void print3DVector(const vector<vector<vector<bool>>>& vec, string name, string rowname) {
    // This function prints a 3-dimensional vector containing booleans
    cout << name << ": \n";
    for (size_t i = 0; i < vec.size(); i++) {
        cout << "   " << rowname << " " << i << ": ";
        if (vec[i].size() == 0) {
            cout << "-\n";
            continue;
        }
        cout << "[";
        for (size_t j = 0; j < vec[i].size(); j++) {
            cout << "(";
            for (size_t k = 0; k < vec[i][j].size(); k++) {
                cout << vec[i][j][k];
                if (k != vec[i][j].size() - 1) cout << ",";
            }
            cout << ")";
            if (j != vec[i].size() - 1) cout << ", ";
        }
        cout << "]\n";
    }
}

void print3DVector(const vector<vector<vector<double>>>& vec, string name, string rowname) {
    // This function prints a 3-dimensional vector containing doubles
    cout << name << ": \n";
    for (size_t i = 0; i < vec.size(); i++) {
        cout << "   " << rowname << " " << i << ": ";
        if (vec[i].size() == 0) {
            cout << "-\n";
            continue;
        }
        cout << "[";
        for (size_t j = 0; j < vec[i].size(); j++) {
            cout << "(";
            for (size_t k = 0; k < vec[i][j].size(); k++) {
                cout << vec[i][j][k];
                if (k != vec[i][j].size() - 1) cout << ",";
            }
            cout << ")";
            if (j != vec[i].size() - 1) cout << ", ";
        }
        cout << "]\n";
    }
}

bool sortOnSecondColumnAscending(const vector<int>& v1, const vector<int>& v2) {
    // This function sorts the rows of a matrix according to the values in the second column (ascendingly)
    return v1[1] < v2[1];
}

bool sortOnSecondColumnDescending(const vector<int>& v1, const vector<int>& v2) {
    // This function sorts the rows of a matrix according to the values in the second column (descendingly)
    return v1[1] > v2[1];
}

vector<int> findComplement(const vector<int>& vec, int len) {
    // This function finds the set {0, 1, ..., len-1} \ vec, where vec is a subset of {0, 1, ..., len-1}
    vector<bool >isPresent(len, false);
    for (int i = 0; i < vec.size(); i++) {
        isPresent[vec[i]] = true;
    }

    vector<int> comp;
    for (int i = 0; i < len; i++) {
        if (not isPresent[i]) {
            comp.push_back(i);
        }
    }
    return comp;
}
