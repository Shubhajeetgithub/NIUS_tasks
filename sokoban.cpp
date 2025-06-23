#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stack>
#include <algorithm>
#include <set>
using namespace std;
#define L 10
vector<string> sokobanComb;
set<string> seen;

//generate all sokoban combinations of length L
void generateCombinations(int id, string temp, vector<string>& sokobanComb) {
    if (id == L) {
        sokobanComb.push_back(temp);
        return;
    }
    temp += 's';
    generateCombinations(id+1, temp, sokobanComb);
    temp.pop_back();
    temp += 'b';
    generateCombinations(id+1, temp, sokobanComb);
    temp.pop_back();
    temp += 'e';
    generateCombinations(id+1, temp, sokobanComb);
}

// convert a string s to its hashed version, i.e.,
/**********
 * first we split the string into tokens Fl | A | F | A ... | Fr where Fl and Fr may be empty
 * then we remove all e's from A
 * we return this as our hashed string
 *********/
void splitToTokens(int index, const string& s, vector<string>& tokens, string prefix) {
    if (index >= L) {
        tokens.push_back(prefix);
        return;
    }
    
    // Find the first occurrence of 's' from index
    int ix = -1;
    for (int i = index; i < L; i++) {
        if (s[i] == 's') {
            ix = i;
            break;
        }
    }
    

    if (ix == -1) {
        // No 's' found, push the rest of the string
        tokens.push_back(prefix + s.substr(index, L - index));
        return;
    }
    
    if (ix == index) {
        // Fl will be null in this case
        tokens.push_back(prefix + "");
    } else {
        int j = ix - 1;
        bool found = false;
        while (j >= index + 1) {
            if (s[j] == 'b' && s[j-1] == 'b') {
                tokens.push_back(prefix + s.substr(index, j - index + 1));
                ix = j + 1;
                found = true;
                break;
            }
            j--;
        }
        if (!found) {
            if (s[index] == 'b') tokens.push_back(prefix + "b");
            else tokens.push_back(prefix + "");
        }
    }
    
    // Now let us see where A extends
    // Find the first occurrence of "bb" after ix
    int iy = L;
    for (int i = ix + 1; i < L - 1; i++) {
        if (s[i] == 'b' && s[i + 1] == 'b') {
            iy = i;
            break;
        }
    }
    
    if (ix < L) {
        tokens.push_back(s.substr(ix, iy - ix));
    }
    
    if (iy != L) {
        splitToTokens(iy + 2, s, tokens, "bb");
    } else {
        return;
    }
}

string getHashed(const string& s) {
    vector<string> tokens;
    splitToTokens(0, s, tokens, "");
    string hashed;
    for (int i = 0; i < tokens.size(); i++) {
        if (i % 2 == 0) {
            hashed += tokens[i];
        } else {
            for (int j = 0; j < tokens[i].size(); j++) {
                if (tokens[i][j] != 'e') {
                    hashed += tokens[i][j];
                }
            }
        }
    }
    return hashed;
}

/*
To determine if a 1D sokoban game can be won or not, first identify the position of targets.
By suitably moving e's in the active region we can check for solution.
*/

int main() {
    generateCombinations(0, "", sokobanComb);
    for (auto& c : sokobanComb) {
        // cout << c << "->";
        string hash = getHashed(c);
        if (seen.find(hash) == seen.end()) {
            seen.insert(hash);
            // cout << hash << "-> 1";
        }
        // else cout << hash << "-> 0";
        // cout << endl;
    }
    cout << seen.size() << endl;
    return 0;
}