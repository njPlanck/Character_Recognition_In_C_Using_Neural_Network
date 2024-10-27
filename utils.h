#include <stdlib.h>
#include <stdio.h>


typedef struct {
    float ** in; //2D array for inputs
    float ** tg; //2D array for targets
    int nips; //number of inputs
    int nops; //number of outputs
    int rows; //number of rows

}Data;


char * readln(FILE * const file);
float ** new2d(const int rows, const int cols);
Data ndata(const int nips, const int nops, const int rows);
void parse(const Data data,char * line, const int row);
void dfree(const Data d);
void shuffle(const Data d);
Data build(const char * path, const int nips, const int nops);
int lns(FILE *const file);
