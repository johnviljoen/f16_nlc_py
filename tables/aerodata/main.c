// C file to compile the program as not a shared library for debugging purposes
//#include"mexndinterp.c"
#include"hifi_F16_AeroData.c"


void main() {
    double alpha = 1.0;
    double beta = 0.1;
    double out;

    double *temp; 
    temp = (double *)malloc(6*sizeof(double));  /*size of 6.1 array*/

    hifi_C_lef(alpha, beta, temp);

    printf("%f \n", temp[0]);
    printf("%f \n", temp[1]);
    printf("%f \n", temp[2]);
    printf("%f \n", temp[3]);
    printf("%f \n", temp[4]);
    printf("%f \n", temp[5]);

    printf("success \n");
}