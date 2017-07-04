// CPU Gold Standard Matrix/Matrix Multiplication Prototype
//  matrixmul_gold.cpp
void computeGold(float* P, // Resultant matrix data
   const float* M, // Matrix M
   const float* N, // Matrix N
   int Mh, // Matrix M height
   int Mw, // Matrix M width
   int Nw); // Matrix N width

// GPU (Device) Matrix/Matrix Multiplication Prototype
// gpu_matrixmul.cpp
void compute_GPU(float *M, float *N, float *P, int Mh,
   int Mw, int Nw);

// prototype for DeviceSelect Routine (util.cu)
int DeviceSelect(int device_id);

// prototype for DeviceInfo Routine (util.cu)
void DeviceInfo(int device_id);

// util2 function prototypes
void write_tm(char * file_name, int width, double run_time);

