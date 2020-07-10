//
// This code tests the correct working of the routines in
// the trid.h header file which solve tridiagonal systems
// within a warp.
//

//
// standard header files
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//
// my tridiagonal header file
//

#include <trid.h>

//
// compute and print rms error
//

template <typename REAL>
__forceinline__ __device__ REAL rms_err(REAL err){
  err = err*err;
  for (uint i=16; i>=1; i=i/2) err += __shfl_xor(err,i);
  return sqrt(err/32.0);
}

template <typename REAL>
__forceinline__ __device__ REAL max_err(REAL err){
  err = fabs(err);
  for (uint i=16; i>=1; i=i/2) err = max(err,__shfl_xor(err,i));
  return err;
}

//
// test kernels
//

template <typename REAL>
__global__ void trid1_warp_test(REAL con) {

  REAL a, c, d, u;
  int  t = threadIdx.x%32;

  a = -0.5f;
  c = -0.5f;
  u =  2.0*t + 1.0f;

  if (t==0)  a = 0.0f;  // essential to zero these out;
  if (t==31) c = 0.0f;  // trid_warp needs them to be zero

  d = a*__shfl_up(u,1) + u + c*__shfl_down(u,1);

  d = trid1_warp(a,c,d);

  REAL err_rms = rms_err(d-u);
  REAL err_max = max_err(d-u);
  if(threadIdx.x==0)
    printf("rms err = %g, max err = %g \n",err_rms,err_max);
}


template <typename REAL>
__global__ void trid1_warp_new_test(REAL con) {

  REAL a, b, c, d, u;
  int  t = threadIdx.x%32;

  a = -0.5f;
  b =  1.0f;
  c = -0.5f;
  u =  2.0*t + 1.0f;

  if (t==0)  a = 0.0f;  // essential to zero these out;
  if (t==31) c = 0.0f;  // trid_warp needs them to be zero

  d = a*__shfl_up(u,1) + b*u + c*__shfl_down(u,1);

  d = trid1_warp_new(a,b,c,d);

  REAL err_rms = rms_err(d-u);
  REAL err_max = max_err(d-u);
  if(threadIdx.x==0)
    printf("rms err = %g, max err = %g \n",err_rms,err_max);
}


template <typename REAL>
__global__ void trid2_warp_test(REAL con) {

  REAL am, cm, dm, um, ap, cp, dp, up;
  int  t = threadIdx.x%32;

  am = -0.5f;
  cm = -0.5f;
  um =  2.0f*t + 1.0f;

  ap = -0.25f;
  cp = -0.25f;
  up =  2.0f*t + 2.0f;

  if (t==0)  am = 0.0f;  // essential to zero these out;
  if (t==31) cp = 0.0f;  // trid2_warp needs them to be zero

  dm = am*__shfl_up(up,1) + um + cm*up;
  dp = ap*um + up +  cp*__shfl_down(um,1);

  trid2_warp(am,cm,dm,ap,cp,dp);

  dm = dm - um;
  dp = dp - up;
  REAL err_rms = rms_err(sqrt(0.5f*(dm*dm+dp*dp)));
  REAL err_max = max_err(max(dm,dp));
  if(threadIdx.x==0)
    printf("rms err = %g, max err = %g \n",err_rms,err_max);
}



template <typename REAL>
__global__ void trid2_warp_s_test(REAL con) {

  __shared__ volatile REAL shared[32];

  REAL am, cm, dm, um, ap, cp, dp, up;
  int  t = threadIdx.x%32;

  am = -0.5f;
  cm = -0.5f;
  um =  2.0f*t + 1.0f;

  ap = -0.25f;
  cp = -0.25f;
  up =  2.0f*t + 2.0f;

  if (t==0)  am = 0.0f;  // essential to zero these out;
  if (t==31) cp = 0.0f;  // trid2_warp needs them to be zero

  dm = am*__shfl_up(up,1) + um + cm*up;
  dp = ap*um + up +  cp*__shfl_down(um,1);

  trid2_warp_s(am,cm,dm,ap,cp,dp,shared);

  dm = dm - um;
  dp = dp - up;
  REAL err_rms = rms_err(sqrt(0.5f*(dm*dm+dp*dp)));
  REAL err_max = max_err(max(dm,dp));
  if(threadIdx.x==0)
    printf("rms err = %g, max err = %g \n",err_rms,err_max);
}


//
// main code
//

int main(int argc, char **argv) {

  printf("\ntrid1_warp_test \n---------------\n");
  trid1_warp_test<<<1,32>>>(1.0f);  // single precision test
  cudaThreadSynchronize();
  trid1_warp_test<<<1,32>>>(1.0);   // double precision test
  cudaThreadSynchronize();

  printf("\ntrid1_warp_new_test \n---------------\n");
  trid1_warp_new_test<<<1,32>>>(1.0f);  // single precision test
  cudaThreadSynchronize();
  trid1_warp_new_test<<<1,32>>>(1.0);   // double precision test
  cudaThreadSynchronize();

  printf("\ntrid2_warp_test \n---------------\n");
  trid2_warp_test<<<1,32>>>(1.0f);  // single precision test
  cudaThreadSynchronize();
  trid2_warp_test<<<1,32>>>(1.0);   // double precision test
  cudaThreadSynchronize();

  printf("\ntrid2_warp_s_test \n-----------------\n");
  trid2_warp_s_test<<<1,32>>>(1.0f);  // single precision test
  cudaThreadSynchronize();
  trid2_warp_s_test<<<1,32>>>(1.0);   // double precision test
  cudaThreadSynchronize();

// CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
  return 0;
}
