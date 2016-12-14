#include <math.h>
#include <stdlib.h>

double *disp_in_box(double * v1, double * v2, 
                    double * box){
  double *out = malloc(24);
  double b0h = box[0]/2, b1h = box[1]/2, b2h = box[2]/2;
  out[0] = v2[0]-v1[0];
  out[1] = v2[1]-v1[1];
  out[2] = v2[2]-v1[2];
  if (out[0] > b0h)
    out[0] -= box[0];
  else if (out[0] < -b0h)
    out[0] += box[0];
  if (out[1] > b1h)
    out[1] -= box[1];
  else if (out[1] < -b1h)
    out[1] += box[1];
  if (out[2] > b2h)
    out[2] -= box[2];
  else if (out[2] < -b2h)
    out[2] += box[2];
  return out;
}

double distance(double * v1, double * v2){
  double d0, d1, d2;
  d0 = v2[0]-v1[0];
  d1 = v2[1]-v1[1];
  d2 = v2[2]-v1[2];
  return sqrt(d0*d0+d1*d1+d2*d2);
}


void normalize2_3d(double *vec){
  double norm = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
  vec[0] /= norm;
  vec[1] /= norm;
  vec[2] /= norm;
}

double norm2_3d(double *vec){
  return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}

double inv_norm2_3d(double *vec){
  return 1.0/sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}

double dot3d(double * vec1, double * vec2){
  return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2];
}

double vec_cos3d(double * vec1, double * vec2){
  double sumsq1 = vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2];
  double sumsq2 = vec2[0]*vec2[0]+vec2[1]*vec2[1]+vec2[2]*vec2[2];
  return 
    (vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2])/sqrt(sumsq1*sumsq2);
}
