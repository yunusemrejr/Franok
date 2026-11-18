#include <wasm_simd128.h>
typedef unsigned char u8;
typedef signed char i8;
typedef unsigned int u32;
static inline float maximum(float a,float b){return a>b?a:b;}

static inline float scale_at(const u8 *p) {
    float f;
    __builtin_memcpy(&f,p,4);
    return f;
}

/* Q4 rows are groups of 32. Each byte packs columns j and j+16. */
void unpack(const u8 *w,float *out,int n) {
    for(int i=0;i<n;i+=32,w+=20) {
        float scale=scale_at(w);
        for(int k=0;k<16;k++) {
            out[i+k]=scale*((w[4+k]&15)-8);
            out[i+k+16]=scale*((w[4+k]>>4)-8);
        }
    }
}

static inline v128_t dot16(v128_t a,v128_t b) {
    v128_t low=wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a),wasm_i16x8_extend_low_i8x16(b));
    v128_t high=wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a),wasm_i16x8_extend_high_i8x16(b));
    return wasm_i32x4_add(low,high);
}

/* Quantize activations per group; a GEMV reads only the selected bank. */
void gemv(const u8 *w,const float *x,float *y,int rows,int cols,i8 *activ,float *scales) {
    for(int j=0;j<cols;j+=32) {
        float peak=1e-12f;
        for(int k=0;k<32;k++) peak=maximum(peak,__builtin_fabsf(x[j+k]));
        float scale=peak/127.0f;
        scales[j/32]=scale;
        for(int k=0;k<32;k++) activ[j+k]=(i8)__builtin_nearbyintf(x[j+k]/scale);
    }
    const v128_t mask=wasm_i8x16_splat(15), zero=wasm_i8x16_splat(8);
    for(int i=0;i<rows;i++) {
        v128_t acc=wasm_f32x4_splat(0);
        for(int j=0;j<cols;j+=32,w+=20) {
            v128_t p=wasm_v128_load(w+4);
            v128_t lo=wasm_i8x16_sub(wasm_v128_and(p,mask),zero);
            v128_t hi=wasm_i8x16_sub(wasm_u8x16_shr(p,4),zero);
            v128_t d=wasm_i32x4_add(dot16(lo,wasm_v128_load(activ+j)),dot16(hi,wasm_v128_load(activ+j+16)));
            v128_t s=wasm_f32x4_splat(scale_at(w)*scales[j/32]);
            acc=wasm_f32x4_add(acc,wasm_f32x4_mul(s,wasm_f32x4_convert_i32x4(d)));
        }
        y[i]=wasm_f32x4_extract_lane(acc,0)+wasm_f32x4_extract_lane(acc,1)+wasm_f32x4_extract_lane(acc,2)+wasm_f32x4_extract_lane(acc,3);
    }
}
void rms(const float *x,const float *g,float *out,int n) {
    float sum=0;
    for(int j=0;j<n;j++)sum+=x[j]*x[j];
    float s=1/__builtin_sqrtf(sum/n+1e-5f);
    for(int j=0;j<n;j++)out[j]=x[j]*g[j]*s;
}
void add(float *x,const float *v,int n){for(int j=0;j<n;j++)x[j]+=v[j];}

__attribute__((import_module("math"),import_name("exp"))) extern float expf_js(float);
void swiglu(float *h,int n) {
    for(int i=0;i<n;i++) h[i]=h[i]*h[n+i]/(1+expf_js(-h[i]));
}

void rotary(float *x,const float *c,const float *s,int heads,int hd) {
    for(int h=0;h<heads;h++,x+=hd) for(int j=0;j<hd;j+=2){
        float even=x[j],odd=x[j+1];
        x[j]=even*c[j/2]-odd*s[j/2];x[j+1]=even*s[j/2]+odd*c[j/2];
    }
}

/* Cached keys/values are [time,kvhead,head_dim]. */
void attend(const float *q,const float *keys,const float *values,float *out,float *scores,int n,int heads,int kv,int hd) {
    float norm=1/__builtin_sqrtf((float)hd);
    for(int h=0;h<heads;h++) {
        int kh=h/(heads/kv);
        float max=-1e30f;
        for(int t=0;t<n;t++){
            float v=0;
            for(int j=0;j<hd;j++)v+=q[h*hd+j]*keys[(t*kv+kh)*hd+j];
            scores[t]=v*norm;max=maximum(max,scores[t]);
        }
        float z=0;
        for(int t=0;t<n;t++){scores[t]=expf_js(scores[t]-max);z+=scores[t];}
        for(int j=0;j<hd;j++)out[h*hd+j]=0;
        for(int t=0;t<n;t++){
            float a=scores[t]/z;
            for(int j=0;j<hd;j++)out[h*hd+j]+=a*values[(t*kv+kh)*hd+j];
        }
    }
}
