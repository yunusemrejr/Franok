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
static inline __attribute__((always_inline)) void matvec(const u8 *w,const float *x,float *y,int rows,int cols,i8 *activ,float *scales, int bits) {
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
        for(int j=0;j<cols;j+=32,w+=(bits==4?20:36)) {
            v128_t p=wasm_v128_load(w+4);
            v128_t lo=bits==4?wasm_i8x16_sub(wasm_v128_and(p,mask),zero):p;
            v128_t hi=bits==4?wasm_i8x16_sub(wasm_u8x16_shr(p,4),zero):wasm_v128_load(w+20);
            v128_t d=wasm_i32x4_add(dot16(lo,wasm_v128_load(activ+j)),dot16(hi,wasm_v128_load(activ+j+16)));
            v128_t s=wasm_f32x4_splat(scale_at(w)*scales[j/32]);
            acc=wasm_f32x4_add(acc,wasm_f32x4_mul(s,wasm_f32x4_convert_i32x4(d)));
        }
        y[i]=wasm_f32x4_extract_lane(acc,0)+wasm_f32x4_extract_lane(acc,1)+wasm_f32x4_extract_lane(acc,2)+wasm_f32x4_extract_lane(acc,3);
    }
}
void gemv(const u8 *w,const float *x,float *y,int rows,int cols,i8 *activ,float *scales) {
    matvec(w,x,y,rows,cols,activ,scales,4);
}
void gemv8(const u8 *w,const float *x,float *y,int rows,int cols,i8 *activ,float *scales) {
    matvec(w,x,y,rows,cols,activ,scales,8);
}
void unpack8(const u8 *w,float *out,int n) {
    for(int j=0;j<n;j+=32,w+=36){
        float s=scale_at(w);
        for(int k=0;k<32;k++)out[j+k]=s*(float)((const i8*)(w+4))[k];
    }
}
void rms(const float *x,const float *g,float *out,int n) {
    float sum=0;
    for(int j=0;j<n;j++)sum+=x[j]*x[j];
    float s=1/__builtin_sqrtf(sum/n+1e-5f);
    for(int j=0;j<n;j++)out[j]=x[j]*g[j]*s;
}
void add(float *x,const float *v,int n){for(int j=0;j<n;j++)x[j]+=v[j];}

/* The two nonlinear kernels need exp(x) only for x <= 0. Reduce x to a
 * half-log(2) interval and evaluate exp(r) through the sixth-order term.
 * Splitting log(2) keeps the residual accurate near the FP32 exponent floor. */
static inline v128_t exp_nonpositive(v128_t x) {
    x = wasm_f32x4_max(x, wasm_f32x4_splat(-87.0f));
    v128_t exponent = wasm_f32x4_nearest(wasm_f32x4_mul(x, wasm_f32x4_splat(1.44269504089f)));
    v128_t r = wasm_f32x4_sub(x, wasm_f32x4_mul(exponent, wasm_f32x4_splat(0.693359375f)));
    r = wasm_f32x4_sub(r, wasm_f32x4_mul(exponent, wasm_f32x4_splat(-0.00021219444f)));
    v128_t p = wasm_f32x4_splat(1.0f/720);
    p = wasm_f32x4_add(wasm_f32x4_mul(p,r), wasm_f32x4_splat(1.0f/120));
    p = wasm_f32x4_add(wasm_f32x4_mul(p,r), wasm_f32x4_splat(1.0f/24));
    p = wasm_f32x4_add(wasm_f32x4_mul(p,r), wasm_f32x4_splat(1.0f/6));
    p = wasm_f32x4_add(wasm_f32x4_mul(p,r), wasm_f32x4_splat(0.5f));
    p = wasm_f32x4_add(wasm_f32x4_mul(p,r), wasm_f32x4_splat(1.0f));
    p = wasm_f32x4_add(wasm_f32x4_mul(p,r), wasm_f32x4_splat(1.0f));
    v128_t bits=wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(exponent),wasm_i32x4_splat(127));
    return wasm_f32x4_mul(p,wasm_i32x4_shl(bits,23));
}

static inline float sum_lanes(v128_t v) {
    return wasm_f32x4_extract_lane(v,0) + wasm_f32x4_extract_lane(v,1)
         + wasm_f32x4_extract_lane(v,2) + wasm_f32x4_extract_lane(v,3);
}

void swiglu(float *h, int n) {
    int i=0;
    const v128_t one=wasm_f32x4_splat(1.f);
    for (;i+4<=n;i+=4) {
        v128_t x=wasm_v128_load(h+i);
        v128_t z=exp_nonpositive(wasm_f32x4_neg(wasm_f32x4_abs(x)));
        v128_t numerator=wasm_v128_bitselect(one,z,wasm_f32x4_ge(x,wasm_f32x4_splat(0)));
        v128_t sig=wasm_f32x4_div(numerator,wasm_f32x4_add(one,z));
        wasm_v128_store(h+i,wasm_f32x4_mul(wasm_f32x4_mul(x,sig),wasm_v128_load(h+n+i)));
    }
    for (;i<n;i++) {
        float x=h[i], z=wasm_f32x4_extract_lane(exp_nonpositive(wasm_f32x4_splat(-__builtin_fabsf(x))),0);
        h[i]=x*h[n+i]*(x<0?z:1.f)/(1.f+z);
    }
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
            const float *key=keys+(t*kv+kh)*hd;
            const float *query=q+h*hd;
            v128_t accum=wasm_f32x4_splat(0);
            int j=0;
            for (;j+4<=hd;j+=4)
                accum=wasm_f32x4_add(accum,wasm_f32x4_mul(wasm_v128_load(query+j),wasm_v128_load(key+j)));
            float v=sum_lanes(accum);
            for (;j<hd;j++)v+=query[j]*key[j];
            scores[t]=v*norm;max=maximum(max,scores[t]);
        }
        v128_t chunks=wasm_f32x4_splat(0), peak=wasm_f32x4_splat(max);
        int t=0;
        for (;t+4<=n;t+=4) {
            v128_t mass=exp_nonpositive(wasm_f32x4_sub(wasm_v128_load(scores+t),peak));
            wasm_v128_store(scores+t,mass);
            chunks=wasm_f32x4_add(chunks,mass);
        }
        float z=sum_lanes(chunks);
        for (;t<n;t++) {
            scores[t]=wasm_f32x4_extract_lane(exp_nonpositive(wasm_f32x4_splat(scores[t]-max)),0);
            z+=scores[t];
        }
        for(int j=0;j<hd;j++)out[h*hd+j]=0;
        for(int t=0;t<n;t++){
            float a=scores[t]/z;
            for(int j=0;j<hd;j++)out[h*hd+j]+=a*values[(t*kv+kh)*hd+j];
        }
    }
}

/* Small FP32 readout matrices have no quantization stage. */
void linear_f32(const float *weights, const float *x, const float *bias, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = weights + i * cols;
        v128_t sum = wasm_f32x4_splat(0);
        int j = 0;
        for (; j + 4 <= cols; j += 4)
            sum = wasm_f32x4_add(sum, wasm_f32x4_mul(wasm_v128_load(row + j), wasm_v128_load(x + j)));
        float value = wasm_f32x4_extract_lane(sum, 0) + wasm_f32x4_extract_lane(sum, 1)
                    + wasm_f32x4_extract_lane(sum, 2) + wasm_f32x4_extract_lane(sum, 3);
        for (; j < cols; j++) value += row[j] * x[j];
        out[i] = value + (bias ? bias[i] : 0);
    }
}
