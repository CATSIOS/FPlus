/* 精简 jni.h（Linux/Android 风格），仅够编译 JNI 导出函数。
 * 本代码不使用 JNIEnv 的任何方法，只需类型与导出宏。
 */
#ifndef JNI_H_
#define JNI_H_

#include <stdint.h>

typedef uint8_t  jboolean;
typedef int8_t   jbyte;
typedef uint16_t jchar;
typedef int16_t  jshort;
typedef int32_t  jint;
typedef int64_t  jlong;
typedef float    jfloat;
typedef double   jdouble;

typedef jint jsize;

struct _jobject;
typedef struct _jobject *jobject;
typedef jobject jclass;
typedef jobject jstring;
typedef jobject jthrowable;

struct JNIEnv_;
typedef struct JNIEnv_ JNIEnv;

typedef union jvalue {
    jboolean z;
    jbyte    b;
    jchar    c;
    jshort   s;
    jint     i;
    jlong    j;
    jfloat   f;
    jdouble  d;
    jobject  l;
} jvalue;

#define JNIEXPORT  __attribute__((visibility("default")))
#define JNIIMPORT
#define JNICALL

#endif /* JNI_H_ */
