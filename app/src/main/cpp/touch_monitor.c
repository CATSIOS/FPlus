/*
 * touch_monitor.c - 真实触摸屏监听 JNI
 *
 * 作用：只读打开真实触摸屏 /dev/input/eventX，判断用户手指是否正按在屏幕上，
 * 供 App 侧"检测让路"使用（用户触摸时暂停自动跟随）。
 *
 * 不依赖 libc：所有系统调用通过 aarch64 svc 指令直接实现，
 * 生成的 .so 零 libc 依赖，可在 Android bionic 环境加载。
 * 运行于 Shizuku UserService 进程（shell UID，有 /dev/input 读权限）。
 *
 * ---------------------------------------------------------------------------
 * 编译方式（本工程不使用 Gradle/CMake 构建 native，产物为预编译的 .so）：
 *
 *   产物路径：app/src/main/jniLibs/arm64-v8a/libtouchmonitor.so
 *
 *   zig cc -target aarch64-linux-musl -shared -fPIC -nostdlib -O2 -Wl,-s \
 *     -I<zig>/lib/libc/include/aarch64-linux-musl \
 *     -I<zig>/lib/libc/include/any-linux-any \
 *     -I<zig>/lib/libc/include/generic-musl \
 *     -I app/src/main/cpp \
 *     app/src/main/cpp/touch_monitor.c \
 *     -o app/src/main/jniLibs/arm64-v8a/libtouchmonitor.so
 *
 *   说明：
 *     - 用 musl target + -nostdlib，避免链接到 musl 的 libc.so
 *       （Android 是 bionic，musl 专有符号如 dl_iterate_phdr 会导致
 *        UnsatisfiedLinkError，故文件内自带 weak 兜底）
 *     - 编译后应确认零依赖：readelf -d 无 NEEDED，nm -D 无未定义符号
 * ---------------------------------------------------------------------------
 */
#include <jni.h>
#include <linux/input.h>

/* ===== 手写 weak 兜底：compiler_rt 可能弱引用这些 libc 符号 ===== */
__attribute__((weak)) long getauxval(unsigned long type) {
    (void) type;
    return 0;
}
__attribute__((weak)) int dl_iterate_phdr(void *callback, void *data) {
    (void) callback;
    (void) data;
    return 0;
}

static int g_touchFd = -1;                // 真实触摸屏 fd（只读）
static volatile int g_userTouching = 0;   // 用户手指是否正按着
static int g_touchX = 0, g_touchY = 0;          // 手指当前坐标（原始值）
static int g_touchXMax = 1, g_touchYMax = 1;    // 坐标范围最大值（用于归一化）

/* ===== aarch64 直接系统调用 ===== */
static inline long __syscall6(long n, long a, long b, long c, long d, long e, long f) {
    register long x8 __asm__("x8") = n;
    register long x0 __asm__("x0") = a;
    register long x1 __asm__("x1") = b;
    register long x2 __asm__("x2") = c;
    register long x3 __asm__("x3") = d;
    register long x4 __asm__("x4") = e;
    register long x5 __asm__("x5") = f;
    __asm__ volatile("svc #0"
                     : "+r"(x0)
                     : "r"(x8), "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5)
                     : "memory");
    return x0;
}

#define SYS_openat 56
#define SYS_read   63
#define SYS_ioctl  29
#define SYS_close  57

#define AT_FDCWD   (-100)
#define O_RDONLY   0
#define O_NONBLOCK 0x800

static int my_open(const char *path, int flags) {
    return (int) __syscall6(SYS_openat, AT_FDCWD, (long) path, flags, 0, 0, 0);
}
static long my_read(int fd, void *buf, unsigned long len) {
    return __syscall6(SYS_read, fd, (long) buf, len, 0, 0, 0);
}
static int my_ioctl(int fd, unsigned long cmd, void *arg) {
    return (int) __syscall6(SYS_ioctl, fd, cmd, (long) arg, 0, 0, 0);
}
static int my_close(int fd) {
    return (int) __syscall6(SYS_close, fd, 0, 0, 0, 0, 0);
}

static void my_memset(void *dst, int c, unsigned long n) {
    unsigned char *p = (unsigned char *) dst;
    while (n--) *p++ = (unsigned char) c;
}

/* ===== 触摸屏探测 ===== */

static int has_bit(const unsigned char *bits, int idx) {
    return (bits[idx / 8] >> (idx % 8)) & 1;
}

static void build_event_path(char *buf, int n) {
    const char *prefix = "/dev/input/event";
    int i = 0;
    while (prefix[i]) { buf[i] = prefix[i]; i++; }
    if (n >= 10) { buf[i++] = (char) ('0' + (n / 10)); }
    buf[i++] = (char) ('0' + (n % 10));
    buf[i] = '\0';
}

/* 跳过自建/遗留的 fplus 虚拟设备（按名字前缀） */
static int is_our_device(int fd) {
    char name[64];
    my_memset(name, 0, sizeof(name));
    if (my_ioctl(fd, EVIOCGNAME(sizeof(name)), name) < 0) return 0;
    const char *tag = "fplus";
    for (int i = 0; i < 5; i++) {
        if (tag[i] == '\0') return 1;
        if (name[i] != tag[i]) return 0;
    }
    return 1;
}

/* 扫描 /dev/input 找真实触摸屏（多指 + DIRECT + BTN_TOUCH），返回只读 fd */
static int find_real_touchscreen(void) {
    char path[32];
    for (int i = 0; i < 32; i++) {
        build_event_path(path, i);
        int fd = my_open(path, O_RDONLY | O_NONBLOCK);
        if (fd < 0) continue;

        if (is_our_device(fd)) { my_close(fd); continue; }

        unsigned char absBits[8];
        unsigned char propBits[4];
        unsigned char keyBits[96];
        my_memset(absBits, 0, sizeof(absBits));
        my_memset(propBits, 0, sizeof(propBits));
        my_memset(keyBits, 0, sizeof(keyBits));

        if (my_ioctl(fd, EVIOCGBIT(EV_ABS, sizeof(absBits)), absBits) < 0 ||
            my_ioctl(fd, EVIOCGPROP(sizeof(propBits)), propBits) < 0 ||
            my_ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(keyBits)), keyBits) < 0) {
            my_close(fd);
            continue;
        }

        if (has_bit(absBits, ABS_MT_POSITION_X)
                && has_bit(absBits, ABS_MT_POSITION_Y)
                && has_bit(propBits, INPUT_PROP_DIRECT)
                && has_bit(keyBits, BTN_TOUCH)) {
            // 读取坐标范围，用于归一化（供右下 1/4 区域判断）
            struct input_absinfo ai;
            if (my_ioctl(fd, EVIOCGABS(ABS_MT_POSITION_X), &ai) == 0 && ai.maximum > 0) {
                g_touchXMax = ai.maximum;
            }
            if (my_ioctl(fd, EVIOCGABS(ABS_MT_POSITION_Y), &ai) == 0 && ai.maximum > 0) {
                g_touchYMax = ai.maximum;
            }
            return fd;   // 找到真实触摸屏
        }
        my_close(fd);
    }
    return -1;
}

/* 排空事件缓冲，更新 g_userTouching（ABS_MT_TRACKING_ID: >=0 按下, -1 抬起） */
static void poll_user_touch(void) {
    if (g_touchFd < 0) return;
    struct input_event evs[64];
    for (int guard = 0; guard < 16; guard++) {
        long n = my_read(g_touchFd, evs, sizeof(evs));
        if (n <= 0) break;   // EAGAIN：没有新事件
        int cnt = (int) (n / (long) sizeof(struct input_event));
        for (int i = 0; i < cnt; i++) {
            unsigned short t = evs[i].type;
            unsigned short c = evs[i].code;
            int v = evs[i].value;
            if (t == EV_ABS && c == ABS_MT_POSITION_X) {
                g_touchX = v;
            } else if (t == EV_ABS && c == ABS_MT_POSITION_Y) {
                g_touchY = v;
            } else if (t == EV_ABS && c == ABS_MT_TRACKING_ID) {
                g_userTouching = (v >= 0) ? 1 : 0;
            } else if (t == EV_KEY && c == BTN_TOUCH && v == 0) {
                g_userTouching = 0;
            }
        }
    }
}

/* ===== JNI 接口 ===== */

JNIEXPORT void JNICALL
Java_com_example_fplus_TouchMonitorService_nativeStart(JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;
    if (g_touchFd >= 0) return;
    g_touchFd = find_real_touchscreen();
}

JNIEXPORT jint JNICALL
Java_com_example_fplus_TouchMonitorService_nativeIsUserTouching(JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;
    poll_user_touch();
    return g_userTouching;
}

JNIEXPORT jfloat JNICALL
Java_com_example_fplus_TouchMonitorService_nativeGetTouchX(JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;
    poll_user_touch();
    if (g_touchXMax <= 0) return 0.0f;
    return (jfloat) g_touchX / (jfloat) g_touchXMax;
}

JNIEXPORT jfloat JNICALL
Java_com_example_fplus_TouchMonitorService_nativeGetTouchY(JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;
    poll_user_touch();
    if (g_touchYMax <= 0) return 0.0f;
    return (jfloat) g_touchY / (jfloat) g_touchYMax;
}

JNIEXPORT void JNICALL
Java_com_example_fplus_TouchMonitorService_nativeDestroy(JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;
    if (g_touchFd >= 0) {
        my_close(g_touchFd);
        g_touchFd = -1;
    }
    g_userTouching = 0;
}
