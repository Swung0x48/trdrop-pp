// JNI entry of the Android app: runs the command line program of trdrop++ in the process of the app.

#include <jni.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/jni.h>
#include <libavutil/log.h>
}

#include "trdrop_library.h"

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *)
{
    // FFmpeg reaches the MediaCodec codecs of the device through the Java VM
    av_jni_set_java_vm(vm, nullptr);
    // the MediaCodec wrappers describe every format change at the info level, the log is shown to the user
    av_log_set_level(AV_LOG_ERROR);
    return JNI_VERSION_1_6;
}

//! NativeAnalyzer.run(arguments, logPath, listener): the exit code of the program.
//! Its messages (stderr) are written to logPath. Only one call at a time: stderr belongs to the process.
extern "C" JNIEXPORT jint JNICALL
Java_io_github_swung0x48_trdrop_NativeAnalyzer_run(JNIEnv * env, jobject, jobjectArray arguments, jstring log_path, jobject listener)
{
    std::vector<std::string> values { "trdrop-pp" };
    unsetenv("TRDROP_GPU_PROFILE");
    const jsize count = env->GetArrayLength(arguments);
    for (jsize i = 0; i < count; ++i)
    {
        const jstring argument = static_cast<jstring>(env->GetObjectArrayElement(arguments, i));
        const char * text = env->GetStringUTFChars(argument, nullptr);
        // not an argument of the program: times the steps of the GPU pipeline (see vk_mediacodec_pipeline.cpp)
        if (std::string(text) == "--gpu-profile") setenv("TRDROP_GPU_PROFILE", "1", 1);
        else values.emplace_back(text);
        env->ReleaseStringUTFChars(argument, text);
        env->DeleteLocalRef(argument);
    }
    std::vector<char *> argv;
    for (std::string & value : values) argv.push_back(value.data());

    const char * log = env->GetStringUTFChars(log_path, nullptr);
    std::fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    const int file = open(log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    env->ReleaseStringUTFChars(log_path, log);
    if (file >= 0)
    {
        dup2(file, STDERR_FILENO);
        close(file);
    }

    jmethodID on_progress = nullptr;
    if (listener)
    {
        const jclass type = env->GetObjectClass(listener);
        on_progress = env->GetMethodID(type, "onProgress", "(JD)Z");
        env->DeleteLocalRef(type);
    }
    using Clock = std::chrono::steady_clock;
    Clock::time_point last = Clock::now() - std::chrono::seconds(1);
    const auto progress = [&](long long frames, double fraction) {
        if (!on_progress) return true;
        // the listener is asked a few times per second, that is often enough to stop the analysis
        const Clock::time_point now = Clock::now();
        if (now - last < std::chrono::milliseconds(150)) return true;
        last = now;
        std::fflush(stderr);
        return env->CallBooleanMethod(listener, on_progress, static_cast<jlong>(frames), static_cast<jdouble>(fraction)) == JNI_TRUE;
    };

    const int code = trdrop_run(static_cast<int>(argv.size()), argv.data(), progress);

    std::fflush(stderr);
    if (saved >= 0)
    {
        dup2(saved, STDERR_FILENO);
        close(saved);
    }
    return code;
}
