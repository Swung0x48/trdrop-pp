plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun nativeDependency(property: String): String =
    rootProject.file(providers.gradleProperty(property).get()).absolutePath.replace('\\', '/')

android {
    namespace = "io.github.swung0x48.trdrop"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "io.github.swung0x48.trdrop"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.1.1"

        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DTRDROP_OPENCV_ANDROID_SDK=${nativeDependency("trdrop.opencvAndroidSdk")}",
                    "-DTRDROP_FFMPEG_ANDROID=${nativeDependency("trdrop.ffmpegAndroid")}",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.1"
        }
    }

    buildTypes {
        debug {
            // the analysis is far too slow without optimization
            externalNativeBuild {
                cmake {
                    arguments += "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
                }
            }
        }
        release {
            isMinifyEnabled = false
            // signed with the debug key so that it can be installed as it is; use your own key to publish
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
}
