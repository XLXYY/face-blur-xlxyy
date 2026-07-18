[app]
title = 星澜小月月的人脸识别处理系统
package.name = xinglanfaceblur
package.domain = org.xinglan
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx
source.include_patterns = models/*
version = 1.0
requirements = python3,kivy==2.2.1,opencv,numpy,requests,plyer,android
orientation = portrait
fullscreen = 1
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.api = 31
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.build_tools_version = 34.0.0
android.gradle_dependencies = 
p4a.branch = develop
p4a.bootstrap = sdl2
p4a.ndk_version = 25b
ios.kivy_ios_branch = master
