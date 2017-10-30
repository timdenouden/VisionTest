# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Tim/Downloads/opencv-2.4.13

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Tim/Downloads/opencv-2.4.13/StaticLibs

# Include any dependencies generated for this target.
include modules/stitching/CMakeFiles/opencv_stitching.dir/depend.make

# Include the progress variables for this target.
include modules/stitching/CMakeFiles/opencv_stitching.dir/progress.make

# Include the compile flags for this target's objects.
include modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make

modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o: ../modules/stitching/src/autocalib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/autocalib.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/autocalib.cpp > CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/autocalib.cpp -o CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o: ../modules/stitching/src/blenders.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/blenders.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/blenders.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/blenders.cpp > CMakeFiles/opencv_stitching.dir/src/blenders.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/blenders.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/blenders.cpp -o CMakeFiles/opencv_stitching.dir/src/blenders.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o: ../modules/stitching/src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/camera.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/camera.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/camera.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/camera.cpp > CMakeFiles/opencv_stitching.dir/src/camera.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/camera.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/camera.cpp -o CMakeFiles/opencv_stitching.dir/src/camera.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o: ../modules/stitching/src/exposure_compensate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/exposure_compensate.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/exposure_compensate.cpp > CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/exposure_compensate.cpp -o CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o: ../modules/stitching/src/matchers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/matchers.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/matchers.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/matchers.cpp > CMakeFiles/opencv_stitching.dir/src/matchers.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/matchers.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/matchers.cpp -o CMakeFiles/opencv_stitching.dir/src/matchers.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o: ../modules/stitching/src/motion_estimators.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/motion_estimators.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/motion_estimators.cpp > CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/motion_estimators.cpp -o CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o: ../modules/stitching/src/seam_finders.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/seam_finders.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/seam_finders.cpp > CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/seam_finders.cpp -o CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o: ../modules/stitching/src/stitcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/stitcher.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/stitcher.cpp > CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/stitcher.cpp -o CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o: ../modules/stitching/src/util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/util.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/util.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/util.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/util.cpp > CMakeFiles/opencv_stitching.dir/src/util.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/util.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/util.cpp -o CMakeFiles/opencv_stitching.dir/src/util.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o


modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o: modules/stitching/CMakeFiles/opencv_stitching.dir/flags.make
modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o: ../modules/stitching/src/warpers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o -c /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/warpers.cpp

modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_stitching.dir/src/warpers.cpp.i"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/warpers.cpp > CMakeFiles/opencv_stitching.dir/src/warpers.cpp.i

modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_stitching.dir/src/warpers.cpp.s"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Tim/Downloads/opencv-2.4.13/modules/stitching/src/warpers.cpp -o CMakeFiles/opencv_stitching.dir/src/warpers.cpp.s

modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.requires:

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.requires

modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.provides: modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.requires
	$(MAKE) -f modules/stitching/CMakeFiles/opencv_stitching.dir/build.make modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.provides.build
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.provides

modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.provides.build: modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o


# Object files for target opencv_stitching
opencv_stitching_OBJECTS = \
"CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/camera.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/util.cpp.o" \
"CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o"

# External object files for target opencv_stitching
opencv_stitching_EXTERNAL_OBJECTS =

lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/build.make
lib/libopencv_stitching.a: modules/stitching/CMakeFiles/opencv_stitching.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Tim/Downloads/opencv-2.4.13/StaticLibs/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX static library ../../lib/libopencv_stitching.a"
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && $(CMAKE_COMMAND) -P CMakeFiles/opencv_stitching.dir/cmake_clean_target.cmake
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_stitching.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/stitching/CMakeFiles/opencv_stitching.dir/build: lib/libopencv_stitching.a

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/build

modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/autocalib.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/blenders.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/camera.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/exposure_compensate.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/matchers.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/motion_estimators.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/seam_finders.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/stitcher.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/util.cpp.o.requires
modules/stitching/CMakeFiles/opencv_stitching.dir/requires: modules/stitching/CMakeFiles/opencv_stitching.dir/src/warpers.cpp.o.requires

.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/requires

modules/stitching/CMakeFiles/opencv_stitching.dir/clean:
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching && $(CMAKE_COMMAND) -P CMakeFiles/opencv_stitching.dir/cmake_clean.cmake
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/clean

modules/stitching/CMakeFiles/opencv_stitching.dir/depend:
	cd /Users/Tim/Downloads/opencv-2.4.13/StaticLibs && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Tim/Downloads/opencv-2.4.13 /Users/Tim/Downloads/opencv-2.4.13/modules/stitching /Users/Tim/Downloads/opencv-2.4.13/StaticLibs /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching /Users/Tim/Downloads/opencv-2.4.13/StaticLibs/modules/stitching/CMakeFiles/opencv_stitching.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/stitching/CMakeFiles/opencv_stitching.dir/depend

