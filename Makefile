# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/filip/Documents/university/c++/zapoctak/neural-net

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/filip/Documents/university/c++/zapoctak/neural-net/cmake-build-debug

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/filip/Documents/university/c++/zapoctak/neural-net/cmake-build-debug/CMakeFiles /Users/filip/Documents/university/c++/zapoctak/neural-net/cmake-build-debug/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/filip/Documents/university/c++/zapoctak/neural-net/cmake-build-debug/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named neural_net

# Build rule for target.
neural_net: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 neural_net
.PHONY : neural_net

# fast build rule for target.
neural_net/fast:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/build
.PHONY : neural_net/fast

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/main.cpp.s
.PHONY : main.cpp.s

network/Network.o: network/Network.cpp.o

.PHONY : network/Network.o

# target to build an object file
network/Network.cpp.o:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/Network.cpp.o
.PHONY : network/Network.cpp.o

network/Network.i: network/Network.cpp.i

.PHONY : network/Network.i

# target to preprocess a source file
network/Network.cpp.i:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/Network.cpp.i
.PHONY : network/Network.cpp.i

network/Network.s: network/Network.cpp.s

.PHONY : network/Network.s

# target to generate assembly for a file
network/Network.cpp.s:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/Network.cpp.s
.PHONY : network/Network.cpp.s

network/layers/DenseRelu.o: network/layers/DenseRelu.cpp.o

.PHONY : network/layers/DenseRelu.o

# target to build an object file
network/layers/DenseRelu.cpp.o:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/layers/DenseRelu.cpp.o
.PHONY : network/layers/DenseRelu.cpp.o

network/layers/DenseRelu.i: network/layers/DenseRelu.cpp.i

.PHONY : network/layers/DenseRelu.i

# target to preprocess a source file
network/layers/DenseRelu.cpp.i:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/layers/DenseRelu.cpp.i
.PHONY : network/layers/DenseRelu.cpp.i

network/layers/DenseRelu.s: network/layers/DenseRelu.cpp.s

.PHONY : network/layers/DenseRelu.s

# target to generate assembly for a file
network/layers/DenseRelu.cpp.s:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/layers/DenseRelu.cpp.s
.PHONY : network/layers/DenseRelu.cpp.s

network/layers/Softmax.o: network/layers/Softmax.cpp.o

.PHONY : network/layers/Softmax.o

# target to build an object file
network/layers/Softmax.cpp.o:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/layers/Softmax.cpp.o
.PHONY : network/layers/Softmax.cpp.o

network/layers/Softmax.i: network/layers/Softmax.cpp.i

.PHONY : network/layers/Softmax.i

# target to preprocess a source file
network/layers/Softmax.cpp.i:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/layers/Softmax.cpp.i
.PHONY : network/layers/Softmax.cpp.i

network/layers/Softmax.s: network/layers/Softmax.cpp.s

.PHONY : network/layers/Softmax.s

# target to generate assembly for a file
network/layers/Softmax.cpp.s:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/network/layers/Softmax.cpp.s
.PHONY : network/layers/Softmax.cpp.s

utils/DataManage.o: utils/DataManage.cpp.o

.PHONY : utils/DataManage.o

# target to build an object file
utils/DataManage.cpp.o:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/utils/DataManage.cpp.o
.PHONY : utils/DataManage.cpp.o

utils/DataManage.i: utils/DataManage.cpp.i

.PHONY : utils/DataManage.i

# target to preprocess a source file
utils/DataManage.cpp.i:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/utils/DataManage.cpp.i
.PHONY : utils/DataManage.cpp.i

utils/DataManage.s: utils/DataManage.cpp.s

.PHONY : utils/DataManage.s

# target to generate assembly for a file
utils/DataManage.cpp.s:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/utils/DataManage.cpp.s
.PHONY : utils/DataManage.cpp.s

utils/MnistManage.o: utils/MnistManage.cpp.o

.PHONY : utils/MnistManage.o

# target to build an object file
utils/MnistManage.cpp.o:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/utils/MnistManage.cpp.o
.PHONY : utils/MnistManage.cpp.o

utils/MnistManage.i: utils/MnistManage.cpp.i

.PHONY : utils/MnistManage.i

# target to preprocess a source file
utils/MnistManage.cpp.i:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/utils/MnistManage.cpp.i
.PHONY : utils/MnistManage.cpp.i

utils/MnistManage.s: utils/MnistManage.cpp.s

.PHONY : utils/MnistManage.s

# target to generate assembly for a file
utils/MnistManage.cpp.s:
	$(MAKE) -f CMakeFiles/neural_net.dir/build.make CMakeFiles/neural_net.dir/utils/MnistManage.cpp.s
.PHONY : utils/MnistManage.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... neural_net"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... network/Network.o"
	@echo "... network/Network.i"
	@echo "... network/Network.s"
	@echo "... network/layers/DenseRelu.o"
	@echo "... network/layers/DenseRelu.i"
	@echo "... network/layers/DenseRelu.s"
	@echo "... network/layers/Softmax.o"
	@echo "... network/layers/Softmax.i"
	@echo "... network/layers/Softmax.s"
	@echo "... utils/DataManage.o"
	@echo "... utils/DataManage.i"
	@echo "... utils/DataManage.s"
	@echo "... utils/MnistManage.o"
	@echo "... utils/MnistManage.i"
	@echo "... utils/MnistManage.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

