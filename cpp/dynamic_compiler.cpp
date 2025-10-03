#include "include/dynamic_compiler.h"
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <memory>

namespace axe {
namespace jit {

// A simple counter for generating unique filenames to avoid collisions
static int function_counter = 0;

DynamicCompiler::DynamicCompiler() {}

DynamicCompiler::~DynamicCompiler() {
    for (void* handle : loaded_handles_) {
        dlclose(handle);
    }
}

CompiledFunction DynamicCompiler::compile_and_load(const std::string& source_code, const std::string& function_name) {
    // 1. Generate unique filenames for temporary files in /tmp/
    std::string base_name = "/tmp/axe_jit_func_" + std::to_string(function_counter++);
    std::string source_filename = base_name + ".cpp";
    std::string library_filename = base_name + ".so";

    // 2. Write the source code to the temporary file
    {
        std::ofstream source_file(source_filename);
        if (!source_file) {
            std::cerr << "Failed to open temporary source file: " << source_filename << std::endl;
            return nullptr;
        }
        source_file << source_code;
    }

    // 3. Construct the compiler command.
    // This is the most fragile part of the implementation, as it assumes a certain
    // project structure and compiler (g++). A more robust solution would use
    // configuration steps to find the correct paths and compiler.
    std::stringstream command;
    command << "g++ -shared -fPIC -std=c++17"
            << " -o " << library_filename
            << " " << source_filename
            // Include directories for our project and dependencies
            << " -I cpp/include"
            << " -I build/_deps/eigen3-src"
            // Link against the object files of our core library to resolve symbols.
            // This is a workaround for not having a shared libaxe_core.so.
            << " build/cpp/CMakeFiles/axe_core.dir/*.o";

    // 4. Execute the compiler command
    int ret = system(command.str().c_str());
    if (ret != 0) {
        std::cerr << "JIT compilation failed. The command was:\n" << command.str() << std::endl;
        unlink(source_filename.c_str());
        return nullptr;
    }

    // 5. Load the compiled shared library
    void* handle = dlopen(library_filename.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Failed to load shared library: " << dlerror() << std::endl;
        unlink(source_filename.c_str());
        unlink(library_filename.c_str());
        return nullptr;
    }
    loaded_handles_.push_back(handle);

    // 6. Get a pointer to the loaded function by its name
    CompiledFunction func = (CompiledFunction)dlsym(handle, function_name.c_str());
    if (!func) {
        std::cerr << "Failed to find symbol '" << function_name << "': " << dlerror() << std::endl;
        unlink(source_filename.c_str());
        unlink(library_filename.c_str());
        return nullptr;
    }

    // 7. Cleanup temporary files. The .so file is now loaded into memory.
    unlink(source_filename.c_str());
    unlink(library_filename.c_str());

    return func;
}

} // namespace jit
} // namespace axe