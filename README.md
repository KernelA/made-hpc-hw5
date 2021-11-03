# made-hpc-hw5

## Description


## Requirements

1. CMake 3.20 or higher.
2. Compiler with support C++17.

## How to run

Create new directory `build`.

```
mkdir build
cmake -S . -B .\build
```

Run CMake:
```
cmake --build .\build --config Release -j 4 --target <input_target>
```
