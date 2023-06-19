if [[ -d build ]]; then
    rm -r build
fi
touch onnc_runtime.h
mkdir build && cd build
cmake ..
make -j2
