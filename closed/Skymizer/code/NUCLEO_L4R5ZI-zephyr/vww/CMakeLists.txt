# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13.1)

option(ENERGY_MODE "Enable energy mode" 0)
# cmake args
set(WEST west)
set(BOARD nucleo_l4r5zi)
set(ENABLE_CMSIS TRUE)

find_package(Zephyr HINTS $ENV{ZEPHYR_BASE})
project(onnc_project)

if(${ENABLE_CMSIS})
  set(CMSIS_PATH $ENV{CMSIS_PATH})

  file(GLOB_RECURSE cmsis_lib_srcs
    ${CMSIS_PATH}/CMSIS/NN/Source/ActivationFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/BasicMathFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/ConcatenationFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/ConvolutionFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/PoolingFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/ReshapeFunctions/*.c
    ${CMSIS_PATH}/CMSIS/NN/Source/SoftmaxFunctions/*.c
  )

  set(cmsis_includes 
    ${CMSIS_PATH}/CMSIS/Core/Include
    ${CMSIS_PATH}/CMSIS/NN/Include
    ${CMSIS_PATH}/CMSIS/DSP/Include
    ${CMSIS_PATH}/CMSIS/DSP/Include/dsp
  )
else()
  set(cmsis_lib_srcs "")
  set(cmsis_includes "")
endif()


# define a library for the model sources.
zephyr_library_named(onnc_model)
file(GLOB_RECURSE onnc_model_srcs vww01_code/*.cpp)
target_sources(onnc_model PRIVATE ${onnc_model_srcs})
target_include_directories(onnc_model PRIVATE
  ${CMAKE_SOURCE_DIR} 
  vww01_code
  ${cmsis_includes}
)
target_compile_options(onnc_model PRIVATE -Wno-unused-variable)  # onnc-generated code tends to include lots of these.
target_link_libraries(app PRIVATE onnc_model)

zephyr_library_named(tinymlperf_internal)
file(GLOB_RECURSE tiny_internal_srcs src/api/*.cpp)
target_sources(tinymlperf_internal PRIVATE ${tiny_internal_srcs})
target_compile_options(tinymlperf_internal PRIVATE -Wno-unused-variable)  # onnc-generated code tends to include lots of these.
target_link_libraries(app PRIVATE tinymlperf_internal)

file(GLOB_RECURSE app_srcs src/**.cpp)
target_sources(app PRIVATE
  ${app_srcs}
  ${cmsis_lib_srcs}
)
target_include_directories(app PRIVATE
  ${CMAKE_SOURCE_DIR} 
  vww01_code
  ${cmsis_includes}
)
target_compile_definitions(tinymlperf_internal PUBLIC -DEE_CFG_ENERGY_MODE=${ENERGY_MODE})