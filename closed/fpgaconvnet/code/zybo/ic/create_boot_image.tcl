# create the workspace
setws ./workspace

# create the hardware project
createhw -name hw -hwspec ./hw/design_1_wrapper.hdf

# create the bootloader
createapp -name fsbl -app {Zynq FSBL} -proc ps7_cortexa9_0 -hwproject hw -os standalone

# create the eembc runner project and configure
createapp -name eembc -app {Empty Application} -proc ps7_cortexa9_0 -hwproject hw -os standalone
configapp -app eembc -add libraries m
configapp -app eembc -add linker-misc {-Wl,--defsym=_HEAP_SIZE=0x20000}
configapp -app eembc -add linker-misc {-Wl,--defsym=_STACK_SIZE=0x20000}
configbsp -bsp eembc_bsp stdin ps7_uart_1
configbsp -bsp eembc_bsp stdout ps7_uart_1
regenbsp -bsp eembc_bsp
importsources -name eembc -path host/
projects -build
exec bootgen -arch zynq -image boot/boot.bif -w -o boot/BOOT.bin
