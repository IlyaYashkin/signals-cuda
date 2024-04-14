LIBS=utils/calc.cu utils/error.cu signal/signal.cu
ARCH=-arch=sm_61
FLAGS=-Xcompiler -fopenmp

build_standard:
	nvcc $(LIBS) signal_standard_akf.cu $(ARCH) $(FLAGS) -o start ; ./start

build_doppler:
	nvcc $(LIBS) signal_doppler.cu $(ARCH) $(FLAGS) -o start ; ./start

build_doppler_backup:
	nvcc $(LIBS) signal_doppler_backup.cu $(ARCH) $(FLAGS) -o start ; ./start

build_standard_fft:
	nvcc signal_fft.cu $(LIBS) $(ARCH) $(FLAGS) -o start ; ./start