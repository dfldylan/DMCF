cudalib = /usr/local/cuda/lib64
cudainc = /usr/local/cuda/include
cudabin = /usr/local/cuda/bin

install_dir = $(HOME)/.local/lib

$(eval TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
$(eval TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))

all: tools
tools: tools/sampling_so.so tools/nn_distance_so.so tools/tf_approxmatch_so.so
    rm -f tools/*.o

tools/sampling_so.so: tools/sampling.cu.o tools/sampling.cpp
	g++ -std=c++14 tools/sampling.cpp tools/sampling.cu.o -o $(install_dir)/sampling_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -I$(cudainc) -lcudart -L$(cudalib) -O2

tools/sampling.cu.o: tools/sampling.cu
	$(cudabin)/nvcc -std=c++14 -c -o tools/sampling.cu.o tools/sampling.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES

tools/nn_distance_so.so: tools/nn_distance.cu.o tools/nn_distance.cpp
	g++ -std=c++14 tools/nn_distance.cpp tools/nn_distance.cu.o -o $(install_dir)/nn_distance_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -I$(cudainc) -lcudart -L$(cudalib) -O2

tools/nn_distance.cu.o: tools/nn_distance.cu
	$(cudabin)/nvcc -std=c++14 -c -o tools/nn_distance.cu.o tools/nn_distance.cu $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED

tools/tf_approxmatch_so.so: tools/tf_approxmatch.cu.o tools/tf_approxmatch.cpp
	g++ -std=c++14 tools/tf_approxmatch.cpp tools/tf_approxmatch.cu.o -o $(install_dir)/tf_approxmatch_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

tools/tf_approxmatch.cu.o: tools/tf_approxmatch.cu
	$(cudabin)/nvcc -std=c++14 -c -o tools/tf_approxmatch.cu.o tools/tf_approxmatch.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES

clean:
	rm -f $(install_dir)/sampling_so.so $(install_dir)/nn_distance_so.so $(install_dir)/tf_approxmatch_so.so
	rm -f tools/*.o
