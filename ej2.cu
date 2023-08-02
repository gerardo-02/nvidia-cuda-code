#include <cstdio>
#include <cstdlib>     // srand, rand
#include <ctime>       // time
#include <sys/time.h>  // get_wall_time

#define TOP_RANGE 10000.0

void* inicializarVector(unsigned long nElementos){
        double* vec = (double*) malloc( nElementos * sizeof(double) );
        for(unsigned long i = 0; i<nElementos; i++){
                vec[i] = (((double)rand()) / RAND_MAX) * TOP_RANGE;
                if (vec[i] < 0.0)
                        vec[i] = vec[i] * -1.0;
        }
        return vec;
}

void maxValCPU(double* vec, unsigned long nElementos, double* max){
        *max = 0.0;
        for(int i = 0; i<nElementos; i++){
                if (vec[i] > *max){
                        *max = vec[i];
                }
        }

}

__global__ void maxValKERNEL(double* v, unsigned long size, double* max){
	extern __shared__ double array[];
	double* temp = (double*) array;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
        double thread_max = 0.0;
        while(tid < size){
                if (v[tid] > thread_max)
                        thread_max = v[tid];
                tid += (gridDim.x * blockDim.x);
        }
        temp[threadIdx.x] = thread_max;
	
        __syncthreads();

        int i, division_impar;
        if (blockDim.x % 2 == 0){
                i = blockDim.x / 2;
                division_impar = 0;
        }
        else{
                i = blockDim.x / 2 + 1;
                division_impar = 1;
        }
        while(i != 0){
                if(threadIdx.x < i){
                        if (division_impar == 0 || threadIdx.x < i-1){
                                if (temp[threadIdx.x + i] > temp[threadIdx.x]){
                                        temp[threadIdx.x] = temp[threadIdx.x + i];
                                }
                        }
                }
                __syncthreads();
                if(i % 2 == 0 || i == 1){
                    i = i/2;
                    division_impar = 0;
                }
                else{
                    i = i/2 + 1;
                    division_impar = 1;
                }
        }
	
        if (threadIdx.x == 0){
                max[blockIdx.x] = temp[0];
        }
}

int main(void){

        srand(time(NULL));
	unsigned long tamVector, tamBloque, tamGrid;
        printf("Introduzca el tamaño del vector: ");
        do{
                scanf("%ld", &tamVector);
        }while(tamVector <= 0);

        printf("Introduzca el tamaño del bloque: ");
        do{
                scanf("%ld", &tamBloque);
        }while(tamBloque <= 0);

        if (tamVector % tamBloque != 0){
                printf("Tamaños incorrectos introducidos. El tamaño del vector debe ser múltiplo del tamaño del bloque.");
                return 0;
        }

        tamGrid = tamVector / tamBloque;

        double* vec = (double*) inicializarVector(tamVector);
        double max_val_cpu = 0.0, max_val_kernel = 0.0;
        maxValCPU(vec, tamVector, &max_val_cpu);
        printf("Valor más alto CPU: %lf\n", max_val_cpu);


        double* dev_vec = 0;
        double* dev_max_bloques = 0;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliSeconds = 0.0;

        cudaMalloc( (void**) &dev_vec, tamVector * sizeof(double) );
        cudaMemcpy( dev_vec, vec, tamVector * sizeof(double), cudaMemcpyHostToDevice );
        cudaMalloc( (void**) &dev_max_bloques, tamGrid * sizeof( double ) );

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        maxValKERNEL<<<tamGrid, tamBloque, tamBloque * sizeof(double)>>>(dev_vec, tamVector, dev_max_bloques);
	
        double max_bloques[tamGrid];
        cudaMemcpy(max_bloques, dev_max_bloques, tamGrid * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i=0; i<tamGrid; i++){
		
                if (max_bloques[i] > max_val_kernel){
                        max_val_kernel = max_bloques[i];
		}
	
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliSeconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("Valor más alto GPU: %lf\n", max_val_kernel);
        
        if(max_val_cpu == max_val_kernel)
                printf("Calculo correcto!!\n");
	else
		printf("Calculo incorrecto D:\n");

        printf("Tiempo de ejecucion del kernel<<<%ld, %ld>>> sobre %ld elementos [s]: %.4f\n", tamGrid, tamBloque, tamVector, milliSeconds / 1000.0);

        free(vec);
        cudaFree(dev_vec);
        cudaFree(dev_max_bloques);
        return 0;
}
