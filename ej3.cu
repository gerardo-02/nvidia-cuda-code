#include <cstdio>
#include <cstdlib>     // srand, rand
#include <ctime>       // time
#include <sys/time.h>  // get_wall_time

#define TOP_RANGE 10.0

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        printf("Error en la medicion de tiempo CPU!!\n");
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void* inicializarMatriz(unsigned long grado){
        double* mat = (double*) malloc( grado * grado * sizeof(double));

        for(unsigned long i = 0; i<grado; i++){
            for (unsigned long j = 0; j<grado; j++){
                mat[i * grado + j] = (((double)rand()) / RAND_MAX) * TOP_RANGE;
                if (mat[i * grado + j] < 0.0)
                        mat[i * grado + j] = mat[i * grado + j] * -1.0;
            }
        }
        return mat;
}

void mostrarMatriz(double* mat, unsigned long grado){
    for (int i=0; i<grado; i++){
        for (int j=0; j<grado; j++){
            printf("%lf\t", mat[i * grado + j]);
        }
        printf("\n");
    }
}

void productoCPU(double* matA, double* matB, double* resultado, unsigned long grado){
    for (int i = 0; i < grado; i++) {
        // Dentro recorremos las filas de la matriz resultado
        for (int j = 0; j < grado; j++) {
            double suma = 0;
            // Y cada columna de la matriz resultado
            for (int k = 0; k < grado; k++) {
                // Multiplicamos y sumamos lo correspondiente entre la primera y segunda
                suma += matA[i * grado + k] * matB[k * grado + j];
            }
            // Lo acomodamos dentro del producto
            resultado[i * grado + j] = suma;
        }
    }
}

__global__ void productoKernel(double* matA, double* matB, double* resultado, unsigned long grado){
        extern __shared__ double array[];
        double* multiplicaciones = (double*) array;
        //multiplicaciones[threadIdx.x] = 0.0;

        int tid = threadIdx.x;
        double suma = 0.0;
        while(tid < grado){
                //atomicAdd( &celda, matA[blockIdx.x * grado + tid] * matB[tid * grado + blockIdx.y] );
                suma += matA[blockIdx.x * grado + tid] * matB[tid * grado + blockIdx.y];
                tid += 1024;
        }

        multiplicaciones[threadIdx.x] = suma;

        __syncthreads();

        int i, division_impar;
        if ( blockDim.x % 2 == 0 ){
                i = blockDim.x / 2;
                division_impar = 0;
        }
        else{
                i = blockDim.x / 2 + 1;
                division_impar = 1;
        }

        while(i != 0){
                if(threadIdx.x < i){
                        if (division_impar == 0 || threadIdx.x < i-1)
                                multiplicaciones[threadIdx.x] += multiplicaciones[threadIdx.x + i];
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
                resultado[blockIdx.x * grado + blockIdx.y] = multiplicaciones[0];
        }
}

int main(void){

        srand(time(NULL));
        unsigned long grado, tamBloque;
        printf("Introduzca el grado de las matrices: ");
        do{
                scanf("%ld", &grado);
        }while(grado <= 0);

        dim3 tamGrid(grado, grado, 1);
        if (grado <= 1024)
            tamBloque = grado;
        else
            tamBloque = 1024;

        double* matA = (double*) inicializarMatriz(grado);
        double* matB = (double*) inicializarMatriz(grado);

        double* resultadoCPU = (double*) malloc( grado * grado * sizeof(double));

        double* resultadoKernel = (double*) malloc( grado * grado * sizeof(double));

        double cpu_start = get_wall_time();
        productoCPU(matA, matB, resultadoCPU, grado);
        double cpu_end = get_wall_time();
        double CPUS = cpu_end - cpu_start;
        printf("Tiempo de CPU [s]: %.4lf\n", CPUS);

        if (grado < 8){
            printf("\n\nMatriz A:\n\n");
            mostrarMatriz(matA, grado);
            printf("\n\nMatriz B:\n\n");
            mostrarMatriz(matB, grado);
            printf("\n\nMatriz resultadoCPU:\n\n");
            mostrarMatriz(resultadoCPU, grado);
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float milliSeconds = 0.0;

        double* dev_matA = 0;
        double* dev_matB = 0;
        double* dev_resultadoKernel = 0;

        cudaMalloc( (void**) &dev_matA, grado * grado * sizeof(double) );
        cudaMemcpy( dev_matA, matA, grado * grado * sizeof(double), cudaMemcpyHostToDevice );

        cudaMalloc( (void**) &dev_matB, grado * grado * sizeof(double) );
        cudaMemcpy( dev_matB, matB, grado * grado * sizeof(double), cudaMemcpyHostToDevice );

        cudaMalloc( (void**) &dev_resultadoKernel, grado * grado * sizeof(double) );

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        productoKernel<<<tamGrid, tamBloque, grado * sizeof(double)>>>(dev_matA, dev_matB, dev_resultadoKernel, grado);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaMemcpy(resultadoKernel, dev_resultadoKernel, grado * grado * sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&milliSeconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (grado < 8){
            printf("\n\nMatriz resultadoKernel:\n\n");
            mostrarMatriz(resultadoKernel, grado);
        }

        int correcto = 1;
        for (int i=0; i<grado*grado; i++){

                if (resultadoCPU[i] - resultadoKernel[i] < -0.00000001 || resultadoCPU[i] - resultadoKernel[i] > 0.00000001){
                    correcto = 0;
                    printf("Incorrecto en posicion %ld %ld con diferencia %.15lf\n", i / grado, i % grado, resultadoCPU[i] - resultadoKernel[i]);
                }

        }
        if (correcto == 1)
            printf("Calculo correcto!!\n");
        else
            printf("Calculo INCORRECTO\n");

        printf("Tiempo de ejecucion del kernel<<<%ld, %ld>>> sobre matrices cuadradas de grado %ld [s]: %.4f\n", grado * grado, tamBloque, grado, milliSeconds / 1000.0);

        free(matA);
        free(matB);
        free(resultadoCPU);
        free(resultadoKernel);
        cudaFree(dev_matA);
        cudaFree(dev_matB);
        cudaFree(dev_resultadoKernel);
        return 0;
}
