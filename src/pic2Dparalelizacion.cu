/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <fftw.h>
#include <fstream>
#include <curand_kernel.h>
//#include <cufft.h>
#include <complex>



cudaError_t error = cudaSuccess;

float L, LL;

int N, C, itera;
float t;

float blockSize = 1024;


using namespace std;


// función Maxwelliana de la distribución de las partículas.
__device__ float distribution(float vb, float aleatorio, curandState *states) //generador de distribución maxwelliana para la velocidad
		{

	// Genera un valor random v
	float fmax = 0.5 * (1.0 + exp(-2.0 * vb * vb));
	float vmin = -5.0 * vb;
	float vmax = +5.0 * vb;
	float v;
	float f;
	float x;
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	while (true) {
		v = vmin + ((vmax - vmin) * aleatorio);
		f = 0.5 * (exp(-(v - vb) * (v - vb) / 2.0) + exp(-(v + vb) * (v + vb) / 2.0));
		x = fmax * aleatorio;
		if (x > f)
			aleatorio = curand_uniform(states + Idx);
		else
			return v;
	}

	//return 0.0;
}
//Distribución aleatoria de las partículas.
__global__ void distribucionParticulas(float *rx, float *ry, float *vx,
		float *vy, int N, curandState *states, float vb, float L, int seed) {
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);

	if (Idx < N) {
		rx[Idx] = L*curand_uniform(states + Idx); //inicializando la posicion aleatoria en x
		ry[Idx] = L*curand_uniform(states + Idx);
		vx[Idx] = distribution(vb, curand_uniform(states + Idx), states); //;L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana
		vy[Idx] = distribution(vb, curand_uniform(states + Idx), states); //L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana

	}

}

// inicialización de la densidad.
__global__ void inicializacionVariables(float *ne, float *n, float *phi, float *Ex, float *Ey, int C){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < C*C){
		ne[i]=0.;
		n[i]=0.;
		phi[i]=0.;
		Ex[i]=0.;
		Ey[i]=0.;

	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////

// Calculo de la densidad.

__global__ void calculoDensidadInicializacionCeldas(float *rx, float *ry,
		int *jx, int *jy, float *yx, float *yy, int N, int C, float L) {
	int Id = blockIdx.x * blockDim.x + threadIdx.x;
	float dx = L / float(C);
	//float dxx = L /float(C*C);
	if (Id < N) {
		jx[Id] = int(rx[Id] / dx); //posicion en x de la particula
		jy[Id] = int(ry[Id] / dx); //posicion en y de la particula
		yx[Id] = (rx[Id] / dx) - (float) jx[Id]; //posicion exacta de la particula en x de la celda "j"
		yy[Id] = (ry[Id] / dx) - (float) jy[Id];
	}

}

__global__ void calculoDensidadThreaded(float *ne, int *jx, int *jy, float *yx, int C,
		float L, int N) {
	float dxx = L / float(C * C);
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N) {
		ne[(jy[i] * C) + jx[i]] += (1. - yx[i]) / dxx;
		if (jx[i] + 1 == C)
			ne[(jy[i] * C)] += yx[i] / dxx;
		else
			ne[(jy[i] * C) + jx[i] + 1] += yx[i] / dxx;
	}

}


__global__ void calculoDensidad(float *ne, int *jx, int *jy, float *yx, int C,
		float L, int N) {
	float dxx = L / float(C * C);
	//int Id = blockIdx.x*blockDim.x + threadIdx.x;
	for (int i = 0; i < N; i++) {
		ne[(jy[i] * C) + jx[i]] += (1. - yx[i]) / dxx;
		if (jx[i] + 1 == C)
			ne[(jy[i] * C)] += yx[i] / dxx;
		else
			ne[(jy[i] * C) + jx[i] + 1] += yx[i] / dxx;
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////
void Densidad(float *ne_d,float *rx_d, float *ry_d, int *jx_d,
		int *jy_d, float *yx_d, float *yy_d, int C, float L, int N) {
	//definicion de los bloques.
//	float blockSize = 1024;
	dim3 dimBlock(ceil(N / blockSize), 1, 1);
	dim3 dimBlock2(ceil(C * C / blockSize), 1, 1);
	dim3 dimGrid(blockSize, 1, 1);

	calculoDensidadInicializacionCeldas<<<blockSize, dimBlock>>>(rx_d, ry_d,
			jx_d, jy_d, yx_d, yy_d, N, C, L);
	cudaDeviceSynchronize();
	calculoDensidadThreaded<<<dimGrid, dimBlock>>>(ne_d, jx_d, jy_d, yx_d, C, L, N); //proceso de mejora.
	cudaDeviceSynchronize();

}
///////////////////////////////////////////////////////////////////////////////////////////////////

// normalización de los datos para la funcion de evaluar.
__global__ void normalizacionDensidadEval(float *ne, float *n, int N, int C,
		float L) {
	int Id = blockIdx.x * blockDim.x + threadIdx.x;
	float n0 = float (N)/L;
	if (Id < C * C) {
		n[Id] = ne[Id] / float(n0) - 1;
	}

}
// normalizacion de la densidad para la función output
__global__ void normalizacionDensidadOutput(float *ne, float *n1, int N, int C,
		float L) {
	int Id = blockIdx.x * blockDim.x + threadIdx.x;
	if (Id < C * C) {
		n1[Id] = float(C*C)* ne[Id]/ float(N) -1;
	}

}
/////////////////////////////////AQUI EMPIEZA POISSON /////////////////////////////////////////////////////////////////////

void poisson (float * rho, float *phi){

	float dx = L/float(C-1);


	vector<float>  Ur(C*C), Ui(C*C);

	fftw_complex U[C][C];
	fftw_complex FF[C],ff[C];


//	ofstream init; //archivo que imprime la matriz de espacio de simulacion con la densidad de carga de cada celda
//		init.open("entrada_poisson");
//		for (int i = 0; i < C; i++){
//			for (int j = 0; j < C; j++){
//				init<<rho[(C*i)+j]<<" ";
//			}
//			init<<endl;
//		}
//		init.close();


		for(int i=0;i<C;i++){
			for(int j=0;j<C;j++){
				c_re (ff[j]) = rho[(C*i)+j];//asignando valores a la parte real de ff
				c_im (ff[j]) = 0.;
			}

			fftw_plan p = fftw_create_plan (C, FFTW_FORWARD, FFTW_ESTIMATE); //se crea el plan para realizar la transformada hacia adelante
			fftw_one (p, ff, FF); //se ejecuta el plan en 1D, primero por cada fila
			fftw_destroy_plan (p);//se libera memoria
			for(int x=0;x<C;x++){
				U[i][x].re=FF[x].re/float(C*C);//el resultado real de la transformada se pone en la matriz U
				U[i][x].im=FF[x].im/float(C*C);//al igual que la parte imaginaria
				//se descargan los valores de la salida FF en la matriz original U
				}
		}


		//init.close();
		for(int i=0;i<C;i++){
			for(int j=0;j<C;j++){
				c_re (ff[j]) = U[j][i].re;//se toman los valores encontrados en la transformada en x y se disponen para realizar la transformada en la direccion "y"
				c_im (ff[j]) = U[j][i].im;
			}
			fftw_plan p = fftw_create_plan (C, FFTW_FORWARD, FFTW_ESTIMATE);//se crea el plan para realizar la transformada la direccion "y"
			fftw_one (p, ff, FF);//se ejecuta el plan en 1D, ahora por columnas, con los resultados adquiridos previamente en la transformada en la direccion "x"
			fftw_destroy_plan (p);//se libera memoria
			for(int j=0;j<C;j++){
				U[j][i].re=FF[j].re/float(C*C);//el resultado real de la transformada se pone en la matriz U
				U[j][i].im=FF[j].im/float(C*C);//al igual que la parte imaginaria
			}
		}



		//calculo de poisson:
		// ver documento para entender esquema de discretizacion

		U[0][0].re =0.0;

		complex <double> i(0.0, L); //creamos una variable compleja para poder aplicar la discretizacion.
		complex<double> W1 = exp(2.0 * M_PI * i / double(C));
		complex<float> W = complex<float>(W1);
		complex <float> Wm = L, Wn = L;
		for (int m = 0; m < C; m++)
		{
			for (int n = 0; n < C; n++)
			{
				complex<float> denom = 4.0;
				denom -= Wm + L / Wm + Wn + L / Wn; //se calcula el denominador para cada celda, segun el equema de discretizacion
				if (denom != float(0.0)){
					U[m][n].re *= dx *dx / denom.real();
					U[m][n].im *= dx *dx / denom.imag();
				}
				Wn *= W;//se multiplica por la constante W
			}
			Wm *= W;
		}


		U[0][0].re=0.;
		U[0][0].im=0.;



		for(int x=0;x<C;x++){
			for(int y=0;y<C;y++) //se toman los resultados de la matriz U y se ponen en los vectores temporales Ur y Ui, los cuales se les aplicara la transformada inversa, para recuperar los valores de phi
			{
				Ur[(x*C)+y]= U[x][y].re;
				Ui[(x*C)+y]= U[x][y].im;
			 }
		}



		for(int i=1;i<C;i++){ //en este caso, el vector de entrada para la transformada es FF y la salida ff. HAY QUE HACERLO DESDE 1 PORQUE O SINO DA NAN.
		  	for(int j=0;j<C;j++){
		  		c_re (FF[j]) = Ur[(C*i)+j];
		  		c_im (FF[j]) = Ui[(C*i)+j];
		  	}

		  	fftw_plan q = fftw_create_plan (C, FFTW_BACKWARD, FFTW_ESTIMATE);
		  	fftw_one (q, FF, ff);//se calcula la transformada inversa en el eje x
		  	fftw_destroy_plan (q);
		  	for(int j=0;j<C;j++){
		  		U[i][j].re = ff[j].re; //se retornan los resultados a la matriz U
		  		U[i][j].im = ff[j].im;
		  	}
		}



		for(int i=0;i<C;i++){//el mismo prodecimiento anterior pero ahora en el eje y
			for(int j=0;j<C;j++){
				c_re (FF[j]) = U[j][i].re;
				c_im (FF[j]) = U[j][i].im;
			}
			fftw_plan q = fftw_create_plan (C, FFTW_BACKWARD, FFTW_ESTIMATE);
			fftw_one (q, FF, ff);//se calcula la transformada
			fftw_destroy_plan (q);
			for(int j=0;j<C;j++){
				U[j][i].re = ff[j].re;
				U[j][i].im = ff[j].im;
			}
		}
		U[0][0].re =0.;


		for(int x=0;x<C;x++){
			for(int y=0;y<C;y++){
				phi[(x*C)+y] = U[x][y].re/float(1.e+7);//en este caso, solo tomamos la parte real, que es la que nos interesa. obtenemos como resultado el potencial electrostatico.
			}
		}

//
//		init.open("despues_inversa_y_salida");//se escribe un archivo de salida para analizar los datos. la salida corresponde al potencial electrostatico en cada celda conocido como phi.
//		for (int i = 0; i < C; i++){
//			for (int j = 0; j < C; j++){
//				init<<phi[(i*C)+j]<<" ";
//			}
//			init<<endl;
//		}
//
//		init.close();




}
//////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ElectricBordes(float *phi, float *Ex, float *Ey, float L, int C) // recibe el potencial electroestatico calculado por la funcion poisson  y se calcula el campo electrico, tanto para X como para Y
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float dx = L / float(C); // el delta de x representa el tamano de la malla

	if (i < C) {
		Ex[i*C] = (phi[((i+1)*C)-1] - phi[(i*C)+1])/(2. * dx);// hallando el campo en x, en la primera columna
		Ex[((i+1)*C)-1] = (phi[((i+1)*C)-2] - phi[(i*C)]) / (2. * dx);// hallando el campo en x, en la ultima columna
		Ey[((C-1)*C)+i] = (phi[((C-2)*C)+i] - phi[i]) / (2. * dx); //hallando el campo en "y" para la ultima fila
		Ey[i] = (phi[((C-1)*C)+i] - phi[i+C]) / (2. * dx);//hallando el campo para la primera fila y la ultima
	}

}

__global__ void electricPart2(float *phi, float *Ex, float L, int C){
	int z = blockIdx.x * blockDim.x + threadIdx.x;
	int w = blockIdx.y * blockDim.y + threadIdx.y;
	w+=1;
	float dx = L / float(C); // el delta de x representa el tamano de la malla

	if ((z<C) && (w<C-1)){
		Ex[w+(C*z)] = (phi[w-1] - phi[w+1]) / (2. * dx);
	}



}

__global__ void electricPart3(float *phi, float *Ey, float L, int C){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i+C;
	float dx = L / float(C);
	if (i<(C*C)-C)
		Ey[i] = (phi[i-C] - phi[i+C]) / (2. * dx);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
void Electric2 (float *phi_d, float *Ex_d, float *Ey_d) // recibe el potencial electroestatico calculado por la funcion poisson  y se calcula el campo electrico, tanto para X como para Y
{

//  float blockSize = 1024;
  dim3 dimBlock(ceil(C / blockSize), 1, 1);
  dim3 dimGrid(blockSize, 1, 1);
  ElectricBordes<<<dimGrid, dimBlock>>>(phi_d, Ex_d, Ey_d,  L, C);
  cudaDeviceSynchronize();

  dim3 dimBlock1(ceil(C  / blockSize),ceil(C  / blockSize),1);
  dim3 dimGrid1(blockSize,blockSize,1);
  electricPart2<<<dimGrid1,dimBlock1>>>(phi_d,Ex_d,L,C);
  cudaDeviceSynchronize();

  dim3 dimBlock2(ceil(C*C  / blockSize), 1, 1);
  electricPart3<<<dimGrid, dimBlock2>>>(phi_d, Ey_d,L,C);
  cudaDeviceSynchronize();

}
//////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void load(float *r,  float *v,
		float *y, int N) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < (N)) {
		y[i] = r[i];
		y[N+i] = v[i];


	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void unLoad(float *y, float *r, float *v,int N) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		r[i] = y[i];
		v[i] = y[N + i];

	}

}

__global__ void escapeParticulas(float *rx, float *ry, int N, float L) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		if (rx[i] < 0.) rx[i] += L;
		if (ry[i] < 0.) ry[i] += L;
		if (rx[i] > L) rx[i] -= L;
		if (ry[i] > L) ry[i] -= L;
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void campoParticula(float *rx, float *ry, float *v1, float *v2, float *Ex, float *Ey, float *r1dot, float *v1dot,
		float *r2dot, float *v2dot, float L, int N, int C){

	for (int i = 0; i < N; i++)
	{
		float dx = L / float (C);
		int jx = int (rx[i] / dx);
		int jy = int (ry[i] / dx);
		float yx = rx[i] / dx - float (jx);
		float yy = ry[i] / dx - float (jy);

		float Efieldx = 0.0;
		float Efieldy = 0.0;


		if ((jx+1)%C == 0)
			Efieldx = Ex[jx] * (1. - yx) + Ex[jx-(C-1)] * yx;
		else
			Efieldx = Ex[jx] * (1. - yx) + Ex[jx+1] * yx;
		if ((jy+1)%C == 0)
			Efieldy = Ey[jy] * (1. - yy) + Ey[jy-(C-1)] * yy;
		else
			Efieldy = Ey[jy] * (1. - yy) + Ey[jy+1] * yy;


		// por el esquema de normalización:
		//derivada de la posicion =velocidad
		//derivada de la velocidad = campo electrico en la posicion de la particula.
		r1dot[i] = v1[i];
		v1dot[i] = - Efieldx;
		r2dot[i] = v2[i];
		v2dot[i] = - Efieldy;

	}

}


/////////////////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void campoParticulaThreads(float *rx, float *ry, float *v1, float *v2, float *Ex, float *Ey, float *r1dot, float *v1dot,
		float *r2dot, float *v2dot, float L, int N, int C){


	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		float dx = L / float (C);
		int jx = int (rx[i] / dx);
		int jy = int (ry[i] / dx);
		float yx = rx[i] / dx - float (jx);
		float yy = ry[i] / dx - float (jy);

		float Efieldx = 0.0;
		float Efieldy = 0.0;


		if ((jx+1)%C == 0)
			Efieldx = Ex[jx] * (1. - yx) + Ex[jx-(C-1)] * yx;
		else
			Efieldx = Ex[jx] * (1. - yx) + Ex[jx+1] * yx;
		if ((jy+1)%C == 0)
			Efieldy = Ey[jy] * (1. - yy) + Ey[jy-(C-1)] * yy;
		else
			Efieldy = Ey[jy] * (1. - yy) + Ey[jy+1] * yy;


		// por el esquema de normalización:
		//derivada de la posicion =velocidad
		//derivada de la velocidad = campo electrico en la posicion de la particula.
		r1dot[i] = v1[i];
		v1dot[i] = - Efieldx;
		r2dot[i] = v2[i];
		v2dot[i] = - Efieldy;

	}

}


/////////////////////////////////////////////////////////////////////////////////////////////////////



void eval (float *rx_d, float *vx_d,float *ry_d, float *vy_d, float *dydt1_d, float *dydt2_d)

//esta función itera en cada uno de los 4 pasos de RK4.
//recibe la posicion y la velocidad en los vectores Y1 y Y2, extrae y los opera de la misma forma que la funciòn output
{

	int *jx_d, *jy_d; // posiciones de la malla.
	float *yx_d, *yy_d; // posiciones de la malla.
	float *ne_d;
	float *ne_h;
	float *n_h; // densidad normalizada.
	float *n_d; // densidad normalizada del dispositivo.
	float *phiFinal_h;
	float *phiFinal_d;
	float *Ex_h;
	float *Ey_h;
	float *Ex_d;
	float *Ey_d;
	float *r1dot_d, *v1dot_d, *r2dot_d, *v2dot_d;

	int size = N * sizeof(float);
	int size_ne = C * C * sizeof(float);

	ne_h = (float *) malloc(size_ne);
	n_h = (float *) malloc(size_ne);
	phiFinal_h = (float *) malloc(size_ne);
	Ex_h = (float *) malloc(size_ne);
	Ey_h = (float *) malloc(size_ne);


	error = cudaMalloc((void **) &jx_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria jx_d");
		exit(0);
	}
	error = cudaMalloc((void **) &jy_d, size);
	if (error != cudaSuccess){
			printf("Error de memoria jy_d");
			exit(0);
		}
	error = cudaMalloc((void **) &yx_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria yx_d");
		exit(0);
	}
	error = cudaMalloc((void **) &yy_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria yy_d");
		exit(0);
		}
	error = cudaMalloc((void **) &r1dot_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria r1dot_d");
		exit(0);
	}
	error = cudaMalloc((void **) &v1dot_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria v1dot_d");
		exit(0);
	}
	error = cudaMalloc((void **) &r2dot_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria r2dot_d");
		exit(0);
	}
	error = cudaMalloc((void **) &v2dot_d, size);
	if (error != cudaSuccess){
		printf("Error de memoria v2dot_d");
		exit(0);
	}
	error = cudaMalloc((void **) &ne_d, size_ne);
	if (error != cudaSuccess){
		printf("Error de memoria ne_d");
		exit(0);
	}
	error = cudaMalloc((void **) &n_d, size_ne);
	if (error != cudaSuccess){
		printf("Error de memoria jn_d");
		exit(0);
	}
	error = cudaMalloc((void **) &phiFinal_d, size_ne);
	if (error != cudaSuccess){
		printf("Error de memoria phiFInal_d");
		exit(0);
	}
	error = cudaMalloc((void **) &Ex_d, size_ne);
	if (error != cudaSuccess){
			printf("Error de memoria Ex_d");
			exit(0);
		}
	error = cudaMalloc((void **) &Ey_d, size_ne);
	if (error != cudaSuccess){
		printf("Error de memoria Ey_d");
		exit(0);
	}

//	float blockSize = 1024;

	// se reinyectan partículas que escapan del espacio de simulación
	//limites de logitud por donde se mueven las particulas
	dim3 dimBlock(ceil(N / blockSize), 1, 1);
	dim3 dimGrid(blockSize, 1, 1);
	escapeParticulas<<<dimGrid, dimBlock>>>(rx_d, ry_d, N, L);
	cudaDeviceSynchronize();

	//calculan la densidad con el numero de electrones que se encuentra dentro de la malla de estudio
	//cout<<"antes de la densidad en eval"<<endl;
	dim3 dimBlock2(ceil(C * C / blockSize), 1, 1);
	inicializacionVariables<<<dimGrid, dimBlock2>>>(ne_d, n_d, phiFinal_d, Ex_d, Ey_d, C);
	cudaDeviceSynchronize();

	Densidad(ne_d, rx_d, ry_d, jx_d, jy_d, yx_d, yy_d, C, L,N);

	normalizacionDensidadEval<<<dimGrid, dimBlock2>>>(ne_d, n_d, N, C,L);
	cudaDeviceSynchronize();

	error = cudaMemcpy(n_h, n_d, size_ne, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess){
		printf("Error copiando n_d_d");
		exit(0);
	}

	poisson (n_h, phiFinal_h);

//	ofstream init;
//	init.open("phiEval.txt");
//	for (int i = 0; i < C*C; i++) {
//		init << phiFinal_h[i] <<endl;
//
//	}
//	init.close();

	error = cudaMemcpy(phiFinal_d, phiFinal_h, size_ne, cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
			printf("Error copiando phiFInal_d");
			exit(0);
		}


	//calculo del campo electrico
	Electric2 ( phiFinal_d, Ex_d, Ey_d);

	campoParticulaThreads<<<dimGrid,dimBlock>>>(rx_d,ry_d,vx_d,vy_d,Ex_d,Ey_d,r1dot_d,v1dot_d,r2dot_d,v2dot_d,L,N,C);
	cudaDeviceSynchronize();

	// se vuelven a cargar los valores de la posicion y la velocidad para una nueva iteracion
	load<<<dimGrid, dimBlock>>>(r1dot_d,v1dot_d,dydt1_d,N);
	cudaDeviceSynchronize();
	load<<<dimGrid, dimBlock>>>(r2dot_d,v2dot_d,dydt2_d,N);
	cudaDeviceSynchronize();



	free(ne_h);
	free(n_h);
	free(phiFinal_h);
	free(Ex_h);
	free(Ey_h);
	cudaFree(jx_d);
	cudaFree(jy_d);
	cudaFree(yx_d);
	cudaFree(yy_d);
	cudaFree(r1dot_d);
	cudaFree(v1dot_d);
	cudaFree(r2dot_d);
	cudaFree(v2dot_d);
	cudaFree(ne_d);
	cudaFree(n_d);
	cudaFree(phiFinal_d);
	cudaFree(Ex_d);
	cudaFree(Ey_d);

}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void inicializardydx1Dydx2(float *dydx1_d, float *dydx2_d, float *k1_d, float *k2_d, float *k3_d, float * k4_d,
		float *l1_d, float *l2_d, float *l3_d, float *l4_d, float *f1_d,float *f2_d, int N){

	int Id = blockIdx.x * blockDim.x + threadIdx.x;
	if (Id < (2*N)) {
		dydx1_d[Id] = 0.0;
		dydx2_d[Id] = 0.0;
		k1_d[Id] = 0.0;
		k2_d[Id] = 0.0;
		k3_d[Id] = 0.0;
		k4_d[Id] = 0.0;
		l1_d[Id] = 0.0;
		l2_d[Id] = 0.0;
		l3_d[Id] = 0.0;
		l4_d[Id] = 0.0;
		f1_d[Id] = 0.0;
		f1_d[Id] = 0.0;
	}
}


__global__ void inicializarAux(float *rx_aux, float *ry_aux, float *vx_aux, float *vy_aux,  int N){

	int Id = blockIdx.x * blockDim.x + threadIdx.x;
		if (Id < (N)) {
			rx_aux[Id] = 0.0;
			ry_aux[Id] = 0.0;
			vx_aux[Id] = 0.0;
			vy_aux[Id] = 0.0;

		}
}



/////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void calculoPrimerasVariablesKLRK4(float *k1_d, float *l1_d, float *dydx1_d,
		float *dydx2_d,int N, float dt){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < 2*N){
		k1_d[i] = dt*dydx1_d[i];
		l1_d[i] = dt*dydx2_d[i];


	}

}

__global__ void calculoSegundasVariablesF1F2RK4(float *k1_d, float *l1_d,float *rx_d,
		float *ry_d,float *vx_d, float *vy_d,float *f1_d, float *f2_d, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N){
		f1_d[i]=(rx_d[i]+k1_d[i]/2.);
		f1_d[i+N]=(vx_d[i]+k1_d[i+N]/2.);
		f2_d[i]=(ry_d[i]+l1_d[i]/2.);
		f2_d[i+N]=(vy_d[i]+l1_d[i+N]/2.);

	}

}

__global__ void calculoDefinitivoRK4(float *k1_d,float *k2_d,float *k3_d,float *k4_d,float *l1_d,
		float *l2_d,float *l3_d,float *l4_d,float *rx_d, float *vx_d, float *ry_d, float *vy_d,
		int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N){
		rx_d[i] += k1_d[i] / 6. + k2_d[i] / 3. + k3_d[i] / 3. + k4_d[i] / 6.;
		vx_d[i] += (k1_d[i+N] / 6. + k2_d[i+N] / 3. + k3_d[i+N] / 3. + k4_d[i+N] / 6.);
		ry_d[i] += l1_d[i] / 6. + l2_d[i] / 3. + l3_d[i] / 3. + l4_d[i] / 6.;
		vy_d[i] += (l1_d[i+N]/ 6. + l2_d[i+N] / 3. + l3_d[i+N] / 3. + l4_d[i+N] / 6.);

	}


}
//////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////

void rungeKutta(float &t, float *rx_d, float *ry_d, float *vx_d, float *vy_d, float dt){

	int size = N * sizeof(float);
	int size1 = 2*N * sizeof(float);
	////////////////////////////////////////////////////////////////////////////////////
	float *dydx1_d, *dydx2_d;
	float *k1_d, *k2_d,*k3_d, *k4_d,*l1_d,*l2_d,*l3_d,*l4_d,*f1_d,*f2_d;
	float *rx_aux, *ry_aux, *vx_aux, *vy_aux;
	///////////////////////////////////////////////////////////////////////////////////
	error =cudaMalloc((void **) &dydx1_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria dydx1_d");
		exit(0);
	}
	error =cudaMalloc((void **) &dydx2_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria dydx2_d");
		exit(0);
	}
	error =cudaMalloc((void **) &k1_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria k1_d");
		exit(0);
	}
	error =cudaMalloc((void **) &k2_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria k2_d");
		exit(0);
	}
	error =cudaMalloc((void **) &k3_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria k3_d");
		exit(0);
	}
	error =cudaMalloc((void **) &k4_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria k4_d");
		exit(0);
	}
	error =cudaMalloc((void **) &l1_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria l1_d");
		exit(0);
	}
	error =cudaMalloc((void **) &l2_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria l2_d");
		exit(0);
	}
	error =cudaMalloc((void **) &l3_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria l3_d");
		exit(0);
	}
	error =cudaMalloc((void **) &l4_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria l4_d");
		exit(0);
	}
	error =cudaMalloc((void **) &f1_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria f1_d");
		exit(0);
	}
	error =cudaMalloc((void **) &f2_d, size1);
	if (error != cudaSuccess){
		printf("Error de memoria k2_d");
		exit(0);
	}
	error =cudaMalloc((void **) &rx_aux, size);
	if (error != cudaSuccess){
		printf("Error de memoria ry_aux");
		exit(0);
	}
	error =cudaMalloc((void **) &ry_aux, size);
	if (error != cudaSuccess){
		printf("Error de memoria ry_aux");
		exit(0);
	}
	error =cudaMalloc((void **) &vx_aux, size);
	if (error != cudaSuccess){
		printf("Error de memoria vx_aux");
		exit(0);
	}
	error =cudaMalloc((void **) &vy_aux, size);
	if (error != cudaSuccess){
		printf("Error de memoria vy_aux");
		exit(0);
	}


	//////////////////////////////////////////////////////////////////////////////////////
//	float blockSize = 1024;

	// cantidad de nloques  utilizar en el calculo.
	dim3 dimBlock(ceil(N / blockSize), 1, 1);
	dim3 dimBlock1(ceil(2*N / blockSize), 1, 1);
	dim3 dimGrid(blockSize, 1, 1);

	inicializardydx1Dydx2<<<dimGrid, dimBlock1>>>(dydx1_d, dydx2_d,k1_d,k2_d, k3_d, k4_d,
												 l1_d,l2_d,l3_d,l4_d,f1_d,f2_d, N);
	cudaDeviceSynchronize();

	inicializarAux<<<dimGrid, dimBlock>>>(rx_aux,ry_aux,vx_aux, vy_aux,N);
	cudaDeviceSynchronize();

	//Paso O.
	eval (rx_d, vx_d,ry_d, vy_d, dydx1_d, dydx2_d);


	calculoPrimerasVariablesKLRK4<<<dimGrid, dimBlock1>>>(k1_d,l1_d, dydx1_d, dydx2_d, N, dt);
	cudaDeviceSynchronize();

	calculoSegundasVariablesF1F2RK4<<<dimGrid, dimBlock>>>(k1_d, l1_d,rx_d,ry_d,vx_d, vy_d,f1_d, f2_d, N);
	cudaDeviceSynchronize();

	unLoad<<<dimGrid, dimBlock>>>(f1_d, rx_aux, vx_aux,N);
	cudaDeviceSynchronize();

	unLoad<<<dimGrid, dimBlock>>>(f2_d, ry_aux, vy_aux,N);
	cudaDeviceSynchronize();

	//paso 1.
	eval (rx_aux, vx_aux, ry_aux, vy_aux, dydx1_d, dydx2_d);

	calculoPrimerasVariablesKLRK4<<<dimGrid, dimBlock1>>>(k2_d,l2_d, dydx1_d, dydx2_d, N, dt);
	cudaDeviceSynchronize();

	calculoSegundasVariablesF1F2RK4<<<dimGrid, dimBlock>>>(k2_d, l2_d,rx_aux,ry_aux,vx_aux, vy_aux,f1_d, f2_d, N);
	cudaDeviceSynchronize();

	unLoad<<<dimGrid, dimBlock>>>(f1_d, rx_aux, vx_aux,N);
	cudaDeviceSynchronize();

	unLoad<<<dimGrid, dimBlock>>>(f2_d, ry_aux, vy_aux,N);
	cudaDeviceSynchronize();

	//paso 2.

	eval (rx_aux, vx_aux, ry_aux, vy_aux, dydx1_d, dydx2_d);

	calculoPrimerasVariablesKLRK4<<<dimGrid, dimBlock1>>>(k3_d,l3_d, dydx1_d, dydx2_d, N, dt);
	cudaDeviceSynchronize();

	calculoSegundasVariablesF1F2RK4<<<dimGrid, dimBlock>>>(k3_d, l3_d,rx_aux,ry_aux,vx_aux, vy_aux,f1_d, f2_d, N);
	cudaDeviceSynchronize();

	unLoad<<<dimGrid, dimBlock>>>(f1_d, rx_aux, vx_aux,N);
	cudaDeviceSynchronize();

	unLoad<<<dimGrid, dimBlock>>>(f2_d, ry_aux, vy_aux,N);
	cudaDeviceSynchronize();

	//paso 3.

	eval (rx_aux, vx_aux, ry_aux, vy_aux, dydx1_d, dydx2_d);

	calculoPrimerasVariablesKLRK4<<<dimGrid, dimBlock1>>>(k4_d,l4_d, dydx1_d, dydx2_d, N, dt);
	cudaDeviceSynchronize();


	calculoDefinitivoRK4<<<blockSize, dimBlock>>>(k1_d,k2_d,k3_d,k4_d,l1_d,
			l2_d,l3_d,l4_d,rx_d, vx_d, ry_d,vy_d, N);
	cudaDeviceSynchronize();
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	t+=dt;


	cudaFree(dydx1_d);
	cudaFree(dydx2_d);
	cudaFree(k1_d);
	cudaFree(k2_d);
	cudaFree(k3_d);
	cudaFree(k4_d);
	cudaFree(l1_d);
	cudaFree(l2_d);
	cudaFree(l3_d);
	cudaFree(l4_d);
	cudaFree(f1_d);
	cudaFree(f2_d);
	cudaFree(rx_aux);
	cudaFree(ry_aux);
	cudaFree(vx_aux);
	cudaFree(vy_aux);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////

//void Output(char *fn1, char *fn2, float t, float *rx_d, float *ry_d, float *vx_d, float *vy_d){
//
//	float *rx_h;
//	float *ry_h;
//	float *vx_h;
//	float *vy_h;
//	float *ne_h;
//	float *n_h;
//	float *phi_h;
//	float *Ex_h;
//	float *Ey_h;
//	float *ne_d;
//	float *n_d;
//	float *phi_d;
//	float *Ex_d;
//	float *Ey_d;
//	int *jx_d, *jy_d; // posiciones de la malla.
//	float *yx_d, *yy_d; // posiciones de la malla.
//
//
//	int size = N * sizeof(float);
//	int size_ne = C*C * sizeof(float);
//
//	rx_h = (float *) malloc(size);
//	ry_h = (float *) malloc(size);
//	vx_h = (float *) malloc(size);
//	vy_h = (float *) malloc(size);
//	ne_h = (float *) malloc(size_ne);
//	n_h = (float *) malloc(size_ne);
//	phi_h = (float *) malloc(size_ne);
//	Ex_h = (float *) malloc(size_ne);
//	Ey_h = (float *) malloc(size_ne);
//
//
//	cudaMalloc((void **) &jx_d, size);
//	cudaMalloc((void **) &jy_d, size);
//	cudaMalloc((void **) &yx_d, size);
//	cudaMalloc((void **) &yy_d, size);
//	cudaMalloc((void **) &ne_d, size_ne);
//	cudaMalloc((void **) &n_d, size_ne);
//	cudaMalloc((void **) &phi_d, size_ne);
//	cudaMalloc((void **) &Ex_d, size_ne);
//	cudaMalloc((void **) &Ey_d, size_ne);
//
//	float blockSize = 1024;
//	dim3 dimBlock(ceil(N / blockSize), 1, 1);
//	dim3 dimBlock2(ceil(C * C / blockSize), 1, 1);
//	dim3 dimGrid(blockSize, 1, 1);
//	dim3 dimGrid3(blockSize, blockSize, 1);
//
//
//	//posicion en x.
//	cudaMemcpy(rx_h, rx_d, size, cudaMemcpyDeviceToHost);
//	// posicion en y.
//	cudaMemcpy(ry_h, ry_d, size, cudaMemcpyDeviceToHost);
//	// velocidad en x.
//	cudaMemcpy(vx_h, vx_d, size, cudaMemcpyDeviceToHost);
//	//velocidad en y.
//	cudaMemcpy(vy_h, vy_d, size, cudaMemcpyDeviceToHost);
//
//	ofstream phase;
//	phase.open(fn1);
//	for(int i=0; i<N; i++)
//		phase<<rx_h[i]<<" "<<ry_h<<" "<<vx_h<<" "<<vx_h<<" "<<vy_h<<endl;
//	phase.close();
//
//	cudaMemcpy(rx_d, rx_h, size, cudaMemcpyHostToDevice);
//	// posicion en y.
//	cudaMemcpy(ry_d, ry_h, size, cudaMemcpyHostToDevice);
//	// velocidad en x.
//	cudaMemcpy(vx_d, vx_h, size, cudaMemcpyHostToDevice);
//	//velocidad en y.
//	cudaMemcpy(vy_d, vy_h, size, cudaMemcpyHostToDevice);
//
//	inicializacionVariables<<<dimGrid,dimBlock2>>>(ne_d, n_d, phi_d, Ex_d, Ey_d,C);
//	cudaDeviceSynchronize();
//
//	Densidad(ne_d, rx_d, ry_d, jx_d, jy_d, yx_d, yy_d, C, L,N);
//
//	normalizacionDensidadOutput<<<dimGrid,dimBlock2>>>(ne_d, n_d, N, C,L);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(n_h, n_d, size_ne, cudaMemcpyDeviceToHost);
//
//	poisson (n_h, phi_h);
//
//	cudaMemcpy(phi_d, phi_h, size_ne, cudaMemcpyHostToDevice);
//
//	//calculo del campo electrico
//	Electric2 ( phi_d, Ex_d, Ey_d);
//
//	//densidad sin normalizar.
//	cudaMemcpy(ne_h, ne_d, size_ne, cudaMemcpyDeviceToHost);
//	// densidades normalizadas para la funcion Eval.
//	cudaMemcpy(n_h, n_d, size_ne, cudaMemcpyDeviceToHost);
//	// campo electrico en x
//	cudaMemcpy(Ex_h, Ex_d, size_ne, cudaMemcpyDeviceToHost);
//	// campo electrico en y
//	cudaMemcpy(Ey_h, Ey_d, size_ne, cudaMemcpyDeviceToHost);
//	// calculo de phi.
//	cudaMemcpy(phi_h, phi_d, size_ne, cudaMemcpyDeviceToHost);
//
//	ofstream data;
//	phase.open(fn2);
//	for(int i=0; i<C*C; i++)
//		data<<ne_h[i]<<" "<<n_h<<" "<<Ex_h<<" "<<Ex_h<<" "<<phi_h<<endl;
//	phase.close();
//
//
//	free(rx_h);
//	free(ry_h);
//	free(vx_h);
//	free(vy_h);
//	free(ne_h);
//	free(n_h);
//	free(phi_h);
//	free(Ex_h);
//	free(Ey_h);
//	cudaFree(jx_d);
//	cudaFree(jy_d);
//	cudaFree(yx_d);
//	cudaFree(yy_d);
//	cudaFree(ne_d);
//	cudaFree(n_d);
//	cudaFree(phi_d);
//	cudaFree(Ex_d);
//	cudaFree(Ey_d);
//
//}
//////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main() {
	// Parametros
	L = 512.0;     // dominio de la solucion 0 <= x <= L (en longitudes de debye)
	N = 900000;            // Numero de particulas
	C = 512;            // Número de celdas.
	float vb = 3.0;    // velocidad promedio de los electrones
	float t = 0.0;

	//double kappa = 2. * M_PI / (L);
	float dt = 0.1;    // delta tiempo (en frecuencias inversas del plasma)
	float tmax = 40;  // cantidad de iteraciones. deben ser 100 mil segun el material
	int skip = int (tmax / dt) / 10; //saltos del algoritmo para reportar datos

	//0 Tesla, 1 780
	cudaSetDevice(0);


	//Inicializacion de la posición de las particulas en x, y y velocidad en vx,vy del host y dispositivo.
	float *rx_h, *ry_h, *vx_h, *vy_h;
	float *rx_d, *ry_d, *vx_d, *vy_d;
	float *dydt1_d, *dydt2_d;


	// longitudes de las matrices.
	int size = N * sizeof(float);
	int size1 = 2*N * sizeof(float);

	//reserva en memoria al host
	rx_h = (float *) malloc(size);
	ry_h = (float *) malloc(size);
	vx_h = (float *) malloc(size);
	vy_h = (float *) malloc(size);


	//reserva de memoria del dispositivo.
	error = cudaMalloc((void **) &rx_d, size);
	if (error != cudaSuccess){
		printf("Error asignando memoria a rx_d");
		exit(0);
	}

	error = cudaMalloc((void **) &ry_d, size);
	if (error != cudaSuccess){
		printf("Error asignando memoria a ry_d");
		exit(0);
	}

	error = cudaMalloc((void **) &vx_d, size);
	if (error != cudaSuccess){
		printf("Error asignando memoria a vx_d");
		exit(0);
	}
	error = cudaMalloc((void **) &vy_d, size);
	if (error != cudaSuccess){
		printf("Error asignando memoria a vy_d");
		exit(0);
	}
	error = cudaMalloc((void **) &dydt1_d, size1);
	if (error != cudaSuccess){
		printf("Error asignando memoria a dydt1_d");
		exit(0);
	}
	error = cudaMalloc((void **) &dydt2_d, size1);
	if (error != cudaSuccess){
		printf("Error asignando memoria a dydt2_d");
		exit(0);
	}

	curandState *devStates;
	error = cudaMalloc((void **) &devStates, N * sizeof(curandState));

	// Tamaño de los hilos y bloques a utilizar en el proceso de paralelizacion del algoritmo.
//	float blockSize = 1024;
	dim3 dimBlock(ceil(N / blockSize), 1, 1);
	dim3 dimBlock2(ceil(C * C / blockSize), 1, 1);
	dim3 dimBlock3(ceil(C * C / blockSize), ceil(C * C / blockSize), 1);
	dim3 dimGrid(blockSize, 1, 1);
	dim3 dimGrid3(blockSize, blockSize, 1);
	int seed = time(NULL);

	distribucionParticulas<<<blockSize, dimBlock>>>(rx_d, ry_d, vx_d, vy_d, N,
			devStates, vb, L, seed);
	cudaDeviceSynchronize();

//	// leer un archivo que contiene las posiciones y las velocidades de las particulas.
//
//	FILE *initFile;
//
//	initFile = fopen("/home/yen/Desktop/phase0.txt","r");
//
//	if(initFile==NULL){
//
//		printf("Archivo inexistente, verifique\n");
//
//	return (0);
//
//	}
//
//	for (int i = 0; i < N; ++i) {
//
//		fscanf(initFile,"%f %f %f %f", &rx_h[i],&ry_h[i],&vx_h[i],&vy_h[i]);
//
//	}
//	fclose(initFile);


	// paso de los datos desde la memoria del host hasta el dispositivo.
	////////////////////////////////////////////////////////////////////////////////////////////////////////
//	cudaMemcpy(rx_d, rx_h, size, cudaMemcpyHostToDevice);
//	// posicion en y.
//	cudaMemcpy(ry_d, ry_h, size, cudaMemcpyHostToDevice);
//	// velocidad en x.
//	cudaMemcpy(vx_d, vx_h, size, cudaMemcpyHostToDevice);
//	//velocidad en y
//	cudaMemcpy(vy_d, vy_h, size, cudaMemcpyHostToDevice);

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	// llamado de las funciones.

	for (int i = 1; i <= 1; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			rungeKutta( t, rx_d, ry_d, vx_d, vy_d, dt);
			escapeParticulas<<<dimGrid,dimBlock>>>(rx_d, ry_d, N,L);
			cudaDeviceSynchronize();
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//Paso de memoria del dispositivo a la memoria de host.

	//posicion en x.
	error = cudaMemcpy(rx_h, rx_d, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("Error copiando rx_d");
		exit(0);
	}
	// posicion en y.
	error = cudaMemcpy(ry_h, ry_d, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("Error copiando ry_d");
		exit(0);
	}
	// velocidad en x.
	error = cudaMemcpy(vx_h, vx_d, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("Error copiando rx_d");
		exit(0);
	}
	//velocidad en y.
	error = cudaMemcpy(vy_h, vy_d, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){
		printf("Error copiando rx_d");
		exit(0);
	}


	/////////////////////////////////////////IMPRIMIR DATOS ///////////////////////////////////////////////

	cout << "test" << endl;

	ofstream init;
	init.open("distribucionInicial.txt");
	for (int i = 0; i < N; i++) {
		init << rx_h[i] << " " << ry_h[i] << " " << vx_h[i] << " " << vy_h[i]<< endl;

	}

	init.close();

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*Liberar memoria*/
	free(rx_h);
	free(ry_h);
	free(vx_h);
	free(vy_h);
	cudaFree(devStates);
	cudaFree(rx_d);
	cudaFree(ry_d);
	cudaFree(vx_d);
	cudaFree(vy_d);
	return 0;
}


