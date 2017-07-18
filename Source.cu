#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;


 void display(int * sample, int len){

	for (int jiter = 0; jiter < len; jiter++){
		printf("%f,", sample[jiter]);
	}
	printf("\n");


}

 // Read CSV file to get data , num of rows and num of colums
int ** readCSV(string filename,int * rowNum,int * colNum){
	ifstream ifs("data.csv", ifstream::in);
	if (!ifs){
		printf("Failed");
	}
	vector<vector<int>> dataMat;
	int **data;
	string temp = "ttt";
	while (getline(ifs, temp)){

		if (temp.empty()) continue;
		vector<int> t1;
		istringstream ss(temp);
		string each;
		while (getline(ss, each, ','))
		{

			t1.push_back(stoi(each));
		}
		dataMat.push_back(t1);
		

	}

	int cols = dataMat[0].size();
	*colNum = cols;
	*rowNum = dataMat.size();
	data = new int*[dataMat.size()];
	
	for (int i = 0; i < dataMat.size(); i++){
		data[i] = new int[cols];

	}

	for (int i = 0; i < dataMat.size(); i++){
		for (int jiter = 0; jiter < cols; jiter++){
		data[i][jiter] = dataMat[i][jiter];
		}
	}
	return data;
}


// Kernel to get calculate euclidean distance
//Each thread computes for each data samples
__global__ void KMeans(int rowNum, int colNum,int k, float *centers,int * dataMat,int *label){
	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	
	if (idx > rowNum) return;
	float min = 999999999;
	
	int curLabel=-1;
	for (int iter=0; iter < k; iter++){
		float dist = 0;
		for (int jiter = 0; jiter < colNum; jiter++){
			dist += (dataMat[idx *colNum + jiter] - centers[iter *colNum + jiter])*(dataMat[idx*colNum + jiter] - centers[iter*colNum + jiter]);
		}
		
		if (dist < min){
			min = dist;
			curLabel = iter;
			
		} 
	}
//	printf("%d - %d - %f\n", idx, curLabel, min);
	label[idx] = curLabel;
}
/*__global__ void UpdateCenter(int rowNum, int colNum, int k, float **centers, int ** dataMat, int *label){
	__shared__ float *centerSum;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	for (int iter = 0; iter < colNum; iter++){
		int locIdx = label[idx] * colNum + iter;
	}

}*/
int main(){

	
	int rowLen, colNum;
	int**	dataMat = readCSV("data.csv",&rowLen,&colNum);
	
	int k = 4;
	int *clusterSize = new int[k]; 
	
	const unsigned int numThreadsPerClusterBlock = 128;
	const unsigned int numClusterBlocks =(rowLen + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

	int * dataMat_Dev, *label_Dev;
	float * kCenters_Dev,*kcenters;
	
	kcenters = new float[k*colNum];
	
	//Initial Random K data points
	
	for (int iter = 0; iter < k; iter++){
	
		int idx = rand() % rowLen;
		clusterSize[iter] = 0;
		for (int jiter = 0; jiter < colNum; jiter++){
			kcenters[iter * colNum + jiter] = dataMat[idx][jiter];
		}

	}

	//Display Initial K Centroids
	cout<<"Initial K Centroids\n";
	for (int iter = 0; iter < k; iter++){
		for (int jiter = 0; jiter < colNum; jiter++){
			printf("%f,", kcenters[iter * colNum + jiter]);
		}
		printf("\n");
	}
	

	int delta = 0;


	//Convert 2D matrix to 1D Matrix
	int *flatData = new int[rowLen*colNum];
	for (int iter = 0; iter < rowLen; iter++){
		for (int jiter = 0; jiter < colNum; jiter++){
			flatData[iter * colNum + jiter] = dataMat[iter][jiter];
		}
		
	}
	int *label = new int[rowLen];
	
	// Initialization 
	cudaMalloc(&dataMat_Dev, rowLen*colNum*sizeof(int));
	cudaMalloc(&kCenters_Dev, k*colNum*sizeof(float));
	cudaMalloc(&label_Dev, rowLen*sizeof(int));
	
	memset(label, 0, rowLen);

	cudaMemcpy(dataMat_Dev, flatData, rowLen*colNum*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kCenters_Dev, kcenters, k*colNum*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(label_Dev, label,rowLen*sizeof(int),cudaMemcpyHostToDevice);

	
	int *prevLabel = new int[rowLen];
	memset(prevLabel, -1, rowLen);
	int prevdelta = 2* rowLen;
	int change= 0;
	
	
	//Start Iteration : Displays Delta
	cout << "\nDelta\n";
	do{
		delta = 0;
		KMeans << < numClusterBlocks, numThreadsPerClusterBlock >> > (rowLen, colNum, k, kCenters_Dev, dataMat_Dev, label_Dev);
		cudaDeviceSynchronize();


		for (int iter = 0; iter < k; iter++){
			clusterSize[iter] = 1;
			for (int jiter = 0; jiter < colNum; jiter++){
				kcenters[iter *colNum+ jiter] = 0;
			}
		}

		cudaMemcpy(label, label_Dev, rowLen*sizeof(int), cudaMemcpyDeviceToHost);
	
		for (int i = 0; i < rowLen; i++){
			int centerId = label[i];
			if (centerId != prevLabel[i]) delta++;
		
			for (int j = 0; j < colNum; j++){
				kcenters[centerId * colNum + j] += dataMat[i][j];
			}
			clusterSize[centerId]++;
		}

		for (int i = 0; i < k; i++){
			for (int j = 0; j < colNum; j++){
				kcenters[i* colNum + j] /= clusterSize[i];
			}
		}
		cudaMemcpy(kCenters_Dev, kcenters, k*colNum*sizeof(float), cudaMemcpyHostToDevice); 
		memcpy(prevLabel, label, rowLen);
		
		cout << delta << "\n";
		change = prevdelta - delta;
		prevdelta = delta;
	} while (change>0);
	
	cout << "\nFinal Centroids\n";
	cudaMemcpy(kcenters, kCenters_Dev, k*colNum*sizeof(float), cudaMemcpyDeviceToHost);
	for (int iter = 0; iter < k; iter++){
		for (int jiter = 0; jiter < colNum; jiter++){
			printf("%f,",kcenters[iter * colNum+ jiter]);
		}
		printf( "\n");
	}
//	cudaMemcpy(label, label_Dev, rowLen*sizeof(int), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < rowLen; i++){
//		cout << label[i] << "\n";
//	}
	getchar();
	return 0;
}