#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <assert.h>
#include <cstring>
#include <string>

using namespace std;

#define DEFINE_TYPE_D2 0
#define DEFINE_TYPE_D3 1
#define BLOCK_SIZE 64
#define STRING_2D string("2D")
#define STRING_3D string("3D")

// 2D
struct Struct2D
{
	int l_x;
	int l_y;
	int width;
	int height;
	float fixed_temp;
};

// 3D
struct Struct3D
{
	int l_x;
	int l_y;
	int l_z;
	int width;
	int height;
	int depth;
	float fixed_temp;
};

// Param
struct StructParam
{
	float fK;
	int iTime;
	int iWidth;
	int iHeight;
	int iDepth;
	float fStart;
	vector<Struct2D> v2D;
	vector<Struct3D> v3D;
};

// Read param
int ReadParam(const string& sFileName, StructParam& stParam)
{
	const int iSize = 1024;
	char sBuffer[iSize];
	ifstream fIn(sFileName.c_str());
	int iType;
	string sSep = ",";
	int iCount = 0;
	Struct2D st2D;
	Struct3D st3D;

	// file exist
	if( !fIn.is_open() )
	{
		return -1;
	}

	while (!fIn.eof())
	{
		fIn.getline(sBuffer, iSize);
		// comment and blank
		if (sBuffer[0] == '#' || strlen(sBuffer) == 0)
		{
			continue;
		}

		cout<<sBuffer<<endl;
		switch(iCount)
		{
			case 0:
				if(string(sBuffer) == STRING_2D)
				{
				    iType = DEFINE_TYPE_D2;
				}
				else
				{
				    iType = DEFINE_TYPE_D3;
				}
				break;
			case 1:
			    stParam.fK = atof(sBuffer);
			    break;
			case 2:
			    stParam.iTime = atoi(sBuffer);
			    break;
			case 3:
			    if( iType== DEFINE_TYPE_D2)
                {
                    sscanf(sBuffer, "%d,%d", &stParam.iWidth, &stParam.iHeight);
                }
                else
                {
                    sscanf(sBuffer, "%d,%d,%d", &stParam.iWidth, &stParam.iHeight, &stParam.iDepth);
                }
			    break;
			case 4:
			    stParam.fStart = atof(sBuffer);
			    break;
			default:
			{
				if( iType == DEFINE_TYPE_D2)
				{
					sscanf(sBuffer, "%d,%d,%d,%d,%f", &st2D.l_x, &st2D.l_y, &st2D.width, &st2D.height, &st2D.fixed_temp);
                    cout<<st2D.l_x<<" "<<st2D.l_y<<" "<<st2D.width<<" "<<st2D.height<<" "<<st2D.fixed_temp<<endl;
					stParam.v2D.push_back(st2D);
				}
				else if( iType == DEFINE_TYPE_D3 )
				{
					sscanf(sBuffer, "%d,%d,%d,%d,%d,%d,%f", &st3D.l_x, &st3D.l_y, &st3D.l_z, &st3D.width, &st3D.height, &st3D.depth, &st3D.fixed_temp);
					cout<<st3D.l_x<<" "<<st3D.l_y<<" "<<st3D.l_z<<" "<<st3D.width<<" "<<st3D.height<<" "<<st3D.depth<<" "<<st3D.fixed_temp<<endl;
					stParam.v3D.push_back(st3D);
				}
			}
				break;
		}
		iCount++;
	}
	cout<<"sFileName:"<<sFileName<<endl;
	fIn.close();
	return iType;
}

__global__ void New2Pre2D(float *pre_arr, const float *new_arr, dim3 dim)
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x + dim.x;
	pre_arr[idx] = new_arr[idx];
}

__global__ void DiffusionCUDA2D(const float *pre_arr, float *new_arr, dim3 dim, const float k)
{

	// Start from dim.x in case subscribes becomes negative
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x + dim.x;
	int idx_up = idx - dim.x;
	int idx_down = idx + dim.x;
	int idx_left = idx - 1;
	int idx_right = idx + 1;

	bool left_edge = idx % dim.x == 0;
	bool right_edge = (idx + 1) % dim.x == 0;
	bool top_edge = idx < 2 * dim.x;
	bool bottom_edge = (idx >= dim.x * dim.y) && (idx < dim.x * (dim.y + 1));

	if (idx < dim.x * (dim.y + 1))
	{
		new_arr[idx] = pre_arr[idx] + k * (pre_arr[idx_up] * (!top_edge) + pre_arr[idx_down] * (!bottom_edge) +
			pre_arr[idx_left] * (!left_edge) + pre_arr[idx_right] * (!right_edge) - 4 * pre_arr[idx] +
			pre_arr[idx] * left_edge + pre_arr[idx] * right_edge + pre_arr[idx] * top_edge + pre_arr[idx] * bottom_edge);
	}
}

__global__ void RecoverFixed2D(float *new_arr, dim3 dim, int x, int y, int width, int height, float temp)
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int offset_x = idx % width;
	int offset_y = idx / width;

	if (idx < width * height)
	{
		new_arr[(y + offset_y) * dim.x + x + offset_x] = temp;
	}
}


__global__ void New2Pre3D(float *pre_arr, const float *new_arr, dim3 dim) {
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x + dim.x * dim.y;
	pre_arr[idx] = new_arr[idx];
}


__global__ void DiffusionCUDA3D(const float *pre_arr, float *new_arr, dim3 dim, const float k)
{

	// Start from dim.x in case subscribes becomes negative
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x + dim.x * dim.y;
	int idx_up = idx - dim.x;
	int idx_down = idx + dim.x;
	int idx_left = idx - 1;
	int idx_right = idx + 1;
	int idx_inner = idx - dim.x * dim.y;
	int idx_outer = idx + dim.x * dim.y;

	bool left_plane = idx % dim.x == 0;
	bool right_plane = (idx + 1) % dim.x == 0;
	bool upper_plane = idx % (dim.x * dim.y) < dim.x;
	bool bottom_plane = idx % (dim.x * dim.y) >= dim.x * (dim.y - 1);
	bool inner_plane = idx < 2 * dim.x * dim.y;
	bool outer_plane = (idx >= dim.x * dim.y * dim.z) && (idx < dim.x * dim.y * (dim.z + 1));

	if (idx < dim.x * dim.y * (dim.z + 1))
	{
		new_arr[idx] = pre_arr[idx] + k * (pre_arr[idx_up] * (!upper_plane) + pre_arr[idx_down] * (!bottom_plane) +
			pre_arr[idx_left] * (!left_plane) + pre_arr[idx_right] * (!right_plane) +
			pre_arr[idx_inner] * (!inner_plane) + pre_arr[idx_outer] * (!outer_plane) - 
			6 * pre_arr[idx] + pre_arr[idx] * left_plane + pre_arr[idx] * right_plane + 
			pre_arr[idx] * upper_plane + pre_arr[idx] * bottom_plane + pre_arr[idx] * inner_plane + 
			pre_arr[idx] * outer_plane);
	}
}


__global__ void RecoverFixed3D(float *new_arr, dim3 dim, int x, int y, int z, int width, int height, int depth, float temp)
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int offset_x = (idx % (height * width)) % width;
	int offset_y = (idx % (height * width)) / width;
	int offset_z = idx / (height * width);

	if (idx < width * height * depth)
	{
		new_arr[(z + offset_z) * dim.x * dim.y + (y + offset_y) * dim.x + x + offset_x] = temp;
	}
}


Struct2D * parseVec(const vector<Struct2D> &vec)
{
	Struct2D * p = (Struct2D*)malloc(sizeof(Struct2D) * vec.size());
	for (int i = 0; i < vec.size(); ++i)
	{
		p[i] = vec[i];
	}
	return p;
}

Struct3D * parseVec(const vector<Struct3D> &vec)
{
	Struct3D * p = (Struct3D*)malloc(sizeof(Struct3D) * vec.size());
	for (int i = 0; i < vec.size(); ++i)
	{
		p[i] = vec[i];
	}
	return p;
}

void usage(const string& sBin)
{
	cout<<sBin<<" 2d.conf"<<endl;
}

// Deal 2D
void Deal2D(StructParam& stParam, const string& sFile)
{
	Struct2D *tvec = parseVec(stParam.v2D);

	// graph
	dim3 dim(stParam.iWidth, stParam.iHeight, 1);
	unsigned int iLength = dim.x * dim.y;
	unsigned int iMemSize = sizeof(float) * (iLength + dim.x);

	float *pre_graph = reinterpret_cast<float *>(malloc(iMemSize));
	float *new_graph = reinterpret_cast<float *>(malloc(iMemSize));
	for (int i = 0; i < dim.y + 1; ++i)
	{
		for (int j = 0; j < dim.x; ++j)
		{
			pre_graph[i * dim.x + j] = stParam.fStart;
		}
	}

	for(int k = 0; k < stParam.v2D.size(); k++)
	{
		Struct2D& st2D = stParam.v2D[k];
		for(int i = st2D.l_y + 1; i < st2D.l_y + 1 + st2D.height; i++)
		{
			for(int j = st2D.l_x; j < st2D.l_x + st2D.width; j++)
			{
				pre_graph[i * dim.x + j] = st2D.fixed_temp;
			}
		}
	}

	cout<<11<<endl;

	int iVecSize = stParam.v2D.size();
	float *d_pre, *d_new;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pre), iMemSize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_new), iMemSize));
	checkCudaErrors(cudaMemcpy(d_pre, pre_graph, iMemSize, cudaMemcpyHostToDevice));

	for (int t = 0; t < stParam.iTime; ++t)
	{
		DiffusionCUDA2D <<< (iLength + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (d_pre, d_new, dim, stParam.fK);
		for (int i = 0; i < iVecSize; ++i)
		{
			RecoverFixed2D <<< (stParam.v2D[i].width * stParam.v2D[i].height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (d_new, dim,
					stParam.v2D[i].l_x, stParam.v2D[i].l_y + 1, stParam.v2D[i].width, stParam.v2D[i].height, stParam.v2D[i].fixed_temp);
		}
		New2Pre2D <<< (iLength + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (d_pre, d_new, dim);
	}
	checkCudaErrors(cudaMemcpy(new_graph, d_new, iMemSize, cudaMemcpyDeviceToHost));

	cout<<111<<endl;
	// Write file
	ofstream fOut;
	fOut.open(sFile.c_str(), ios::out);
	assert(fOut);
	for (int i = 1; i < stParam.iHeight + 1; ++i)
	{
		for (int j = 0; j < stParam.iWidth - 1; ++j)
		{
			fOut << new_graph[i * stParam.iWidth + j] << ", ";
		}
		if (i != stParam.iHeight)
		{
			fOut << new_graph[i * stParam.iWidth + stParam.iWidth - 1] << endl;
		}
		else
		{
			fOut << new_graph[i * stParam.iWidth + stParam.iWidth - 1];
		}
	}

	// clean
	fOut.close();
	free(new_graph);
	free(pre_graph);
	checkCudaErrors(cudaFree(d_pre));
	checkCudaErrors(cudaFree(d_new));
}



// Function to manipulate 3D heat diffusion
void Deal3D(StructParam& stParam, const string& sFile)
{
	Struct3D * tvec = parseVec(stParam.v3D);
	dim3 dim(stParam.iWidth, stParam.iHeight, stParam.iDepth);
	unsigned int iLength = dim.x * dim.y * dim.z;
	unsigned int iMemSize = sizeof(float) * (iLength + dim.x * dim.y);

	float *pre_graph = reinterpret_cast<float *>(malloc(iMemSize));
	float *new_graph = reinterpret_cast<float *>(malloc(iMemSize));

	for (int m = 0; m < stParam.iDepth; ++m)
	{
		for (int i = 0; i < stParam.iHeight; ++i)
		{
			for (int j = 0; j < stParam.iWidth; ++j)
			{
				pre_graph[m * stParam.iHeight * stParam.iWidth + i * stParam.iWidth + j] = stParam.fStart;
			}
		}
	}

	for(int k = 0; k < stParam.v3D.size(); k++)
	{
		Struct3D& st3D = stParam.v3D[k];
		for (int m = st3D.l_z + 1; m < st3D.l_z + st3D.depth + 1; ++m)
		{
			for (int i = st3D.l_y; i < st3D.l_y + st3D.height; ++i)
			{
				for (int j = st3D.l_x; j < st3D.l_x + st3D.width; ++j)
				{
					pre_graph[m * stParam.iHeight * stParam.iWidth + i * stParam.iWidth + j] = st3D.fixed_temp;
				}
			}
		}
	}
	
	int iVecSize = stParam.v3D.size();
	float *d_pre, *d_new;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_pre), iMemSize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_new), iMemSize));
	checkCudaErrors(cudaMemcpy(d_pre, pre_graph, iMemSize, cudaMemcpyHostToDevice));
	for (int t = 0; t < stParam.iTime; ++t)
	{
		DiffusionCUDA3D <<< (iLength + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (d_pre, d_new, dim, stParam.fK);
		for (int i = 0; i < iVecSize; ++i)
		{
			RecoverFixed3D <<< (stParam.v3D[i].width * stParam.v3D[i].height * stParam.v3D[i].depth + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(
				d_new, dim, stParam.v3D[i].l_x, stParam.v3D[i].l_y, stParam.v3D[i].l_z + 1, stParam.v3D[i].width, stParam.v3D[i].height,
						stParam.v3D[i].depth, stParam.v3D[i].fixed_temp);
		}
		New2Pre3D <<< (iLength + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>> (d_pre, d_new, dim);
	}
	checkCudaErrors(cudaMemcpy(new_graph, d_new, iMemSize, cudaMemcpyDeviceToHost));

	// Write file
	ofstream fOut;
	fOut.open(sFile.c_str(), ios::out);
	for (int m = 1; m < stParam.iDepth + 1; ++m)
	{
		for (int i = 0; i < stParam.iHeight; ++i)
		{
			for (int j = 0; j < stParam.iWidth - 1; ++j)
			{
				fOut << new_graph[m * stParam.iHeight * stParam.iWidth + i * stParam.iWidth + j] << ", ";
			}
			if (m == stParam.iDepth && i == stParam.iHeight - 1)
			{
				fOut << new_graph[m * stParam.iHeight * stParam.iWidth + i * stParam.iWidth + stParam.iWidth - 1];
			}
			else
			{
				fOut << new_graph[m * stParam.iHeight * stParam.iWidth + i * stParam.iWidth + stParam.iWidth - 1] << endl;
			}
		}
		if (m != stParam.iDepth)
		{
			fOut << endl;
		}
	}

	fOut.close();
	free(new_graph);
	free(pre_graph);
	checkCudaErrors(cudaFree(d_pre));
	checkCudaErrors(cudaFree(d_new));
}

int main(int argc, char** argv)
{
	if( argc != 2 )
	{
		usage(argv[0]);
		return -1;
	}

	// Read Parameters
	StructParam stParam;
	string sFileName = argv[1];
	string sOutFile = "out" + sFileName + ".csv";
	int iType = ReadParam(sFileName, stParam);

	cout<<stParam.iTime<<" "<<stParam.fK<<" "<<stParam.iWidth<<" "<<stParam.iHeight<<" "<<stParam.iDepth<<" "<<stParam.fStart<<" "<<stParam.v2D.size()<<" "<<stParam.v3D.size()<<endl;
	switch(iType)
	{
		case DEFINE_TYPE_D2:
			Deal2D(stParam, sOutFile);
			break;
		case DEFINE_TYPE_D3:
			Deal3D(stParam, sOutFile);
			break;
		default:
			cout<<"iType:"<<iType<<endl;
			break;
	}
	return 0;
}