//Analysis Type - Classification 
#include <stdio.h>
#include <conio.h>
#include <math.h>
#include <stdlib.h>
#include <windows.h>

#include "mex.h" 

#define DLLEXPORT  __declspec(dllexport) 

 
DLLEXPORT void StatNeuroResultsRAVM(double* input, int* ind, double* m);
void __cdecl mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ); 


double k_fdm_2_MLP_7_5_4_input_hidden_weights[5][7]=
{
 {-2.83527958159302e-1, 2.89934657209722e-1, 4.93363880849622e-1, -8.75724651080467e-1, 3.86000525771560e-1, 1.82146865118403e-1, -2.05279873575766e-2 },
 {2.97812861471928e-1, 1.42301799139965e-1, -7.58626265889516e-1, -7.82614296791588e-1, -1.02063543314337, 6.38774255373127e-1, 4.20075850130804e-1 },
 {3.43415056829368e-1, 3.84781629695125e-3, -6.33110359044437e-2, 5.02148106000819e-1, -1.27475623675377e-1, 1.60987105441913e-1, 4.26474307659468e-1 },
 {-4.36753073296557e-1, 1.76403838052590e-1, 1.71362386921746, -2.27483908688549e-1, 2.11207987348687, -8.89472360928051e-1, -5.05838802786523e-1 },
 {-2.02800528826114e-1, 1.48588755981540e-1, 4.63190700047228e-1, -1.70823353038845e-1, 2.77028658832428e-1, -2.42203737060510e-1, -2.03067848177509e-1 } 
};

double k_fdm_2_MLP_7_5_4_hidden_bias[5]={ -6.73686816815229e-2, 7.22496824891226e-2, 4.54283879143130e-1, -4.08285902648488e-1, -9.38365269570421e-2 };

double k_fdm_2_MLP_7_5_4_hidden_output_wts[4][5]=
{
 {-3.30540876410893e-1, 8.03483723644652e-1, 6.22202553803680e-2, -3.75709843965768e-1, 9.71766437797953e-2 },
 {-9.12979067767539e-2, -1.14577174142916, -1.63050195202422e-1, 8.56127075917471e-1, 4.10048928154471e-2 },
 {-4.03744658113511e-1, 1.71055867370482, -6.75479951013817e-2, -1.90478492220575, -3.15517074957689e-1 },
 {6.50730589902806e-1, 6.09347120302843e-2, -8.06634675624268e-1, 4.55050895033275e-1, 2.67965376280203e-1 }
};

double k_fdm_2_MLP_7_5_4_output_bias[4]={ -4.05667237897579, -2.27344473043056, 6.65589660176576e-1, -7.41574398847519e-1 };

double k_fdm_2_MLP_7_5_4_max_input[7]={ 9.70000000000000e+1, 3.10000000000000e+1, 1.00000000000000e+2, 2.50000000000000e+1, 6.70000000000000e+1, 1.87000000000000e+2, 1.05000000000000e+2 };

double k_fdm_2_MLP_7_5_4_min_input[7]={ 6.10000000000000e+1, 1.30000000000000e+1, 6.40000000000000e+1, 7.00000000000000, 1.80000000000000e+1, 1.60000000000000e+2, 5.00000000000000e+1 };

double k_fdm_2_MLP_7_5_4_input[7];
double k_fdm_2_MLP_7_5_4_hidden[5];
double k_fdm_2_MLP_7_5_4_output[4];

void k_fdm_2_MLP_7_5_4_ScaleInputs(double* input, double minimum, double maximum, int size)
{
 double delta;
 long i;
 for(i=0; i<size; i++)
 {
	delta = (maximum-minimum)/(k_fdm_2_MLP_7_5_4_max_input[i]-k_fdm_2_MLP_7_5_4_min_input[i]);
	input[i] = minimum - delta*k_fdm_2_MLP_7_5_4_min_input[i]+ delta*input[i];
 }
}

double k_fdm_2_MLP_7_5_4_logistic(double x)
{
  if(x > 100.0) x = 1.0;
  else if (x < -100.0) x = 0.0;
  else x = 1.0/(1.0+exp(-x));
  return x;
}

void k_fdm_2_MLP_7_5_4_Normalise(double out[],long length)
{
  long i, j;
  double sum = 0.0;
  for(i=0; i<length; i++)
  {
   if(out[i]>100)
   {
    out[i] = 1.0;
    j = i;
    for(i=0; i<length; i++)
    {
     if(i!=j) out[i] = 0.0;
    }
    break;
   }
   else out[i] = exp(out[i]);
  }
  for(i=0; i<length; i++)
  {
   sum += out[i];
  }
  for(i=0; i<length; i++)
  {
   out[i] = out[i]/sum;
  }
}

void k_fdm_2_MLP_7_5_4_ComputeFeedForwardSignals(double* MAT_INOUT,double* V_IN,double* V_OUT, double* V_BIAS,int size1,int size2,int layer)
{
  int row,col;
  for(row=0;row < size2; row++) 
    {
      V_OUT[row]=0.0;
      for(col=0;col<size1;col++)V_OUT[row]+=(*(MAT_INOUT+(row*size1)+col)*V_IN[col]);
      V_OUT[row]+=V_BIAS[row];
      if(layer==1) V_OUT[row] = k_fdm_2_MLP_7_5_4_logistic(V_OUT[row]);
   }
}

void k_fdm_2_MLP_7_5_4_RunNeuralNet_Classification () 
{
  k_fdm_2_MLP_7_5_4_ComputeFeedForwardSignals((double*)k_fdm_2_MLP_7_5_4_input_hidden_weights,k_fdm_2_MLP_7_5_4_input,k_fdm_2_MLP_7_5_4_hidden,k_fdm_2_MLP_7_5_4_hidden_bias,7, 5,0);
  k_fdm_2_MLP_7_5_4_ComputeFeedForwardSignals((double*)k_fdm_2_MLP_7_5_4_hidden_output_wts,k_fdm_2_MLP_7_5_4_hidden,k_fdm_2_MLP_7_5_4_output,k_fdm_2_MLP_7_5_4_output_bias,5, 4,1);
}

/* 
    Тело (реализация) заявленного выше прототипа функции.
    Производит некие действия и возвращает результат
  */
  
DLLEXPORT void StatNeuroResultsRAVM(double* input, int* ind, double* m)
{
	int i=0;
	int j=0;
	*m=3.e-300;
	
	for(j = 0; j <= 6; j++)
	{
			k_fdm_2_MLP_7_5_4_input[i] = input[i]; 
	}
	
    k_fdm_2_MLP_7_5_4_ScaleInputs(k_fdm_2_MLP_7_5_4_input,0,1,7);
	k_fdm_2_MLP_7_5_4_RunNeuralNet_Classification();
   //Normalise output if output activation is not Softmax;
 	k_fdm_2_MLP_7_5_4_Normalise(k_fdm_2_MLP_7_5_4_output,4);
	
	for(i=0;i<4;i++)
	{
      if(*m<k_fdm_2_MLP_7_5_4_output[i])
      {
        *m=k_fdm_2_MLP_7_5_4_output[i];
        *ind=i+1;
      }
	}
	return;
}

void /*__cdecl*/ mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	int i;
	int ok = 1;
	double *d; 
	memset(plhs, nlhs, 0);
	// magic
	if (7 == nrhs) {
		double m = 0.0; 
		int ind = 0;
		double *input = malloc(sizeof *input * nlhs);
		if (NULL == input) return;
		for (i = 0; i < nrhs; i++) {
			ok = 0;
			if (mxIsDouble(prhs[i])) {
				ok = 1;
				d = mxGetPr(prhs[i]);
				input[i] = *d;
			}
		}	
		if (ok) {		     
			StatNeuroResultsRAVM(input, &ind, &m); 
		    //if (good) 
			plhs[0] = mxCreateDoubleScalar(m); 
			plhs[1] = mxCreateDoubleScalar(ind+1);
			//nlhs = 2;
		}
		free(input);
	}
	
	
	return;
}
