/* rand example */
#include <stdio.h>
#include <math.h>

float logSig(const float & x)
{
  return 1 / ( 1 + (float)exp(-x)); 
}

int main ()
{
  int i = 20;
	int & j = i;
	printf ("j is equal to %d\n", j);
	j++; 
	printf ("After j++, i is equal to %d and j is equal to %d \n", i, j);
	j = 50;
	printf ("After j = 50, i is equal to %d and j is equal to %d \n", i, j);
	printf ("Log sig of 5 is equal to %f \n",logSig(5));
	
	
  return 0;
}