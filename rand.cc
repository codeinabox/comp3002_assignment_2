/* rand example */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main ()
{
  /* initialize random generator */
  srand ( time(NULL) );

  /* generate some random numbers */
  printf ("A number between 0 and RAND_MAX (%d): %d\n", RAND_MAX, rand());
  printf ("A number between 0 and 99: %d\n", rand()%100);
  printf ("A number between 20 and 29: %d\n", rand()%10+20);
  printf ("A number between -1 and 1: %d\n", rand()%2-1);
	float randNum = (rand() / (float)RAND_MAX) * 2 - 1;
  printf ("A number between -1 and 1: %f\n", randNum);
	
  return 0;
}