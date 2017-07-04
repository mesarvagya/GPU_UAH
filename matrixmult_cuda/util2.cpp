#include <stdio.h>
#include <stdlib.h>
#include<sys/time.h>

void write_tm(const char * file_name, int width, double run_time) {
   FILE *fp;
   if((fp=fopen (file_name, "a+" ))==NULL) {
      printf("Not able to open %s file\n",file_name);
      exit(1);
   }
   fprintf(fp,"%d  %f\n",width,run_time);
}

