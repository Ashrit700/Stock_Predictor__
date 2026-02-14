#include<stdio.h>
void main()
{
    int r;
    scanf("%d",&r);
    for(int i=0;i<=r/2;i++){
        for(int k=0;k<=(r/2-i);k++){
            printf(" ");
        }
        for(int j=0;j<=i;j++){
            printf("*");
            printf(" ");
        }
        printf("\n");

    }
    
}