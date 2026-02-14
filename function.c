#include<stdio.h>
int Sum(int a,int b);
int main(){
    int a,b;
    scanf("%d %d",&a,&b);
    int result;
   result=Sum(a,b);
    printf("%d",result);
    return 0;

}
int Sum(int a,int b){
    int sum;
    sum=a+b;
    return sum;
}